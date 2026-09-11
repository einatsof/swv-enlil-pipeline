"""Exercise the real NetCDF -> field + track artifact pipeline on a tiny run."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

import netCDF4 as nc
import numpy as np

from extract import extract_run, quantize_dp, quantize_ratio, read_frame
from regions import AU_KM, TrackingConfig


class ExtractionTrackingTests(unittest.TestCase):
    def write_run(self, folder):
        # NOAA pv stores latitude north -> south, unlike longitude/radius.
        lon, lat, rad = np.arange(12) * 30., np.array([20., 0., -20.]), np.arange(12) * 0.1 + 0.1
        epoch = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
        for frame in range(2):
            with nc.Dataset(folder / f'pv-tim.{frame * 3:04d}.nc', 'w') as ds:
                ds.createDimension('time', 1)
                for name, data in (('longitude', lon), ('latitude', lat), ('radius', rad)):
                    ds.createDimension(name, len(data))
                    ds.createVariable(name, 'f8', (name,))[:] = data
                ds.createVariable('time', 'f8', ('time',))[:] = epoch + frame * 10800
                dims = ('time', 'longitude', 'latitude', 'radius')
                dp = np.zeros((12, 3, 12))
                dp[1:3, 1, 1 + frame * 3:3 + frame * 3] = 0.251
                # An off-slice cloud should also have a track in the manifest.
                dp[7, 2, 1 + frame * 3:3 + frame * 3] = 1
                ds.createVariable('DP', 'f8', dims)[0] = dp
                ds.createVariable('Density', 'f8', dims)[0] = np.full(dp.shape, 5.)
                ds.createVariable('Vr', 'f8', dims)[0] = np.full(dp.shape, 0.1 * AU_KM / 3600)
        return dp

    def write_run_with_a_cone(self, folder):
        """A run whose cone resolves one frame later than its cloud is labelled.

        The cloud is parked on the inner shells for both frames so the only thing
        moving is the attribution wait.
        """
        lon, lat, rad = np.arange(12) * 30., np.array([20., 0., -20.]), np.arange(12) * 0.1 + 0.1
        epoch = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
        for frame in range(2):
            with nc.Dataset(folder / f'pv-tim.{frame * 3:04d}.nc', 'w') as ds:
                ds.createDimension('time', 1)
                for name, data in (('longitude', lon), ('latitude', lat), ('radius', rad)):
                    ds.createDimension(name, len(data))
                    ds.createVariable(name, 'f8', (name,))[:] = data
                ds.createVariable('time', 'f8', ('time',))[:] = epoch + frame * 10800
                dims = ('time', 'longitude', 'latitude', 'radius')
                dp = np.zeros((12, 3, 12))
                dp[1:3, 1, 1:3] = 1.0
                ds.createVariable('DP', 'f8', dims)[0] = dp
                ds.createVariable('Density', 'f8', dims)[0] = np.full(dp.shape, 5.)
                ds.createVariable('Vr', 'f8', dims)[0] = np.full(dp.shape, 1.0)
        # No evo.earth.nc, so Earth falls back to grid lon 180 and -135 puts the
        # cone's wedge (+-20 deg about grid lon 45) over the cloud at lon 1-2.
        # A radial cell is 0.1 AU here, which 1500 km/s clears in 2.8 h — long
        # enough that frame 0000 is skipped by the wait and 0003 resolves.
        (folder / 'metadata.json').write_text(json.dumps({
            'cme_time': '2026-09-01T00:00', 'cme_latitude': '0', 'cme_longitude': '-135',
            'cme_cone_half_angle': '20', 'cme_radial_velocity': '1500'}))

    def test_a_cone_is_published_on_every_frame_of_its_cloud(self):
        """`coneIdxs` must describe the cloud, not the extractor's loop progress.

        Without ConeAttributor.backfill the first frame ships an unnamed blob —
        live on 20260910_58504 that was three frames, 3 h, during which the
        monitor drew a real cloud with no CME attached.
        """
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / 'raw', Path(tmp) / 'out'
            raw.mkdir()
            self.write_run_with_a_cone(raw)
            with redirect_stdout(io.StringIO()):
                meta = extract_run(str(raw), str(out), volumes=False)
            # The lag this repairs: identified on the second frame, not the first.
            self.assertEqual(meta['cmes'][0]['tracking']['frame'], '0003')
            manifest = json.loads((out / meta['blobTracking']['manifest']).read_text())
            owners = [[r['coneIdxs'] for r in f['regions']] for f in manifest['frames']]
            self.assertEqual(owners, [[[0]], [[0]]])

    def test_artifacts_track_full_volume_and_export_little_endian_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, out = root / 'raw', root / 'out'
            raw.mkdir()
            last_dp = self.write_run(raw)
            # Deliberately omit evo.earth.nc: the documented fallback must work.
            with redirect_stdout(io.StringIO()):
                meta = extract_run(str(raw), str(out), run_id='synthetic', volumes=False)
            tracking = meta['blobTracking']
            manifest = json.loads((out / tracking['manifest']).read_text())
            self.assertEqual(manifest['runId'], 'synthetic')
            self.assertEqual([f['frame'] for f in manifest['frames']], ['0000', '0003'])
            self.assertEqual(tracking['labelDtype'], 'uint16')
            first, second = manifest['frames']
            self.assertEqual(len(first['regions']), 2)
            self.assertEqual([r['trackId'] for r in first['regions']], [r['trackId'] for r in second['regions']])
            self.assertEqual([r['sliceCellCount'] for r in second['regions']], [4, 0])
            labels = np.fromfile(out / second['labelsFile'], dtype='<u2').reshape(12, 12)
            self.assertEqual(np.count_nonzero(labels), 4)
            self.assertEqual(labels[1, 4], second['regions'][0]['trackId'])
            self.assertEqual((out / second['labelsFile']).stat().st_size, 12 * 12 * 2)
            self.assertFalse(list(out.glob('vol_*')))
            self.assertEqual((out / 'sldp_0003.bin').read_bytes(), quantize_dp(last_dp[:, 1, :]).tobytes())
            self.assertEqual((out / 'slice_0003.bin').read_bytes(), quantize_ratio(np.ones((12, 12))).tobytes())
            saved_meta = json.loads((out / 'meta.json').read_text())
            self.assertEqual(saved_meta['blobTracking'], tracking)

    def test_changed_threshold_has_distinct_artifact_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / 'raw', Path(tmp) / 'out'
            raw.mkdir()
            self.write_run(raw)
            with redirect_stdout(io.StringIO()):
                original = extract_run(str(raw), str(out), volumes=False)
                changed = extract_run(str(raw), str(out), volumes=False,
                                      tracking_config=TrackingConfig(dp_threshold=0.3))
            self.assertNotEqual(original['blobTracking']['manifest'], changed['blobTracking']['manifest'])
            self.assertTrue((out / original['blobTracking']['manifest']).exists())
            manifest = json.loads((out / changed['blobTracking']['manifest']).read_text())
            self.assertEqual(len(manifest['frames'][0]['regions']), 1)

    def test_netcdf_fill_values_do_not_become_clouds_or_prediction_speeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            self.write_run(raw)
            path = raw / 'pv-tim.0000.nc'
            with nc.Dataset(path, 'a') as ds:
                ds.variables['DP'][0, 0, 0, 0] = np.ma.masked
                ds.variables['Vr'][0, 0, 0, 0] = np.ma.masked
            frame = read_frame(path)
            self.assertTrue(np.isnan(frame['dp'][0, 0, 0]))
            self.assertTrue(np.isnan(frame['vr'][0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
