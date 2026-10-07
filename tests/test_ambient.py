"""The ambient product: suball.py extraction, nomads.py helpers, per-folder R2 rules.

The synthetic file mirrors the real `suball.nc` layout (verified on runs 58518,
58545 and the 2026-10-07 ambient run): int16 planes packed against per-variable
`<tag>_min/_max` attributes, x = radius in metres, z = longitude in radians,
`time` in seconds from REFDATE_CAL. Only the frame count is shrunk.

The A/B test at the bottom runs against real data when SWV_AB_DIR points at a
folder holding `58545.suball.nc` + pv-tim frames of run 20261006_58545; it is
how the reader was validated and it is skipped everywhere else.
"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone

import netCDF4 as nc
import numpy as np

import nomads
import pipeline
from extract import VR_SCALE, quantize, quantize_ratio
from regions import AU_KM
from suball import ambient_run_id, decode, extract_ambient_run

AU_M = AU_KM * 1000.0


def pack(values, lo, hi):
    return np.round((values - lo) / (hi - lo) * 65535.0 - 32768.0).astype(np.int16)


def write_suball(path, frames=3, z=180, x=512, density=None, speed_ms=400_000.0):
    """A tiny suball.nc: uniform density except a 2x stripe, uniform speed."""
    ds = nc.Dataset(path, 'w')
    for name, n in (('x', x), ('y', 4), ('z', z), ('t', frames), ('earth_t', 5)):
        ds.createDimension(name, n)
    # Real files: x0 = 0.1016 AU in 0.003125 AU cells, so 8-cell blocks centre
    # on the published 0.1125 + 0.025 k; lon centres at 1, 3, 5 ... degrees.
    ds.createVariable('x_coord', 'f4', ('x',))[:] = (0.1 + 0.003125 * (np.arange(x) + 0.5)) * AU_M
    ds.createVariable('y_coord', 'f4', ('y',))[:] = np.radians([31, 89, 91, 149])
    ds.createVariable('z_coord', 'f4', ('z',))[:] = np.radians(1 + 2 * np.arange(z))
    ds.createVariable('time', 'f4', ('t',))[:] = (np.arange(frames) - 1) * 3600.0
    if density is None:
        density = np.full((frames, z, x), 1e-20)
        density[:, 10:12, :] *= 2.0                     # one published lon cell (index 5) at 2x
    dd = ds.createVariable('dd13_3d', 'i2', ('t', 'z', 'x'))
    lo, hi = 0.5e-20, 4e-20
    dd.setncattr('dd13_min', np.float32(lo)); dd.setncattr('dd13_max', np.float32(hi))
    dd[:] = pack(density, lo, hi)
    vv = ds.createVariable('vv13_3d', 'i2', ('t', 'z', 'x'))
    vv.setncattr('vv13_min', np.float32(200_000.0)); vv.setncattr('vv13_max', np.float32(1_200_000.0))
    vv[:] = pack(np.full((frames, z, x), speed_ms), 200_000.0, 1_200_000.0)
    ds.createVariable('Earth_X2', 'f4', ('earth_t',))[:] = np.radians(90 - 6.7)
    ds.createVariable('Earth_X3', 'f4', ('earth_t',))[:] = np.pi
    ds.setncattr('REFDATE_CAL', '2026-10-07T00:00:00')
    ds.setncattr('model_run_id', '0')
    ds.close()


class AmbientExtractionTest(unittest.TestCase):
    def setUp(self):
        # The synthetic file is 3 frames; the real floor is MIN_FRAMES.
        patcher = mock.patch('suball.MIN_FRAMES', 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def extract(self, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src, out = Path(tmp.name, 'suball.nc'), Path(tmp.name, 'out')
        write_suball(src, **kw)
        with redirect_stdout(io.StringIO()):
            meta = extract_ambient_run(str(src), str(out), source={'dir': 'wsa_enlil.20261007'})
        return meta, out

    def test_meta_is_ambient_and_withholds_volume_fields(self):
        meta, _ = self.extract()
        self.assertEqual(meta['runId'], 'amb_20261007_00')
        self.assertEqual(meta['kind'], 'ambient')
        self.assertEqual(meta['capabilities'], {'volumes': False, 'tracking': False})
        self.assertEqual(meta['cmes'], [])
        # The monitor gates volumes/tilt on grid.lat — it must be absent.
        self.assertNotIn('lat', meta['grid'])
        self.assertNotIn('eclipticLatIndex', meta['grid'])
        self.assertNotIn('blobTracking', meta)
        self.assertEqual(meta['source']['product'], 'nomads-suball')
        self.assertEqual(meta['source']['dir'], 'wsa_enlil.20261007')
        self.assertAlmostEqual(meta['earth']['lon'], 180.0, places=3)
        self.assertAlmostEqual(meta['earth']['lat'], 6.7, places=3)

    def test_grid_folds_exactly_onto_the_published_cone_grid(self):
        meta, _ = self.extract()
        g = meta['grid']
        self.assertEqual((g['lon']['n'], g['rad']['n']), (90, 64))
        self.assertAlmostEqual(g['lon']['min'], 2.0, places=4)
        self.assertAlmostEqual(g['lon']['max'], 358.0, places=4)
        self.assertAlmostEqual(g['rad']['min'], 0.1125, places=5)
        self.assertAlmostEqual(g['rad']['max'], 1.6875, places=5)

    def test_frames_times_and_files(self):
        meta, out = self.extract()
        self.assertEqual(meta['frames'], ['0000', '0001', '0002'])
        self.assertEqual(meta['times'], ['2026-10-06T23:00:00Z', '2026-10-07T00:00:00Z',
                                         '2026-10-07T01:00:00Z'])
        self.assertEqual(meta['rundate'], '2026-10-07T00:00:00Z')
        for num in meta['frames']:
            for field in ('slice', 'slvr', 'sldp'):
                self.assertEqual(os.path.getsize(out / f'{field}_{num}.bin'), 90 * 64)
        self.assertEqual(os.path.getsize(out / 'line.bin'), 3 * 64 * 3)
        self.assertFalse(list(out.glob('vol*')))
        self.assertFalse(list(out.glob('labels_*')))
        with open(out / 'meta.json') as f:
            self.assertEqual(json.load(f)['runId'], meta['runId'])

    def test_values_decode_resample_and_quantize_like_cone_runs(self):
        _, out = self.extract()
        ratio = np.fromfile(out / 'slice_0001.bin', dtype=np.uint8).reshape(90, 64)
        vr = np.fromfile(out / 'slvr_0001.bin', dtype=np.uint8).reshape(90, 64)
        dp = np.fromfile(out / 'sldp_0001.bin', dtype=np.uint8)
        # Baseline = azimuthal median, so the uniform background is 1x and the
        # stripe 2x; int16 packing costs well under one u8 level.
        self.assertTrue(np.all(np.abs(ratio[0].astype(int) - quantize_ratio(np.array(1.0))) <= 1))
        self.assertTrue(np.all(np.abs(ratio[5].astype(int) - quantize_ratio(np.array(2.0))) <= 1))
        # vv is m/s in the file; slvr is km/s like the cone runs.
        self.assertTrue(np.all(np.abs(vr.astype(int) - quantize(np.array(400.0), *VR_SCALE)) <= 1))
        # No CMEs: sldp exists (hasField needs it) and is all zero.
        self.assertFalse(dp.any())

    def test_refuses_an_unexpected_grid(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name, 'suball.nc')
        write_suball(src, z=90)
        with self.assertRaisesRegex(ValueError, 'unexpected suball grid'):
            extract_ambient_run(str(src), str(Path(tmp.name, 'out')))

    def test_decode_round_trips_the_packing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name, 'suball.nc')
        write_suball(src)
        ds = nc.Dataset(src)
        ds.set_auto_mask(False)
        try:
            got = decode(ds, 'dd13_3d', 0)
            self.assertLess(abs(got[0, 0] - 1e-20) / 1e-20, 1e-3)
            self.assertLess(abs(got[10, 0] - 2e-20) / 2e-20, 1e-3)
            self.assertEqual(ambient_run_id(ds), 'amb_20261007_00')
        finally:
            ds.close()


class NomadsHelpersTest(unittest.TestCase):
    def test_day_dirs_newest_first(self):
        html = ('<a href="wsa_enlil.20261006/">wsa_enlil.20261006/</a> 06-Oct-2026 22:07 -\n'
                '<a href="wsa_enlil.20261007/">wsa_enlil.20261007/</a> 07-Oct-2026 04:07 -\n'
                '<a href="/pub/data/">Parent Directory</a>')
        self.assertEqual(nomads.day_dirs(html), ['wsa_enlil.20261007/', 'wsa_enlil.20261006/'])

    def test_parse_ncmes(self):
        self.assertEqual(nomads.parse_ncmes(' &cmes\n  ncmes=0,\n /'), 0)
        self.assertEqual(nomads.parse_ncmes('NCMES = 4'), 4)
        self.assertIsNone(nomads.parse_ncmes('no such key'))

    def test_ncmes_from_inputs_finds_the_suffixed_enlil_in(self):
        # The real member is `enlil.in.<pid>` under a deep job path.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            for name, text in (('lfs/tmp/wsa_enlil_bkgrnd.1/wsa2bc.in.594940', 'x=1'),
                               ('lfs/tmp/wsa_enlil_bkgrnd.1/enlil.in.594940', ' ncmes=0,\n')):
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        self.assertEqual(nomads.ncmes_from_inputs(buf.getvalue()), 0)

    def test_is_settled(self):
        now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
        self.assertFalse(nomads.is_settled(now - timedelta(minutes=5), now))
        self.assertTrue(nomads.is_settled(now - timedelta(minutes=nomads.SETTLE_MINUTES), now))


class R2LayoutTest(unittest.TestCase):
    def test_run_id_patterns_are_disjoint(self):
        for rid, kind in (('20261006_58545', 'cone'), ('amb_20261007_00', 'ambient')):
            other = 'ambient' if kind == 'cone' else 'cone'
            self.assertTrue(pipeline.RUN_ID_RE[kind].match(rid))
            self.assertFalse(pipeline.RUN_ID_RE[other].match(rid))

    def test_prune_keeps_newest_per_product_and_only_touches_its_own_runs(self):
        cone = [f'enlil/cone/2026100{d}_5854{d}/' for d in range(1, 5)]
        stray = ['enlil/cone/scratch/', 'enlil/cone/amb_20261007_00/']
        got = pipeline.runs_to_prune(cone + stray, 'cone', keep=2)
        self.assertEqual(got, cone[:2])

    def test_prune_never_deletes_the_published_run(self):
        amb = [f'enlil/ambient/amb_2026100{d}_00/' for d in range(1, 5)]
        got = pipeline.runs_to_prune(amb, 'ambient', keep=2, protected='amb_20261001_00')
        self.assertEqual(got, ['enlil/ambient/amb_20261002_00/'])

    def test_prune_ignores_the_other_products_folder(self):
        self.assertEqual(pipeline.runs_to_prune(['enlil/ambient/amb_20261001_00/'] * 3,
                                                'cone', keep=0), [])

    def test_upload_refuses_a_runid_from_the_wrong_product(self):
        with self.assertRaisesRegex(ValueError, 'not a valid ambient runId'):
            pipeline.upload_artifacts(None, '.', '20261006_58545', 'ambient')

    def test_coverage_end_adds_the_largest_gap(self):
        meta = {'times': ['2026-10-07T00:00:00Z', '2026-10-07T01:00:00Z', '2026-10-07T04:00:00Z']}
        self.assertEqual(pipeline.coverage_end(meta), datetime(2026, 10, 7, 7, tzinfo=timezone.utc))
        self.assertIsNone(pipeline.coverage_end(None))


@unittest.skipUnless(os.environ.get('SWV_AB_DIR'), 'set SWV_AB_DIR to run the real-data A/B check')
class RealDataABTest(unittest.TestCase):
    """suball vs pv-tim for the same run (20261006_58545): the evidence that the
    ambient product is the cone product's plane, units and encoding."""

    def test_suball_slice_matches_pv_tim(self):
        from extract import read_frame
        from suball import PLANE, fold
        d = Path(os.environ['SWV_AB_DIR'])
        ds = nc.Dataset(d / '58545.suball.nc')
        ds.set_auto_mask(False)
        try:
            base_s = np.median(fold(decode(ds, f'dd{PLANE}_3d', 0)), axis=0)
            base_p = np.maximum(np.median(read_frame(d / 'pv-tim.0000.nc')['density'], axis=0), 1e-6)[13]
            for num in ('0048', '0072', '0100', '0140'):
                fr = read_frame(d / f'pv-tim.{num}.nc')
                a = quantize_ratio(fr['density'][:, 13, :] / base_p[None, :]).astype(int)
                b = quantize_ratio(fold(decode(ds, f'dd{PLANE}_3d', int(num))) / base_s[None, :]).astype(int)
                self.assertLessEqual(np.median(np.abs(a - b)), 1, num)
                self.assertLessEqual(np.percentile(np.abs(a - b), 95), 5, num)
                va = quantize(fr['vr'][:, 13, :], *VR_SCALE).astype(int)
                vb = quantize(fold(decode(ds, f'vv{PLANE}_3d', int(num))) / 1000.0, *VR_SCALE).astype(int)
                self.assertLessEqual(np.median(np.abs(va - vb)), 1, num)
        finally:
            ds.close()


if __name__ == '__main__':
    unittest.main()
