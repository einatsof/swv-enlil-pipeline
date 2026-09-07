"""
WSA-Enlil artifact extractor for the SpaceWeatherViz heliosphere view.

Reads a run's `pv-tim.NNNN.nc` snapshots (the pre-downsampled 90 lon x 30 lat x 64 r
files NOAA publishes under each run's pv-ready-data-*/ prefix on s3://noaa-wsa-enlil-pds)
and emits compact u8 binary artifacts + meta.json for the web frontend.

The pv files' `Density` is already r^2-scaled (r^2*N, flat ~4-5 in quiet wind, ~5 p/cc
at 1 AU after un-scaling) — verified 2026-09-04 on run 20260903_58484. On top of that
this script normalizes to the *excess-density ratio* Density / ambient(lat, r), where
ambient is the azimuthal median of the run's first frame. This normalizes radial
contrast; it does not remove structured background wind or identify CME material.

Artifacts (field binaries are u8, layouts documented in meta.json):
  slice_NNNN.bin   ecliptic-plane ratio field, lon-major [90 x 64]
  slvr_NNNN.bin    ecliptic-plane radial velocity (km/s), lon-major [90 x 64]
  sldp_NNNN.bin    ecliptic-plane CME cloud tracer DP, lon-major [90 x 64]
  vol_NNNN.bin     full ratio volume, [90 lon x 30 lat x 64 r], lon-major
  voldp_NNNN.bin   full DP volume, same layout
  line.bin         Sun->Earth line profiles, [nframes x 64 r x 3 (ratio, vr, dp)]
  meta.json        run id/times, grid, scales, ambient curve, Earth position, CME params
  blobs_<tag>.json  per-frame 3D blob tracks, motion evidence and merge/split ancestry
  labels_<tag>_NNNN.bin  Earth-plane track IDs, little-endian uint16 [90 x 64]

Usage:
  python extract.py --run-dir <dir with pv-tim.*.nc [+ metadata.json, evo.earth.nc]>
                    --out <output dir> [--no-volumes]
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import netCDF4 as nc
import numpy as np

from regions import AU_KM, BlobTracker, TrackingConfig

# Shared with the tracker's own lineage rule so both agree what a sliver is.
MIN_BRANCH_FRACTION = TrackingConfig().min_branch_fraction

# Encodings are fixed (run-independent) so iso-thresholds keep physical meaning.
# ratio: log2 over [-2, 4.5] => 0.25x..22.6x ambient, ~1.8% relative precision
#   (linear clips: run 20260903 had p99.9 = 9.4, abs max 20.6 at the shock nose).
# dp: sqrt over [0, 6] => resolution concentrated at the faint cloud edge.
RATIO_LOG2 = (-2.0, 4.5)
VR_SCALE = (200.0, 1200.0)  # km/s, linear
DP_MAX = 6.0


def quantize(a, lo, hi):
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def quantize_ratio(a):
    return quantize(np.log2(np.maximum(a, 1e-6)), *RATIO_LOG2)


def quantize_dp(a):
    return np.clip(np.sqrt(np.clip(a, 0, DP_MAX) / DP_MAX) * 255.0, 0, 255).astype(np.uint8)


def frame_number(path):
    m = re.search(r'pv-tim\.(\d+)\.nc$', os.path.basename(path))
    return m.group(1) if m else None


def read_frame(path):
    ds = nc.Dataset(path)
    t = float(ds.variables['time'][0])
    out = {
        'time': datetime.fromtimestamp(t, tz=timezone.utc),
        'density': np.array(ds.variables['Density'][0]),  # (lon, lat, rad), r^2-scaled
        'vr': np.ma.filled(ds.variables['Vr'][0], np.nan),  # km/s despite 'm/s' attr
        # Preserve missing-value semantics: np.array(masked_array) exposes a
        # finite NetCDF fill value, which would look like a huge cloud/speed.
        'dp': np.ma.filled(ds.variables['DP'][0], np.nan),
        'lon': np.array(ds.variables['longitude'][:]),
        'lat': np.array(ds.variables['latitude'][:]),
        'rad': np.array(ds.variables['radius'][:]),
    }
    ds.close()
    return out


def earth_position(run_dir):
    """Earth's (lon_deg, lat_deg) in the model frame, from evo.earth.nc X/Y/Z."""
    path = os.path.join(run_dir, 'evo.earth.nc')
    if not os.path.exists(path):
        return None, None
    ds = nc.Dataset(path)
    x = np.array(ds.variables['X'][:], dtype=float)
    y = np.array(ds.variables['Y'][:], dtype=float)
    z = np.array(ds.variables['Z'][:], dtype=float)
    ds.close()
    lon = np.degrees(np.arctan2(y, x)) % 360.0
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return float(np.median(lon)), float(np.median(lat))


def parse_cmes(run_dir):
    """CME cone parameters from NOAA's pv-ready metadata.json (comma-joined strings)."""
    path = os.path.join(run_dir, 'metadata.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        md = json.load(f)

    def nums(key):
        raw = str(md.get(key, '') or '')
        return [float(v) for v in re.findall(r'-?\d+\.?\d*', raw)]

    times = re.findall(r"[\d-]+T[\d:]+", str(md.get('cme_time', '') or ''))
    lats, lons = nums('cme_latitude'), nums('cme_longitude')
    half, vel = nums('cme_cone_half_angle'), nums('cme_radial_velocity')
    cmes = []
    for i, t in enumerate(times):
        cmes.append({
            'time': t + 'Z',
            'latitude': lats[i] if i < len(lats) else None,
            'longitude': lons[i] if i < len(lons) else None,
            'halfAngle': half[i] if i < len(half) else None,
            'speed': vel[i] if i < len(vel) else None,
        })
    return cmes


class ConeAttributor:
    """Join NOAA's cone inputs to blob tracks, then follow them frame to frame.

    `regions.py` deliberately knows nothing about CMEs — it tracks visible
    material. This is the join, and it lives here because it needs the run's cone
    list *and* the full 3D label volume, which is never exported (only the
    Earth-plane slice is), so no downstream consumer could do it later.

    **A cone is read off its injection footprint**, not matched to a new track.
    Matching a cone to a newly-born component only works when the cone creates
    one: measured on run 20260906_58495, three of nine cones (the two at
    lat -4/lon -28 and the one at lat +19/lon -75) were injected into material
    that already existed, so no component was born and birth-matching never saw
    them at all. Reading the label already present in the wedge the cone occupies
    covers both cases with one rule — verified on that run, seven of nine cones
    resolved with **exactly one track present in the footprint**, no tie to break.

    The remaining two were injected before the run's first saved frame (spin-up).
    The same rule reaches them because the shell range extends to where the cone's
    nose could have travelled by the sampled frame, rather than to a fixed depth.

    Once attributed, a cone follows its track through the tracker's `links`.
    **A split hands the cone to every substantial fragment, but not to slivers.**
    `regions.py` does not invent a boundary inside connected material, so which
    fragment carries which CME is genuinely unrecoverable and picking one would be
    a guess. Spreading without any bound is worse, though: it compounds, and late
    in a 57-frame run — where the field merges into one cloud and then breaks up
    at the outer shell — it put all 9 cones of 20260906_58495 onto all 7 pieces,
    every CME lighting every blob. The cut is `min_branch_fraction`, the same rule
    the tracker uses for its own lineage, so the two agree on what a sliver is.
    """

    def __init__(self, cones, lon, lat, rad, earth_lon):
        self.cones, self.lon, self.lat, self.rad = cones, lon, lat, rad
        self.dr = (rad[-1] - rad[0]) / (len(rad) - 1)
        self.earth_lon = earth_lon
        self.tracks = {}        # cone index -> live track ids carrying it
        self.info = {}          # cone index -> how it was attributed
        self.pending = []
        for k, cone in enumerate(cones):
            when = cone.get('time')
            if when is None or cone.get('latitude') is None or cone.get('longitude') is None:
                self.info[k] = {'frame': None, 'trackIds': [], 'note': 'cone has no time or direction'}
                continue
            self.pending.append((k, datetime.fromisoformat(when.replace('Z', '+00:00'))))

    def _footprint(self, cone, frame_time, cme_time):
        """Grid indices of the cone's wedge, from the inner boundary to its nose."""
        half = cone.get('halfAngle') or 20.0
        # metadata.json longitudes are Earth-relative; the grid's are not.
        grid_lon = (self.earth_lon + cone['longitude']) % 360.0
        i = np.flatnonzero(np.abs(((self.lon - grid_lon + 180) % 360) - 180) <= half)
        j = np.flatnonzero(np.abs(self.lat - cone['latitude']) <= half)
        # The cloud spans the boundary out to roughly its ballistic nose. One
        # extra shell absorbs frame quantisation — a cone injected just after a
        # frame is not sampled until the next one, up to a full interval later.
        hours = max(0.0, (frame_time - cme_time).total_seconds() / 3600)
        reach = (cone.get('speed') or 0.0) * hours * 3600 / AU_KM
        k = int(np.clip(round(reach / self.dr) + 1, 1, len(self.rad) - 1))
        return i, j, np.arange(k + 1)

    def update(self, num, frame_time, labels, records):
        """Carry cones onto this frame's tracks, then annotate `coneIdxs`."""
        predecessors = [{link['trackId'] for link in r['links']} for r in records]
        live = {r['trackId'] for r in records}
        for k, held in self.tracks.items():
            moved = [r for r, p in zip(records, predecessors) if p & held]
            # ⚠️ **A split must not hand the cone to every fragment.** Doing so
            # compounds: late in a 57-frame run the whole field merges into one
            # cloud and then breaks up at the outer shell, and an unbounded rule
            # put all 9 cones of 20260906_58495 on all 7 pieces — every CME
            # lighting every blob, which is worse than no answer. Keep only the
            # fragments that carry real material, using the tracker's own lineage
            # rule (min_branch_fraction) so the two agree on what a sliver is.
            if moved:
                biggest = max(r['cellCount'] for r in moved)
                floor = biggest * MIN_BRANCH_FRACTION
                self.tracks[k] = {r['trackId'] for r in moved if r['cellCount'] >= floor}
            else:
                # Nothing linked: keep an unlinked survivor rather than dropping a
                # cone on one weak frame; an ended track resolves to empty.
                self.tracks[k] = held & live
            if self.tracks[k]:
                self.info[k]['lastFrame'] = num

        for entry in list(self.pending):
            k, cme_time = entry
            if frame_time < cme_time:
                continue                      # not injected yet; try again next frame
            self.pending.remove(entry)
            i, j, kk = self._footprint(self.cones[k], frame_time, cme_time)
            patch = labels[np.ix_(i, j, kk)].ravel() if len(i) and len(j) else np.empty(0, labels.dtype)
            present = patch[patch > 0]
            if not len(present):
                self.info[k] = {'frame': None, 'trackIds': [],
                                'note': 'no tracked material in the injection footprint'}
                continue
            ids, counts = np.unique(present, return_counts=True)
            best = int(ids[int(np.argmax(counts))])
            self.tracks[k] = {best}
            # Kept for auditing: a low share or several tracks in one footprint is
            # the signature of an attribution that deserves a second look.
            self.info[k] = {'frame': num, 'trackIds': [best], 'lastFrame': num,
                            'footprintShare': round(float(counts.max()) / patch.size, 4),
                            'tracksInFootprint': int(len(ids))}

        for record in records:
            record['coneIdxs'] = sorted(k for k, held in self.tracks.items()
                                        if record['trackId'] in held)

    def summary(self):
        """Per-cone attribution, positionally aligned with the run's cone list.

        ⚠️ `trackIds` is **where the cone was identified**, at injection — not
        where its material had drifted by the last frame. Those differ once
        clouds merge, and reporting the drifted set made every cone in a run look
        identical (all 9 of 20260906_58495 came back as the same 7 late-run
        fragments, which says nothing about any of them). The live, per-frame
        answer is `coneIdxs` on each region in the manifest; this field is the
        provenance of the match, and `footprintShare`/`tracksInFootprint` are the
        evidence for it.
        """
        return [dict(self.info.get(k, {'frame': None, 'trackIds': []}),
                     lastTrackIds=sorted(self.tracks.get(k, [])))
                for k in range(len(self.cones))]


def extract_run(run_dir, out_dir, run_id=None, volumes=True, tracking_config=None):
    """Extract one downloaded run into web artifacts; returns the meta dict.

    Callable so pipeline.py can drive it directly; the CLI below wraps it.
    """
    frames = sorted(glob.glob(os.path.join(run_dir, 'pv-tim.*.nc')))
    if not frames:
        raise SystemExit(f'no pv-tim.*.nc in {run_dir}')
    os.makedirs(out_dir, exist_ok=True)

    first = read_frame(frames[0])
    lat, rad, lon = first['lat'], first['rad'], first['lon']
    earth_lon, earth_lat = earth_position(run_dir)
    # Slice plane: latitude cell nearest Earth (ENLIL's "earth plane"), not blindly the middle.
    ecl = int(np.argmin(np.abs(lat - (earth_lat if earth_lat is not None else 0.0))))
    earth_grid_lon = earth_lon if earth_lon is not None else 180.0
    ilon_earth = int(np.argmin(np.abs(((lon - earth_grid_lon + 180) % 360) - 180)))
    tracker = BlobTracker(lon, lat, rad, config=tracking_config)
    tracking_tag = tracker.config.artifact_tag()
    tracking_frames = []
    cmes = parse_cmes(run_dir)
    attributor = ConeAttributor(cmes, lon, lat, rad, earth_grid_lon)

    # A display baseline, not a CME-free background simulation.
    ambient = np.median(first['density'], axis=0)          # (lat, rad)
    ambient = np.maximum(ambient, 1e-6)

    n_r = len(rad)
    line = np.zeros((len(frames), n_r, 3), dtype=np.uint8)
    times, frame_ids, dp_max = [], [], 0.0

    for fi, path in enumerate(frames):
        fr = read_frame(path) if fi else first
        if any(not np.array_equal(fr[key], axis) for key, axis in zip(('lon', 'lat', 'rad'), (lon, lat, rad))):
            raise ValueError(f'tracking grid changed in {path}')
        num = frame_number(path)
        frame_ids.append(num)
        times.append(fr['time'].strftime('%Y-%m-%dT%H:%M:%SZ'))

        ratio = fr['density'] / ambient[None, :, :]        # (lon, lat, rad)
        dp = np.nan_to_num(fr['dp'], nan=0.0, posinf=0.0, neginf=0.0)
        dp_max = max(dp_max, float(dp.max()))

        # Track original float volumes, even when --no-volumes is selected.
        # Publishing the Earth slice must not break a cloud's 3D identity.
        track_labels, tracked = tracker.update(fr['dp'], fr['vr'], fr['time'])
        labels_file = f'labels_{tracking_tag}_{num}.bin'
        slice_labels = track_labels[:, ecl, :]
        slice_labels.astype('<u2').tofile(os.path.join(out_dir, labels_file))
        visible_ids, visible_counts = np.unique(slice_labels, return_counts=True)
        counts = dict(zip(visible_ids.tolist(), visible_counts.tolist()))
        for region in tracked['regions']:
            region['sliceCellCount'] = counts.get(region['trackId'], 0)
        # Needs the 3D labels, so it has to happen here — only the slice is written.
        attributor.update(num, fr['time'], track_labels, tracked['regions'])
        tracking_frames.append({'frame': num, 'time': times[-1], 'labelsFile': labels_file, **tracked})

        sl_ratio = quantize_ratio(ratio[:, ecl, :])
        sl_vr = quantize(fr['vr'][:, ecl, :], *VR_SCALE)
        sl_dp = quantize_dp(dp[:, ecl, :])
        sl_ratio.tofile(os.path.join(out_dir, f'slice_{num}.bin'))
        sl_vr.tofile(os.path.join(out_dir, f'slvr_{num}.bin'))
        sl_dp.tofile(os.path.join(out_dir, f'sldp_{num}.bin'))

        if volumes:
            quantize_ratio(ratio).tofile(os.path.join(out_dir, f'vol_{num}.bin'))
            quantize_dp(dp).tofile(os.path.join(out_dir, f'voldp_{num}.bin'))

        line[fi, :, 0] = sl_ratio[ilon_earth]
        line[fi, :, 1] = sl_vr[ilon_earth]
        line[fi, :, 2] = sl_dp[ilon_earth]

    line.tofile(os.path.join(out_dir, 'line.bin'))

    meta = {
        'runId': run_id or os.path.basename(os.path.normpath(run_dir)),
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'frames': frame_ids,
        'times': times,
        'grid': {
            'lon': {'n': len(lon), 'min': float(lon[0]), 'max': float(lon[-1])},
            'lat': {'n': len(lat), 'min': float(lat[0]), 'max': float(lat[-1])},
            'rad': {'n': n_r, 'min': float(rad[0]), 'max': float(rad[-1]), 'units': 'AU'},
            'eclipticLatIndex': ecl,
        },
        'layout': {
            'slice': ['lon', 'rad'],
            'vol': ['lon', 'lat', 'rad'],
            'line': ['frame', 'rad', ['ratio', 'vr', 'dp']],
        },
        'scales': {
            'ratio': {'encoding': 'log2', 'min': RATIO_LOG2[0], 'max': RATIO_LOG2[1]},
            'vr': {'encoding': 'linear', 'min': VR_SCALE[0], 'max': VR_SCALE[1], 'units': 'km/s'},
            'dp': {'encoding': 'sqrt', 'max': DP_MAX},
        },
        'densityIsR2Scaled': True,
        'ambientEcliptic': [round(float(v), 4) for v in ambient[ecl]],
        'earth': {'lon': earth_lon, 'lat': earth_lat, 'lonIndex': ilon_earth},
        # Each cone carries `tracking`: which blob track holds it, and the
        # evidence for that (see ConeAttributor). `trackIds: []` is an honest
        # "this cone has no visible material", not a lookup failure.
        'cmes': [dict(cone, tracking=info) for cone, info in zip(cmes, attributor.summary())],
        'dpMaxObserved': round(dp_max, 4),
        'blobTracking': {
            **tracker.config.metadata(),
            'artifactTag': tracking_tag,
            'manifest': f'blobs_{tracking_tag}.json',
            'labelPattern': f'labels_{tracking_tag}_NNNN.bin',
            'labelDtype': 'uint16',
            'labelByteOrder': 'little',
            'labelLayout': ['lon', 'rad'],
            'backgroundLabel': 0,
            'scope': '3d',
            # Regions carry `coneIdxs` (indices into `cmes`); cones carry
            # `tracking.trackIds`. Both sides of the join are published so a
            # consumer never has to re-derive it from geometry.
            'coneAttribution': 'injection-footprint',
        },
        'credit': 'NOAA SWPC WSA-Enlil via the NOAA Open Data Dissemination Program',
    }
    with open(os.path.join(out_dir, meta['blobTracking']['manifest']), 'w') as f:
        json.dump({'runId': meta['runId'], **meta['blobTracking'], 'grid': meta['grid'],
                   'frames': tracking_frames}, f, separators=(',', ':'), allow_nan=False)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    sizes = {p: os.path.getsize(os.path.join(out_dir, p))
             for p in ['slice_' + frame_ids[0] + '.bin', 'line.bin', 'meta.json']}
    print(f'{len(frames)} frames -> {out_dir}')
    print(f'ecliptic lat index {ecl} (lat {lat[ecl]:.1f}, Earth lat {earth_lat}), '
          f'Earth lon {earth_grid_lon:.1f} (index {ilon_earth})')
    print(f'dp max observed: {dp_max:.3f}')
    print('sizes:', {k: f'{v/1024:.1f} KB' for k, v in sizes.items()})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-id', default=None, help='run identifier for meta (default: run-dir name)')
    ap.add_argument('--no-volumes', action='store_true')
    ap.add_argument('--dp-threshold', type=float, default=0.25,
                    help='floating-point DP threshold for blob tracking (default: 0.25)')
    ap.add_argument('--tracking-longitude-rate', type=float, default=0.0,
                    help='known longitude drift in grid degrees/day (default: 0; pv has Vr only)')
    args = ap.parse_args()
    config = TrackingConfig(dp_threshold=args.dp_threshold,
                            longitude_rate_deg_per_day=args.tracking_longitude_rate)
    extract_run(args.run_dir, args.out, run_id=args.run_id, volumes=not args.no_volumes,
                tracking_config=config)


if __name__ == '__main__':
    main()
