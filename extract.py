"""
WSA-Enlil artifact extractor for the SpaceWeatherViz heliosphere view.

Reads a run's `pv-tim.NNNN.nc` snapshots (the pre-downsampled 90 lon x 30 lat x 64 r
files NOAA publishes under each run's pv-ready-data-*/ prefix on s3://noaa-wsa-enlil-pds)
and emits compact u8 binary artifacts + meta.json for the web frontend.

The pv files' `Density` is already r^2-scaled (r^2*N, flat ~4-5 in quiet wind, ~5 p/cc
at 1 AU after un-scaling) — verified 2026-09-04 on run 20260903_58484. On top of that
this script normalizes to the *excess-density ratio* Density / ambient(lat, r), where
ambient is the azimuthal median of the run's first (pre-CME) frame, so the CME is the
only bright object and thresholds like "1.8x ambient" are run-independent.

Artifacts (all u8, little-endian, layouts documented in meta.json):
  slice_NNNN.bin   ecliptic-plane ratio field, lon-major [90 x 64]
  slvr_NNNN.bin    ecliptic-plane radial velocity (km/s), lon-major [90 x 64]
  sldp_NNNN.bin    ecliptic-plane CME cloud tracer DP, lon-major [90 x 64]
  vol_NNNN.bin     full ratio volume, [90 lon x 30 lat x 64 r], lon-major
  voldp_NNNN.bin   full DP volume, same layout
  line.bin         Sun->Earth line profiles, [nframes x 64 r x 3 (ratio, vr, dp)]
  meta.json        run id/times, grid, scales, ambient curve, Earth position, CME params

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
        'vr': np.array(ds.variables['Vr'][0]),            # km/s despite 'm/s' attr
        'dp': np.array(ds.variables['DP'][0]),
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


def extract_run(run_dir, out_dir, run_id=None, volumes=True):
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
    ecl = int(np.argmin(np.abs(lat - (earth_lat or 0.0))))
    ilon_earth = int(np.argmin(np.abs(((lon - (earth_lon or 180.0) + 180) % 360) - 180)))

    # Ambient = azimuthal median of the first (pre-CME) frame, per (lat, rad) cell.
    ambient = np.median(first['density'], axis=0)          # (lat, rad)
    ambient = np.maximum(ambient, 1e-6)

    n_r = len(rad)
    line = np.zeros((len(frames), n_r, 3), dtype=np.uint8)
    times, frame_ids, dp_max = [], [], 0.0

    for fi, path in enumerate(frames):
        fr = read_frame(path) if fi else first
        num = frame_number(path)
        frame_ids.append(num)
        times.append(fr['time'].strftime('%Y-%m-%dT%H:%M:%SZ'))

        ratio = fr['density'] / ambient[None, :, :]        # (lon, lat, rad)
        dp = np.nan_to_num(fr['dp'])
        dp_max = max(dp_max, float(dp.max()))

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
        'cmes': parse_cmes(run_dir),
        'dpMaxObserved': round(dp_max, 4),
        'credit': 'NOAA SWPC WSA-Enlil via the NOAA Open Data Dissemination Program',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    sizes = {p: os.path.getsize(os.path.join(out_dir, p))
             for p in ['slice_' + frame_ids[0] + '.bin', 'line.bin', 'meta.json']}
    print(f'{len(frames)} frames -> {out_dir}')
    print(f'ecliptic lat index {ecl} (lat {lat[ecl]:.1f}, Earth lat {earth_lat}), '
          f'Earth lon {earth_lon:.1f} (index {ilon_earth})')
    print(f'dp max observed: {dp_max:.3f}')
    print('sizes:', {k: f'{v/1024:.1f} KB' for k, v in sizes.items()})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-id', default=None, help='run identifier for meta (default: run-dir name)')
    ap.add_argument('--no-volumes', action='store_true')
    args = ap.parse_args()
    extract_run(args.run_dir, args.out, run_id=args.run_id, volumes=not args.no_volumes)


if __name__ == '__main__':
    main()
