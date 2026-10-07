"""
Ambient WSA-Enlil run (NCEP NOMADS `suball.nc`) -> transit-field artifacts.

Why this exists: s3://noaa-wsa-enlil-pds carries CONE runs only, so in a quiet
stretch the monitor's transit field has nothing to draw (~40 of ~280 days in
2026). NCEP runs a daily *ambient* run (`ncmes=0`, no CMEs) and publishes it only
to NOMADS, as `wsa_enlil.mrid00000000.suball.nc`. That file is three orthogonal
2D cut planes, not a volume — enough here, because with no CMEs there is nothing
for regions.py to track, and the ecliptic plane is all the transit field draws.

Verified 2026-10-07 by A/B against run 58545 from both sources (suball vs S3
pv-tim, see swv-data-sources): the `13` plane IS the Earth plane the cone runs
publish (r = 0.998-0.9996 for the ratio field, 0.998-0.9997 for vr), `vv` is
m/s, and the published u8 slices differ by a median of 1 level of 255. Rendered
side by side with the monitor's ramp, a cone run and the ambient run at the same
instant have the same brightness (mean level 78.0 vs 78.7) and the same
corotating structure; only the CME is missing, which is correct.

Output is the cone runs' slice format on their exact grid (90 lon x 64 r), so
the monitor cannot tell the two apart beyond `kind`/`capabilities`:
  slice_NNNN.bin / slvr_NNNN.bin / sldp_NNNN.bin   [90 x 64] u8, every frame
  line.bin                                         Sun->Earth profiles
  meta.json                                        no `grid.lat` -> no volumes, no tilt

Not produced: vol/voldp (no volume exists), labels/blobs (nothing to track).
`sldp` is written as zeros and must stay present — the monitor's hasField()
requires it. ⚠️ Never derive it from a suball `cc` tracer: on cone runs `cc` is a
different normalisation from DP (12.9 vs 1.7 at the same frame of 58545), and
ambient files carry no `cc` variables at all.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import netCDF4 as nc
import numpy as np

from extract import (LINE_LAYOUT, VR_SCALE, field_scales, quantize, quantize_ratio)
from regions import AU_KM

# suball's native grid and how it folds onto the published one. Both factors
# are exact: suball lon centres 1,3 -> published 2, and eight 0.003125 AU radial
# cells centre on the published 0.1125 ... 1.6875 AU. Asserted per file, because
# a silent NCEP grid change would otherwise shift the whole field.
NATIVE = {'z': 180, 'x': 512}
LON_FACTOR = 2
RAD_FACTOR = 8
PLANE = '13'                  # (lon, r) at Earth's latitude
# Same floor pipeline.describe_run applies to cone runs: fewer frames than
# this is a partial or broken file, not a short run.
MIN_FRAMES = 20
PROTON_MASS_KG = 1.67262192e-27
AU_M = AU_KM * 1000.0


def decode(ds, name, index):
    """One int16-packed plane, as float64.

    Values are packed against per-variable *global* bounds carried as
    attributes (`dd13_min`/`dd13_max`), across the full int16 range. A raw read
    is meaningless — it looks like large negative numbers.
    """
    var = ds.variables[name]
    tag = name.split('_')[0]
    lo = float(var.getncattr(f'{tag}_min'))
    hi = float(var.getncattr(f'{tag}_max'))
    raw = np.asarray(var[index], dtype=np.float64)
    return lo + (raw + 32768.0) * (hi - lo) / 65535.0


def fold(a):
    """(180 lon, 512 r) -> published (90 lon, 64 r), by block mean."""
    nl, nr = a.shape
    return a.reshape(nl // LON_FACTOR, LON_FACTOR, nr // RAD_FACTOR, RAD_FACTOR).mean(axis=(1, 3))


def refdate(ds):
    """NCEP's run reference time (`REFDATE_CAL`) — this run's `rundate`."""
    return datetime.strptime(str(ds.getncattr('REFDATE_CAL')), '%Y-%m-%dT%H:%M:%S').replace(
        tzinfo=timezone.utc)


def ambient_run_id(ds):
    """`amb_<YYYYMMDD>_<HH>` — a pattern no cone runId (`<YYYYMMDD>_<number>`)
    can match, which is what lets the worker route a runId to its R2 folder."""
    return refdate(ds).strftime('amb_%Y%m%d_%H')


def validate(ds):
    dims = {k: len(v) for k, v in ds.dimensions.items()}
    for dim, n in NATIVE.items():
        if dims.get(dim) != n:
            raise ValueError(f'unexpected suball grid: {dim}={dims.get(dim)}, expected {n}')
    if dims.get('t', 0) < MIN_FRAMES:
        raise ValueError(f'suball has only {dims.get("t")} frames')
    for name in (f'dd{PLANE}_3d', f'vv{PLANE}_3d', 'time', 'x_coord', 'z_coord'):
        if name not in ds.variables:
            raise ValueError(f'suball lacks {name}')
    if 'REFDATE_CAL' not in ds.ncattrs():
        raise ValueError('suball lacks REFDATE_CAL')


def earth_position(ds):
    """Earth (lon, lat) in degrees, model frame, from the file's own Earth series
    (X2 = colatitude, X3 = longitude, both radians; X3 = pi is the HEEQ+180
    convention's lon 180). Falls back to (180, 0) if the series is absent."""
    try:
        colat = np.asarray(ds.variables['Earth_X2'][:], dtype=float)
        lon = np.asarray(ds.variables['Earth_X3'][:], dtype=float)
        return float(np.degrees(np.median(lon)) % 360.0), float(90.0 - np.degrees(np.median(colat)))
    except KeyError:
        return 180.0, 0.0


def extract_ambient_run(nc_path, out_dir, run_id=None, source=None):
    """Extract an ambient `suball.nc` into the monitor's slice artifacts; returns meta."""
    ds = nc.Dataset(nc_path)
    ds.set_auto_mask(False)
    try:
        validate(ds)
        os.makedirs(out_dir, exist_ok=True)
        run_id = run_id or ambient_run_id(ds)
        rundate = refdate(ds)

        lon = fold(np.tile(np.degrees(np.asarray(ds.variables['z_coord'][:], dtype=float))[:, None],
                           (1, NATIVE['x'])))[:, 0]
        rad = fold(np.tile((np.asarray(ds.variables['x_coord'][:], dtype=float) / AU_M)[None, :],
                           (NATIVE['z'], 1)))[0]
        earth_lon, earth_lat = earth_position(ds)
        ilon_earth = int(np.argmin(np.abs(((lon - earth_lon + 180) % 360) - 180)))

        # Same display baseline as extract_run: the azimuthal median of the first
        # frame, per radius. Converted to r²-scaled protons/cc first so
        # `ambientEcliptic` is in the cone runs' units — the ratio itself is
        # scale-free either way. (Measured against pv-tim the physical m_p
        # conversion sits ~4% low; harmless, and it is only informational.)
        def number_density(i):
            mass = fold(decode(ds, f'dd{PLANE}_3d', i))                  # kg/m³
            return mass / PROTON_MASS_KG / 1e6 * rad[None, :] ** 2        # r²·N, p/cc·AU²

        baseline = np.maximum(np.median(number_density(0), axis=0), 1e-6)
        seconds = np.asarray(ds.variables['time'][:], dtype=float)
        n = len(seconds)
        frames, times = [], []
        line = np.zeros((n, len(rad), 3), dtype=np.uint8)
        zeros = np.zeros((len(lon), len(rad)), dtype=np.uint8)

        for i in range(n):
            num = f'{i:04d}'
            ratio = quantize_ratio(number_density(i) / baseline[None, :])
            vr = quantize(fold(decode(ds, f'vv{PLANE}_3d', i)) / 1000.0, *VR_SCALE)   # m/s -> km/s
            ratio.tofile(os.path.join(out_dir, f'slice_{num}.bin'))
            vr.tofile(os.path.join(out_dir, f'slvr_{num}.bin'))
            zeros.tofile(os.path.join(out_dir, f'sldp_{num}.bin'))
            line[i, :, 0] = ratio[ilon_earth]
            line[i, :, 1] = vr[ilon_earth]
            frames.append(num)
            times.append((rundate + timedelta(seconds=float(seconds[i]))).strftime('%Y-%m-%dT%H:%M:%SZ'))
        line.tofile(os.path.join(out_dir, 'line.bin'))

        meta = {
            'runId': run_id,
            'kind': 'ambient',
            'capabilities': {'volumes': False, 'tracking': False},
            'generated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'rundate': rundate.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'frames': frames,
            'times': times,
            # ⚠️ No `lat` and no `eclipticLatIndex`, on purpose: the monitor gates
            # its whole volume/tilt path on `grid.lat` (enlilField.hasVolume,
            # transit tilt arming), so omitting it is what keeps it from asking
            # for vol_ files that do not exist.
            'grid': {
                'lon': {'n': len(lon), 'min': float(lon[0]), 'max': float(lon[-1])},
                'rad': {'n': len(rad), 'min': float(rad[0]), 'max': float(rad[-1]), 'units': 'AU'},
            },
            # The slice is cut at Earth's latitude, like a cone run's
            # `eclipticLatIndex` plane (verified, see the module docstring).
            'planeLat': round(earth_lat, 3),
            'layout': {'slice': ['lon', 'rad'], 'line': LINE_LAYOUT},
            'scales': field_scales(),
            'densityIsR2Scaled': True,
            'ambientEcliptic': [round(float(v), 4) for v in baseline],
            'earth': {'lon': earth_lon, 'lat': earth_lat, 'lonIndex': ilon_earth},
            'cmes': [],
            'dpMaxObserved': 0.0,
            'source': dict(source or {}, product='nomads-suball',
                           modelRunId=str(ds.getncattr('model_run_id')) if 'model_run_id' in ds.ncattrs() else None),
            'credit': 'NOAA NCEP WSA-Enlil (daily ambient run) via NOMADS',
        }
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        print(f'{n} ambient frames -> {out_dir}  ({run_id}, rundate {meta["rundate"]}, '
              f'Earth lon {earth_lon:.1f} lat {earth_lat:.2f})')
        return meta
    finally:
        ds.close()


def main():
    ap = argparse.ArgumentParser(description='Extract an ambient suball.nc into transit-field artifacts')
    ap.add_argument('--file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-id', default=None)
    args = ap.parse_args()
    extract_ambient_run(args.file, args.out, run_id=args.run_id)


if __name__ == '__main__':
    main()
