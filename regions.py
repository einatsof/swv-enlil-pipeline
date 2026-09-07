"""Track connected ENLIL DP clouds in the source (lon, lat, radius) volume.

IDs describe visible material, not CME identities. A one-to-one continuation
keeps its ID; merges/splits start new tracks with explicit ancestry. No boundary
is invented inside connected material and no DONKI attribution is inferred.
Only the previous volume is retained, so a run can be processed as a stream.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json

import numpy as np


AU_KM = 149_600_000.0
# Exported label width. A run produces a few hundred tracks at most (55 over the
# 57 frames of 20260906_58495), so uint16 halves the per-frame artifact against
# uint32 with three orders of magnitude of headroom. update() enforces the
# ceiling rather than letting IDs silently wrap.
LABEL_DTYPE = np.uint16
# Bump when algorithm/output semantics change: artifacts are immutable in R2.
TRACKING_VERSION = 2


@dataclass(frozen=True)
class TrackingConfig:
    dp_threshold: float = 0.25
    min_cells: int = 1              # preserve small, newly injected clouds
    min_overlap: float = 0.25       # overlap / size of the smaller component
    min_branch_fraction: float = 0.05  # ignore tiny branches beside a dominant link
    tolerance_cells: int = 1       # used only when direct prediction fails
    max_gap_hours: float = 6.0
    # pv provides Vr only. Do not substitute the Sun's rotation rate for a
    # missing transverse plasma velocity. Supply a known grid-frame rate only.
    longitude_rate_deg_per_day: float = 0.0

    def __post_init__(self):
        for value in (self.dp_threshold, self.min_overlap, self.min_branch_fraction, self.max_gap_hours,
                      self.longitude_rate_deg_per_day):
            if not np.isfinite(value):
                raise ValueError('tracking settings must be finite')
        if (self.dp_threshold <= 0 or not 0 < self.min_overlap <= 1
                or not 0 <= self.min_branch_fraction <= 1 or self.max_gap_hours <= 0):
            raise ValueError('invalid tracking threshold, overlap or maximum gap')
        if not isinstance(self.min_cells, int) or self.min_cells < 1:
            raise ValueError('min_cells must be a positive integer')
        if not isinstance(self.tolerance_cells, int) or self.tolerance_cells < 0:
            raise ValueError('tolerance_cells must be a nonnegative integer')

    def metadata(self):
        return {'version': TRACKING_VERSION, 'config': asdict(self)}

    def artifact_tag(self):
        """Version + configuration identify a reproducible set of artifacts."""
        payload = json.dumps(self.metadata(), sort_keys=True, separators=(',', ':'))
        return sha256(payload.encode('utf-8')).hexdigest()[:12]


def connected_components(mask, min_cells=1):
    """Six-connected labels; longitude alone is periodic. Labels start at 1."""
    nl, nt, nr = mask.shape
    plane = nt * nr
    size = mask.size
    active = np.asarray(mask, dtype=bool).ravel()
    labels = np.zeros(size, dtype=np.uint32)
    components = []
    # Visit small components too, then remove them without rescanning their cells.
    visited = np.zeros(size, dtype=bool)
    for seed in np.flatnonzero(active):
        if visited[seed]:
            continue
        visited[seed] = True
        stack, cells = [int(seed)], []
        while stack:
            c = stack.pop()
            cells.append(c)
            lat_index, radial_index = (c % plane) // nr, c % nr
            neighbours = [(c - plane) % size, (c + plane) % size]
            if lat_index > 0:
                neighbours.append(c - nr)
            if lat_index + 1 < nt:
                neighbours.append(c + nr)
            if radial_index > 0:
                neighbours.append(c - 1)
            if radial_index + 1 < nr:
                neighbours.append(c + 1)
            for nb in neighbours:
                if active[nb] and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(cells) >= min_cells:
            cells = np.asarray(cells, dtype=np.intp)
            components.append(cells)
            labels[cells] = len(components)
    return labels.reshape(mask.shape), components


def _dilate(mask, radius):
    """Box tolerance in grid cells, with no latitude/radius edge wrap."""
    for axis in range(3):
        expanded = mask.copy()
        for offset in range(1, min(radius, mask.shape[axis] - 1) + 1):
            if axis == 0:
                expanded |= np.roll(mask, offset, axis=0) | np.roll(mask, -offset, axis=0)
            else:
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[axis], hi[axis] = slice(None, -offset), slice(offset, None)
                expanded[tuple(lo)] |= mask[tuple(hi)]
                expanded[tuple(hi)] |= mask[tuple(lo)]
        mask = expanded
    return mask


class BlobTracker:
    """Process consecutive float DP/Vr volumes using their actual UTC times.

    update() returns (LABEL_DTYPE track-label volume, JSON-compatible frame record).
    Track IDs are positive and unique within this tracker/run; zero is background.
    Association scores are geometric evidence, not calibrated probabilities.
    """

    def __init__(self, lon, lat, rad, config=None):
        self.config = config or TrackingConfig()
        self.axes = tuple(np.asarray(a, dtype=float).copy() for a in (lon, lat, rad))
        for dimension, (axis, minimum) in enumerate(zip(self.axes, (2, 1, 2))):
            if axis.ndim != 1 or len(axis) < minimum or not np.all(np.isfinite(axis)):
                raise ValueError('invalid tracking coordinate axis')
            if len(axis) > 1:
                steps = np.diff(axis)
                step = (axis[-1] - axis[0]) / (len(axis) - 1)
                # NOAA pv latitude runs north -> south. Preserve that storage
                # order; only lon/radius must increase. Allow float32 roundoff.
                monotonic = np.all(steps > 0) or (dimension == 1 and np.all(steps < 0))
                if not monotonic or not np.allclose(steps, step, rtol=1e-4):
                    raise ValueError('tracking requires uniform, monotonic axes (increasing lon/radius)')
        self.lon, self.lat, self.rad = self.axes
        self.dlon = (self.lon[-1] - self.lon[0]) / (len(self.lon) - 1)
        self.dr = (self.rad[-1] - self.rad[0]) / (len(self.rad) - 1)
        if not np.isclose(self.dlon * len(self.lon), 360) or np.any(np.abs(self.lat) > 90) or self.rad[0] <= 0:
            raise ValueError('expected full-circle longitude, latitude degrees and positive AU radii')
        self.shape = tuple(len(a) for a in self.axes)
        self._previous = []
        self._time = None
        self._next_id = 1

    def _predict(self, blob, dt):
        i, j, k = np.unravel_index(blob['cells'], self.shape)
        speed = blob['speed']
        # Unknown velocity cannot provide motion evidence. Valid cells in a
        # partially missing velocity field can still support a continuation.
        valid = np.isfinite(speed) & (speed >= 0)
        r = self.rad[k] + np.where(valid, speed, 0) * dt / AU_KM
        valid &= (r >= self.rad[0] - self.dr / 2) & (r < self.rad[-1] + self.dr / 2)
        k = np.floor((r[valid] - self.rad[0]) / self.dr + 0.5).astype(np.intp)
        shift = self.config.longitude_rate_deg_per_day * dt / 86400 / self.dlon
        i = np.floor(i[valid] + shift + 0.5).astype(np.intp) % self.shape[0]
        # Drop exiting material rather than clamping it onto the outer shell.
        return np.unique(np.ravel_multi_index((i, j[valid], k), self.shape))

    def _links(self, labels, components, dt):
        links = []
        sizes = np.array([0] + [len(c) for c in components])
        flat = labels.ravel()
        for parent_index, blob in enumerate(self._previous):
            predicted = self._predict(blob, dt)
            if not len(predicted):
                continue
            for tolerant in (False, True):
                if tolerant:
                    if not self.config.tolerance_cells:
                        break
                    mask = np.zeros(self.shape, dtype=bool)
                    mask.ravel()[predicted] = True
                    sample = flat[_dilate(mask, self.config.tolerance_cells).ravel()]
                else:
                    sample = flat[predicted]
                ids, counts = np.unique(sample[sample > 0], return_counts=True)
                accepted = []
                for cid, count in zip(ids, counts):
                    score = min(1.0, float(count) / min(len(predicted), int(sizes[cid])))
                    if score >= self.config.min_overlap:
                        accepted.append({'parent': parent_index, 'component': int(cid) - 1,
                                         'overlap': round(score, 4),
                                         'method': 'tolerant' if tolerant else 'direct'})
                if accepted:
                    links.extend(accepted)
                    break  # padding must not add spurious neighbours to a direct match
        direct = {link['component'] for link in links if link['method'] == 'direct'}
        links = [link for link in links
                 if link['method'] == 'direct' or link['component'] not in direct]
        # A one-cell tracer speck peeling off a 1,000-cell cloud is not evidence
        # that its main track needs a new identity. Compare branches with their
        # own largest sibling, never a global size cutoff: young clouds survive.
        # Small components still get tracks; only these weak lineage edges drop.
        largest_child, largest_parent = {}, {}
        for link in links:
            pi, ci = link['parent'], link['component']
            largest_child[pi] = max(largest_child.get(pi, 0), len(components[ci]))
            largest_parent[ci] = max(largest_parent.get(ci, 0), len(self._previous[pi]['cells']))
        fraction = self.config.min_branch_fraction
        return [link for link in links
                if len(components[link['component']]) >= largest_child[link['parent']] * fraction
                and len(self._previous[link['parent']]['cells']) >= largest_parent[link['component']] * fraction]

    def _statistics(self, cells, dp, vr):
        i, j, k = np.unravel_index(cells, self.shape)
        # Physical spherical-cell volumes, up to a constant angular factor.
        # This is a geometric centroid, not a DP-derived centre of mass.
        weight = np.maximum(np.cos(np.deg2rad(self.lat[j])), 1e-12) * self.rad[k] ** 2
        angle = np.deg2rad(self.lon[i])
        x, y = np.sum(weight * np.cos(angle)), np.sum(weight * np.sin(angle))
        longitude = float(np.rad2deg(np.arctan2(y, x)) % 360) if np.hypot(x, y) > 1e-10 * weight.sum() else None
        speed = vr.ravel()[cells]
        valid = speed[np.isfinite(speed) & (speed >= 0)]
        return {
            'cellCount': len(cells),
            'centroid': {'lon': longitude, 'lat': float(np.average(self.lat[j], weights=weight)),
                         'rAU': float(np.average(self.rad[k], weights=weight))},
            'radialRangeAU': [float(self.rad[k].min()), float(self.rad[k].max())],
            'latitudeRangeDeg': [float(self.lat[j].min()), float(self.lat[j].max())],
            'peakDp': float(dp.ravel()[cells].max()),
            'medianVrKms': float(np.median(valid)) if len(valid) else None,
        }

    def update(self, dp, vr, time):
        if not isinstance(time, datetime) or time.tzinfo is None or time.utcoffset() is None:
            raise ValueError('frame time must be a timezone-aware datetime')
        seconds = time.timestamp()
        dt = seconds - self._time if self._time is not None else None
        if dt is not None and dt <= 0:
            raise ValueError('frame times must increase strictly')
        dp, vr = np.asarray(dp), np.asarray(vr)
        if dp.shape != self.shape or vr.shape != self.shape:
            raise ValueError('DP and Vr must match the tracking grid')
        labels, components = connected_components(
            np.isfinite(dp) & (dp > self.config.dp_threshold), self.config.min_cells)
        gap = dt is not None and dt > self.config.max_gap_hours * 3600
        links = self._links(labels, components, dt) if self._previous and not gap else []
        incoming = [[] for _ in components]
        outgoing = [[] for _ in self._previous]
        for link in links:
            incoming[link['component']].append(link)
            outgoing[link['parent']].append(link['component'])

        records, previous = [], []
        remap = np.zeros(len(components) + 1, dtype=LABEL_DTYPE)
        for ci, cells in enumerate(components):
            parents = [self._previous[link['parent']]['record'] for link in incoming[ci]]
            split = any(len(outgoing[link['parent']]) > 1 for link in incoming[ci])
            if len(parents) == 1 and not split:
                track_id = parents[0]['trackId']
                event = 'continue'
                parent_ids = parents[0]['parentTrackIds']
            else:
                if self._next_id > np.iinfo(LABEL_DTYPE).max:
                    raise OverflowError(f'too many blob tracks for {LABEL_DTYPE.__name__} labels')
                track_id, self._next_id = self._next_id, self._next_id + 1
                parent_ids = sorted(p['trackId'] for p in parents)
                event = ('reconfigure' if len(parents) > 1 and split else
                         'merge' if len(parents) > 1 else 'split' if split else
                         'initial' if dt is None else 'gap' if gap else 'birth')
            origins = sorted({root for p in parents for root in p['originTrackIds']}) if parents else [track_id]
            record = {
                'trackId': track_id, 'event': event, 'parentTrackIds': parent_ids,
                'originTrackIds': origins,
                'links': [{'trackId': self._previous[link['parent']]['record']['trackId'],
                           'overlap': link['overlap'], 'method': link['method']} for link in incoming[ci]],
                **self._statistics(cells, dp, vr),
            }
            records.append(record)
            previous.append({'cells': cells, 'speed': vr.ravel()[cells].copy(), 'record': record})
            remap[ci + 1] = track_id
        live_ids = {r['trackId'] for r in records}
        ended = sorted(p['record']['trackId'] for p in self._previous if p['record']['trackId'] not in live_ids)
        self._previous, self._time = previous, seconds
        return remap[labels], {'regions': records, 'endedTrackIds': ended, 'gapReset': gap}
