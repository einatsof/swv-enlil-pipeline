"""Cone -> blob-track attribution (extract.ConeAttributor).

The cases that matter are the ones single-frame geometry gets wrong: a cone
injected into material that already exists, two cones sharing a direction, and a
cone that predates the run's first saved frame.
"""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from extract import ATTRIBUTION_WINDOW_HOURS, ConeAttributor
from regions import AU_KM, BlobTracker

LON = np.arange(36) * 10.0                 # 10 deg cells, full circle
LAT = np.linspace(58.0, -58.0, 30)         # NOAA pv order: north -> south
RAD = np.linspace(0.1125, 1.6875, 64)
EARTH_LON = 180.0
START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def cone(time_h, lat, lon, speed=600.0, half=20.0):
    return {'time': (START + timedelta(hours=time_h)).strftime('%Y-%m-%dT%H:%MZ'),
            'latitude': lat, 'longitude': lon, 'halfAngle': half, 'speed': speed}


def blob(dp, lat, lon, k_from=0, k_to=2, half=15.0):
    """Paint DP into the wedge a cone at (lat, lon) occupies."""
    i = np.flatnonzero(np.abs(((LON - (EARTH_LON + lon) + 180) % 360) - 180) <= half)
    j = np.flatnonzero(np.abs(LAT - lat) <= half)
    dp[np.ix_(i, j, np.arange(k_from, k_to + 1))] = 1.0
    return dp


class ConeAttributionTests(unittest.TestCase):
    def run_frames(self, cones, painters, hours=3.0):
        """Drive tracker + attributor over frames painted by `painters`."""
        tracker = BlobTracker(LON, LAT, RAD)
        attributor = ConeAttributor(cones, LON, LAT, RAD, EARTH_LON)
        vr = np.full((len(LON), len(LAT), len(RAD)), 1.0)  # ~stationary
        frames = []
        for fi, paint in enumerate(painters):
            dp = paint(np.zeros((len(LON), len(LAT), len(RAD))))
            when = START + timedelta(hours=hours * fi)
            labels, tracked = tracker.update(dp, vr, when)
            attributor.update(f'{fi * 3:04d}', when, labels, tracked['regions'])
            frames.append(tracked['regions'])
        return attributor, frames

    def test_cone_injected_into_existing_material_is_still_attributed(self):
        """The case birth-matching cannot see: no new component is created."""
        first, second = cone(0.5, 0.0, -40.0), cone(3.5, 0.0, -40.0, speed=900.0)
        attributor, frames = self.run_frames(
            [first, second],
            [lambda dp: dp,                                   # empty
             lambda dp: blob(dp, 0.0, -40.0),                 # first cone arrives
             lambda dp: blob(dp, 0.0, -40.0, k_to=4)])        # second joins it
        # One track throughout — the second cone never births its own component.
        self.assertEqual([len(f) for f in frames], [0, 1, 1])
        self.assertEqual(frames[1][0]['trackId'], frames[2][0]['trackId'])
        summary = attributor.summary()
        self.assertEqual(summary[0]['trackIds'], summary[1]['trackIds'])
        self.assertTrue(summary[1]['trackIds'], 'second cone must still be attributed')
        self.assertEqual(frames[2][0]['coneIdxs'], [0, 1])

    def test_co_directional_cones_share_one_blob_rather_than_splitting_it(self):
        """Two cones down the same line are one cloud, and both own it."""
        cones = [cone(0.5, -4.0, -28.0, speed=690.0), cone(3.5, -4.0, -28.0, speed=970.0)]
        _, frames = self.run_frames(
            cones, [lambda dp: dp, lambda dp: blob(dp, -4.0, -28.0), lambda dp: blob(dp, -4.0, -28.0)])
        self.assertEqual(len(frames[2]), 1)
        self.assertEqual(frames[2][0]['coneIdxs'], [0, 1])

    def test_separate_directions_do_not_borrow_each_others_cones(self):
        cones = [cone(0.5, 0.0, -40.0), cone(0.5, 0.0, 80.0)]
        _, frames = self.run_frames(
            cones, [lambda dp: dp,
                    lambda dp: blob(blob(dp, 0.0, -40.0), 0.0, 80.0)])
        owners = {tuple(r['coneIdxs']) for r in frames[1]}
        self.assertEqual(len(frames[1]), 2)
        self.assertEqual(owners, {(0,), (1,)})

    def test_cone_injected_before_the_first_saved_frame_is_reached(self):
        """Run spin-up: material is already mid-flight when frame 0 is written."""
        # 900 km/s for 12 h ~ 0.26 AU, so the cloud sits well off the boundary.
        early = cone(-12.0, 0.0, -40.0, speed=900.0)
        reach = int(round(900.0 * 12 * 3600 / AU_KM / (RAD[1] - RAD[0])))
        _, frames = self.run_frames(
            [early], [lambda dp: blob(dp, 0.0, -40.0, k_from=reach - 1, k_to=reach + 1)])
        self.assertEqual(len(frames[0]), 1)
        self.assertEqual(frames[0][0]['coneIdxs'], [0])

    def test_cone_with_no_material_reports_an_empty_list_not_a_guess(self):
        lonely = cone(0.5, 40.0, 120.0)
        attributor, frames = self.run_frames(
            [lonely], [lambda dp: dp, lambda dp: blob(dp, -30.0, -60.0)])
        self.assertEqual(attributor.summary()[0]['trackIds'], [])
        self.assertEqual(frames[1][0]['coneIdxs'], [])
        self.assertIn('note', attributor.summary()[0])

    def test_cone_without_direction_or_time_is_skipped_cleanly(self):
        broken = [{'time': None, 'latitude': 1.0, 'longitude': 2.0, 'halfAngle': 20, 'speed': 500},
                  {'time': '2026-09-01T00:30Z', 'latitude': None, 'longitude': None,
                   'halfAngle': None, 'speed': None}]
        attributor, frames = self.run_frames(broken, [lambda dp: blob(dp, 0.0, -40.0)])
        self.assertEqual([s['trackIds'] for s in attributor.summary()], [[], []])
        self.assertEqual(frames[0][0]['coneIdxs'], [])

    def test_a_cone_follows_its_cloud_through_a_merge(self):
        """Ancestry, end to end: the merged blob carries both cones."""
        cones = [cone(0.5, 0.0, -50.0), cone(0.5, 0.0, -10.0)]
        _, frames = self.run_frames(
            cones,
            [lambda dp: dp,
             lambda dp: blob(blob(dp, 0.0, -50.0, half=8), 0.0, -10.0, half=8),   # two clouds
             lambda dp: blob(dp, 0.0, -30.0, half=30)])                            # merged
        self.assertEqual(len(frames[1]), 2)
        self.assertEqual(len(frames[2]), 1)
        self.assertEqual(frames[2][0]['coneIdxs'], [0, 1])

    def test_a_split_gives_the_cone_to_every_substantial_fragment(self):
        """Shared DP cannot say which fragment kept the CME; claiming one would guess."""
        cones = [cone(0.5, 0.0, -30.0, half=40.0)]
        _, frames = self.run_frames(
            cones,
            [lambda dp: dp,
             lambda dp: blob(dp, 0.0, -30.0, half=30),
             lambda dp: blob(blob(dp, 0.0, -55.0, half=8), 0.0, -5.0, half=8)])
        self.assertEqual(len(frames[2]), 2)
        for region in frames[2]:
            self.assertEqual(region['coneIdxs'], [0])

    def test_a_sliver_breaking_off_does_not_inherit_the_cone(self):
        """Unbounded spread compounds: late in a run it put every cone on every blob."""
        cones = [cone(0.5, 0.0, -30.0, half=40.0)]
        def sliver(dp):
            blob(dp, 0.0, -30.0, half=30)            # the cloud, unchanged
            dp[0, 0, 40] = 1.0                       # one detached cell far away
            return dp
        _, frames = self.run_frames(
            cones, [lambda dp: dp, lambda dp: blob(dp, 0.0, -30.0, half=30), sliver])
        big = max(frames[2], key=lambda r: r['cellCount'])
        small = [r for r in frames[2] if r is not big]
        self.assertEqual(big['coneIdxs'], [0])
        for r in small:
            self.assertEqual(r['coneIdxs'], [], 'a sliver must not inherit the cone')


class AttributionTimingTests(unittest.TestCase):
    """Attribution must not depend on which frames the pipeline downloaded.

    Regression: `pipeline.frame_schedule` went hourly through the hindcast, which
    moved the first look at cone1 of 20260907_58497 from +2.08 h to +5 min. Its
    DP cloud had not cleared a radial cell yet (0 labelled cells in the wedge at
    frame 0004, 8 at 0005, 12 at 0006), so a one-shot read marked a real CME
    untraced and the monitor drew a density-inferred outline on top of the cloud
    its co-directional partner already owned. The coarse cadence was never right
    here either, only lucky: a cone injected just before a 3-hourly frame failed
    identically.
    """

    def run_at(self, cones, schedule):
        """Drive tracker + attributor over explicit (hours, painter) frames."""
        tracker = BlobTracker(LON, LAT, RAD)
        attributor = ConeAttributor(cones, LON, LAT, RAD, EARTH_LON)
        vr = np.full((len(LON), len(LAT), len(RAD)), 1.0)
        frames = []
        for hours, paint in schedule:
            dp = paint(np.zeros((len(LON), len(LAT), len(RAD))))
            when = START + timedelta(hours=hours)
            labels, tracked = tracker.update(dp, vr, when)
            attributor.update(f'{int(hours):04d}', when, labels, tracked['regions'])
            frames.append(tracked['regions'])
        return attributor, frames

    # Injected just after frame 0; its cloud is not labellable until ~1.5 h in.
    CONE = staticmethod(lambda: cone(0.1, -4.0, -28.0, speed=690.0))

    @staticmethod
    def _paint(hours):
        return (lambda dp: blob(dp, -4.0, -28.0)) if hours >= 2.0 else (lambda dp: dp)

    def test_frame_moments_after_injection_does_not_abandon_the_cone(self):
        hourly = [(h, self._paint(h)) for h in range(0, 7)]
        attributor, frames = self.run_at([self.CONE()], hourly)
        summary = attributor.summary()[0]
        self.assertTrue(summary['trackIds'],
                        'a look before the cloud exists must not settle the cone')
        self.assertEqual(frames[-1][0]['coneIdxs'], [0])

    def test_one_and_three_hourly_cadences_agree(self):
        hourly = [(h, self._paint(h)) for h in range(0, 7)]
        three = [(h, self._paint(h)) for h in (0, 3, 6)]
        a1, f1 = self.run_at([self.CONE()], hourly)
        a3, f3 = self.run_at([self.CONE()], three)
        self.assertEqual(bool(a1.summary()[0]['trackIds']), bool(a3.summary()[0]['trackIds']))
        self.assertEqual(f1[-1][0]['coneIdxs'], f3[-1][0]['coneIdxs'])

    def test_search_stops_when_the_window_closes(self):
        """The wait is bounded — it must not latch onto whatever turns up later."""
        late = ATTRIBUTION_WINDOW_HOURS + 2.0
        schedule = [(h, (lambda dp: blob(dp, -4.0, -28.0)) if h >= late else (lambda dp: dp))
                    for h in range(0, int(late) + 2)]
        attributor, frames = self.run_at([self.CONE()], schedule)
        summary = attributor.summary()[0]
        self.assertEqual(summary['trackIds'], [])
        self.assertEqual(summary['note'], 'no tracked material in the injection footprint')
        self.assertEqual(frames[-1][0]['coneIdxs'], [],
                         'material appearing after the window is not this cone')


if __name__ == '__main__':
    unittest.main()
