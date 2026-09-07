"""Synthetic motion/topology cases; no network or downloaded NOAA data needed."""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from regions import AU_KM, BlobTracker, TrackingConfig, connected_components


START = datetime(2026, 9, 1, tzinfo=timezone.utc)


class BlobTrackingTests(unittest.TestCase):
    def setUp(self):
        self.lon = np.arange(12) * 30.0
        self.lat = np.array([-20., 0., 20.])
        self.rad = np.arange(12) * 0.1 + 0.1
        self.shape = (12, 3, 12)

    def tracker(self, **config):
        return BlobTracker(self.lon, self.lat, self.rad, TrackingConfig(**config))

    def field(self):
        return np.zeros(self.shape, dtype=float)

    def frame(self, tracker, dp, hours=0, speed=0):
        vr = np.full(self.shape, speed, dtype=float)
        return tracker.update(dp, vr, START + timedelta(hours=hours))

    def test_periodic_longitude_but_open_latitude_and_radius(self):
        mask = self.field().astype(bool)
        mask[0, 1, 5] = mask[-1, 1, 5] = True
        labels, components = connected_components(mask)
        self.assertEqual(len(components), 1)
        self.assertEqual(labels[0, 1, 5], labels[-1, 1, 5])
        mask[:] = False
        mask[3, 0, 5] = mask[3, -1, 5] = True
        mask[6, 1, 0] = mask[6, 1, -1] = True
        self.assertEqual(len(connected_components(mask)[1]), 4)

    def test_diagonal_cells_are_separate(self):
        mask = self.field().astype(bool)
        mask[1, 1, 4] = mask[2, 1, 5] = True
        self.assertEqual(len(connected_components(mask)[1]), 2)

    def test_threshold_uses_floats_and_rejects_nonfinite_tracer(self):
        dp = self.field()
        dp[1, 1, 5:10] = [0.25, 0.251, np.nan, np.inf, -np.inf]
        labels, record = self.frame(self.tracker(), dp)
        self.assertEqual(np.count_nonzero(labels), 1)
        self.assertAlmostEqual(record['regions'][0]['peakDp'], 0.251)

    def test_fast_radial_motion_keeps_identity_without_original_overlap(self):
        tracker = self.tracker(tolerance_cells=0)
        dp = self.field()
        dp[2:4, 1, 1:3] = 1
        speed = 0.3 * AU_KM / 3600
        first, _ = self.frame(tracker, dp, speed=speed)
        dp[:] = 0
        dp[2:4, 1, 4:6] = 1
        second, record = self.frame(tracker, dp, hours=1, speed=speed)
        self.assertEqual(first[2, 1, 1], second[2, 1, 4])
        self.assertEqual(record['regions'][0]['event'], 'continue')
        self.assertEqual(record['regions'][0]['links'][0]['method'], 'direct')

    def test_co_directional_clouds_and_new_injection_do_not_share_ids(self):
        tracker = self.tracker()
        dp = self.field()
        dp[2, 1, 1] = dp[2, 1, 6] = 1
        speed = 0.2 * AU_KM / 3600
        first, _ = self.frame(tracker, dp, speed=speed)
        dp[:] = 0
        dp[2, 1, 0] = dp[2, 1, 3] = dp[2, 1, 8] = 1
        second, _ = self.frame(tracker, dp, hours=1, speed=speed)
        self.assertEqual(first[2, 1, 1], second[2, 1, 3])
        self.assertEqual(first[2, 1, 6], second[2, 1, 8])
        self.assertNotIn(second[2, 1, 0], first[first > 0])

    def test_known_longitude_drift_crosses_seam(self):
        tracker = self.tracker(tolerance_cells=0, longitude_rate_deg_per_day=30 * 24)
        dp = self.field()
        dp[-1, 1, 5] = 1
        first, _ = self.frame(tracker, dp)
        dp[:] = 0
        dp[0, 1, 5] = 1
        second, record = self.frame(tracker, dp, hours=1)
        self.assertEqual(first[-1, 1, 5], second[0, 1, 5])
        self.assertEqual(record['regions'][0]['event'], 'continue')

    def test_merge_then_split_records_ancestry_without_inventing_identity(self):
        tracker = self.tracker(tolerance_cells=0)
        dp = self.field()
        dp[2, 1, 3:5] = dp[2, 1, 6:8] = 1
        _, initial = self.frame(tracker, dp)
        roots = [r['trackId'] for r in initial['regions']]
        dp[2, 1, 5] = 1
        _, merged = self.frame(tracker, dp, hours=1)
        region = merged['regions'][0]
        self.assertEqual(region['event'], 'merge')
        self.assertEqual(region['parentTrackIds'], roots)
        self.assertEqual(merged['endedTrackIds'], roots)
        dp[2, 1, 5] = 0
        _, split = self.frame(tracker, dp, hours=2)
        self.assertEqual(len(split['regions']), 2)
        for child in split['regions']:
            self.assertEqual(child['event'], 'split')
            self.assertEqual(child['parentTrackIds'], [region['trackId']])
            self.assertEqual(child['originTrackIds'], roots)
            self.assertNotIn(child['trackId'], roots)

    def test_many_to_many_reconfiguration_is_explicit(self):
        tracker = self.tracker(tolerance_cells=0)
        dp = self.field()
        dp[1, 1, 3:8] = dp[3, 1, 3:8] = 1
        _, first = self.frame(tracker, dp)
        dp[:] = 0
        dp[1:4, 1, 3] = dp[1:4, 1, 7] = 1
        _, second = self.frame(tracker, dp, hours=1)
        parents = [r['trackId'] for r in first['regions']]
        self.assertEqual(len(second['regions']), 2)
        for region in second['regions']:
            self.assertEqual(region['event'], 'reconfigure')
            self.assertEqual(region['parentTrackIds'], parents)

    def test_small_fragment_detaches_and_rejoins_without_resetting_main_track(self):
        tracker = self.tracker(tolerance_cells=0)
        dp = self.field()
        dp[2:6, :, 3:9] = 1
        dp[6:8, 1, 5] = 1
        first, _ = self.frame(tracker, dp)
        main_id = first[2, 1, 5]
        dp[6, 1, 5] = 0
        split, record = self.frame(tracker, dp, hours=1)
        self.assertEqual(split[2, 1, 5], main_id)
        self.assertNotEqual(split[7, 1, 5], main_id)
        self.assertGreater(split[7, 1, 5], 0)  # speck is retained
        self.assertEqual(record['regions'][0]['event'], 'continue')
        dp[6, 1, 5] = 1
        joined, record = self.frame(tracker, dp, hours=2)
        self.assertEqual(joined[2, 1, 5], main_id)
        self.assertEqual(record['regions'][0]['event'], 'continue')

    def test_small_cloud_can_grow_without_a_global_size_cutoff(self):
        tracker = self.tracker(tolerance_cells=0)
        dp = self.field()
        dp[3, 1, 5] = 1
        first, _ = self.frame(tracker, dp)
        dp[2:6, :, 3:9] = 1
        grown, record = self.frame(tracker, dp, hours=1)
        self.assertEqual(first[3, 1, 5], grown[3, 1, 5])
        self.assertEqual(record['regions'][0]['event'], 'continue')

    def test_full_volume_keeps_off_slice_cloud(self):
        tracker = self.tracker()
        dp = self.field()
        dp[2, 1, 5] = 1
        first, _ = self.frame(tracker, dp)
        dp[:] = 0
        dp[2, 2, 5] = 1
        second, record = self.frame(tracker, dp, hours=1)
        self.assertEqual(np.count_nonzero(second[:, 1, :]), 0)
        self.assertEqual(first[2, 1, 5], second[2, 2, 5])
        self.assertEqual(record['regions'][0]['links'][0]['method'], 'tolerant')

    def test_direct_link_does_not_capture_nearby_new_blob(self):
        tracker = self.tracker()
        dp = self.field()
        dp[3, 1, 5] = 1
        first, _ = self.frame(tracker, dp)
        # Diagonal neighbour is within the tolerance box but a separate cloud.
        dp[4, 1, 6] = 1
        second, record = self.frame(tracker, dp, hours=1)
        self.assertEqual(first[3, 1, 5], second[3, 1, 5])
        self.assertNotEqual(second[3, 1, 5], second[4, 1, 6])
        self.assertEqual([r['event'] for r in record['regions']], ['continue', 'birth'])

    def test_exiting_material_is_not_clamped_to_boundary(self):
        tracker = self.tracker()
        dp = self.field()
        dp[2, 1, -1] = 1
        first, _ = self.frame(tracker, dp, speed=0.3 * AU_KM / 3600)
        second, record = self.frame(tracker, dp, hours=1)
        self.assertNotEqual(first[2, 1, -1], second[2, 1, -1])
        self.assertEqual(record['regions'][0]['event'], 'birth')

    def test_gap_and_empty_frames_do_not_claim_continuity(self):
        tracker = self.tracker()
        dp = self.field()
        dp[2, 1, 5] = 1
        first, _ = self.frame(tracker, dp)
        second, record = self.frame(tracker, dp, hours=7)
        self.assertTrue(record['gapReset'])
        self.assertEqual(record['regions'][0]['event'], 'gap')
        self.assertNotEqual(first[2, 1, 5], second[2, 1, 5])
        empty, record = self.frame(tracker, self.field(), hours=8)
        self.assertFalse(np.any(empty))
        self.assertEqual(record['endedTrackIds'], [int(second[2, 1, 5])])
        third, record = self.frame(tracker, dp, hours=9)
        self.assertNotEqual(second[2, 1, 5], third[2, 1, 5])

    def test_missing_velocity_does_not_fabricate_prediction(self):
        tracker = self.tracker()
        dp = self.field()
        dp[2, 1, 5] = 1
        first, initial = self.frame(tracker, dp, speed=np.nan)
        second, record = self.frame(tracker, dp, hours=1)
        self.assertIsNone(initial['regions'][0]['medianVrKms'])
        self.assertNotEqual(first[2, 1, 5], second[2, 1, 5])
        self.assertEqual(record['regions'][0]['links'], [])

    def test_centroid_respects_longitude_seam(self):
        dp = self.field()
        dp[0, 1, 5] = dp[-1, 1, 5] = 1
        _, record = self.frame(self.tracker(), dp)
        self.assertAlmostEqual(record['regions'][0]['centroid']['lon'], 345)

    def test_descending_latitude_preserves_coordinates(self):
        tracker = BlobTracker(self.lon, self.lat[::-1], self.rad)
        dp = self.field()
        dp[2, 0, 5] = 1
        labels, record = self.frame(tracker, dp)
        self.assertGreater(labels[2, 0, 5], 0)
        self.assertEqual(record['regions'][0]['centroid']['lat'], 20)

    def test_configuration_and_time_validation(self):
        for settings in ({'dp_threshold': 0}, {'min_overlap': 2}, {'max_gap_hours': 0},
                         {'longitude_rate_deg_per_day': np.nan}, {'min_cells': 0},
                         {'tolerance_cells': 0.5}):
            with self.assertRaises(ValueError):
                TrackingConfig(**settings)
        tracker = self.tracker()
        self.frame(tracker, self.field())
        with self.assertRaises(ValueError):
            self.frame(tracker, self.field())
        with self.assertRaises(ValueError):
            tracker.update(self.field(), self.field(), datetime(2026, 9, 1))
        with self.assertRaises(ValueError):
            tracker.update(self.field()[0], self.field(), START + timedelta(hours=1))
        with self.assertRaises(ValueError):
            BlobTracker(self.lon, self.lat, [0.1, 0.2, 0.4])

    def test_artifact_tag_changes_with_configuration(self):
        self.assertEqual(TrackingConfig().artifact_tag(), TrackingConfig().artifact_tag())
        self.assertNotEqual(TrackingConfig().artifact_tag(), TrackingConfig(dp_threshold=0.3).artifact_tag())


if __name__ == '__main__':
    unittest.main()
