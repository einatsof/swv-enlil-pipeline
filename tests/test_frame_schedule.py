"""Which pv-tim frames the pipeline downloads (pipeline.frame_schedule).

The schedule encodes two facts that are invisible from the code alone: cones are
injected in the hindcast, and a step above 5 frames trips the tracker's
max_gap_hours because frame spacing jitters around 3600 s. Both are cheap to
break by editing a constant, so they are asserted here.
"""

import unittest

import pipeline
from regions import TrackingConfig

# Worst measured spacing on 20260903_58484: stride-3 deltas ran 10700-10903 s.
WORST_FRAME_SECONDS = 10903.0 / 3


def keys(n):
    return [f'wsa_enlil.20260907_58498/pv-ready-data-1/pv-tim.{i:04d}.nc'
            for i in range(n)]


def nums(selected):
    return [int(k.split('pv-tim.')[1][:4]) for k in selected]


class FrameScheduleTest(unittest.TestCase):
    def test_standard_run_is_hourly_then_three_hourly(self):
        got = nums(pipeline.frame_schedule(keys(pipeline.STANDARD_FRAMES)))
        self.assertEqual(got[:pipeline.HOURLY_THROUGH + 1],
                         list(range(pipeline.HOURLY_THROUGH + 1)))
        tail = got[pipeline.HOURLY_THROUGH + 1:]
        self.assertEqual(tail, list(range(pipeline.HOURLY_THROUGH + pipeline.TAIL_STRIDE,
                                          pipeline.STANDARD_FRAMES, pipeline.TAIL_STRIDE)))
        self.assertEqual(len(got), 113)

    def test_every_cone_injection_frame_is_sampled_hourly(self):
        """Cones fire in the hindcast, so it must be sampled at full cadence.

        Frame N is rundate + (N - 48) h and every `cme_time` observed across
        20260905_58489/58491, 20260906_58495 and 20260907_58498 was <= rundate:
        cones come from observed events, so they are always in the past at run
        time. The earliest lands near frame 0, the latest at rundate itself.
        """
        got = set(nums(pipeline.frame_schedule(keys(pipeline.STANDARD_FRAMES))))
        self.assertTrue(set(range(0, 49)).issubset(got))

    def test_no_step_can_reset_the_tracker(self):
        got = nums(pipeline.frame_schedule(keys(pipeline.STANDARD_FRAMES)))
        widest = max(b - a for a, b in zip(got, got[1:]))
        limit = TrackingConfig().max_gap_hours * 3600
        self.assertLess(widest * WORST_FRAME_SECONDS, limit,
                        'a step this wide drops every track link (regions.update)')

    def test_last_frame_always_ships(self):
        for n in (pipeline.STANDARD_FRAMES, 45, 20):
            got = nums(pipeline.frame_schedule(keys(n)))
            self.assertEqual(got[-1], n - 1)

    def test_non_standard_run_falls_back_to_even_sampling(self):
        # 20260905_58489 published 45 frames; describe_run accepts >= 20, so a
        # truncated run reaches here and must not be indexed by position.
        got = nums(pipeline.frame_schedule(keys(45)))
        self.assertEqual(got[:3], [0, pipeline.FALLBACK_STRIDE, 2 * pipeline.FALLBACK_STRIDE])
        self.assertLessEqual(max(b - a for a, b in zip(got, got[1:])), 5)

    def test_gaps_in_frame_numbering_are_not_treated_as_standard(self):
        broken = keys(pipeline.STANDARD_FRAMES)
        del broken[10]
        got = nums(pipeline.frame_schedule(broken))
        self.assertEqual(got[1], 3)


if __name__ == '__main__':
    unittest.main()
