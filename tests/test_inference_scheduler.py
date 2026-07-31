"""Real-time macro decision schedule tests."""

import unittest

from sc2wmrl.deployment.inference_scheduler import InferenceScheduler


class InferenceSchedulerTests(unittest.TestCase):
    """Validate periodic and urgent triggers."""
    def test_trigger_rules(self) -> None:
        scheduler = InferenceScheduler(32); self.assertTrue(scheduler.should_infer(0, urgent_event=False, skill_finished=False)); scheduler.mark_inference(0)
        self.assertFalse(scheduler.should_infer(31, urgent_event=False, skill_finished=False)); self.assertTrue(scheduler.should_infer(31, urgent_event=True, skill_finished=False)); self.assertTrue(scheduler.should_infer(32, urgent_event=False, skill_finished=False))
