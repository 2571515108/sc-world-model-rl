"""Fallback policy legality and empty-mask recovery tests."""

import unittest

import numpy as np

from sc2wmrl.deployment.safety_fallback import SafetyFallbackPolicy
from sc2wmrl.envs.base_macro_env import MacroAction


class SafetyFallbackTests(unittest.TestCase):
    """Ensure fallback prefers safe legal actions and handles malformed masks."""
    def test_priority_and_empty_mask(self) -> None:
        fallback = SafetyFallbackPolicy(); mask = np.zeros(len(MacroAction), dtype=bool); mask[MacroAction.TRAIN_WORKERS] = True
        self.assertEqual(fallback.select_action(np.zeros(1), mask), int(MacroAction.TRAIN_WORKERS)); self.assertEqual(fallback.select_action(np.zeros(1), np.zeros(len(MacroAction), dtype=bool)), int(MacroAction.NO_OP))
