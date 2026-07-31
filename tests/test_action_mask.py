"""Macro action legality tests."""

import unittest

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.envs.synthetic_macro_env import SyntheticMacroEnv


class ActionMaskTests(unittest.TestCase):
    """Ensure masking reflects concrete resource and prerequisite constraints."""

    def test_mask_has_fixed_shape_and_no_empty_state(self) -> None:
        env = SyntheticMacroEnv(); env.reset(); mask = env.get_action_mask()
        self.assertEqual(mask.shape, (len(MacroAction),)); self.assertEqual(mask.dtype, np.bool_); self.assertTrue(mask.any())

    def test_opening_cannot_train_army_before_barracks(self) -> None:
        env = SyntheticMacroEnv(); env.reset()
        self.assertFalse(env.get_action_mask()[MacroAction.TRAIN_BASIC_ARMY])
        env.step(MacroAction.BUILD_BARRACKS)
        self.assertTrue(env.get_action_mask()[MacroAction.TRAIN_BASIC_ARMY])

    def test_illegal_action_is_rejected(self) -> None:
        env = SyntheticMacroEnv(); env.reset()
        with self.assertRaises(ValueError): env.step(MacroAction.TRAIN_ANTI_AIR)
