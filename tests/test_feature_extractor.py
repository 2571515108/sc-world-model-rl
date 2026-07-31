"""Feature schema and normalization tests."""

import unittest

import numpy as np

from sc2wmrl.envs.synthetic_macro_env import SyntheticMacroEnv


class FeatureExtractorTests(unittest.TestCase):
    """Ensure observations are fixed and normalized on every reset."""

    def test_reset_observation_is_fixed_finite_and_bounded(self) -> None:
        env = SyntheticMacroEnv(); observation, _ = env.reset(seed=11)
        self.assertEqual(observation.shape, (env.observation_dim,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(((0.0 <= observation) & (observation <= 1.0)).all())

    def test_missing_enemy_position_has_presence_encoding(self) -> None:
        env = SyntheticMacroEnv(); observation, _ = env.reset(seed=11)
        index = env.extractor.spec.names.index("enemy_position_present")
        self.assertEqual(observation[index], 0.0)
