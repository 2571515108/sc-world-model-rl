"""Real adapter protocol test without requiring an installed SC2 client."""

import unittest

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.envs.real_sc2_macro_env import RealSC2MacroEnv, RealSC2UnavailableError


class Backend:
    """Minimal in-memory backend confirming the production adapter contract."""
    def __init__(self) -> None: self.calls = 0
    def reset_game(self, seed=None): return {"minerals": 500, "maximum_supply": 15, "current_supply": 12}, {"opponent_id": "mock"}
    def execute_macro(self, action, duration): self.calls += 1; return {"minerals": 450, "maximum_supply": 15, "current_supply": 12}, 0.0, False, True, {}
    def get_action_mask(self): mask = np.zeros(len(MacroAction), dtype=np.bool_); mask[MacroAction.NO_OP] = True; return mask
    def close(self): return None


class RealAdapterTests(unittest.TestCase):
    """Ensure the optional real environment fails explicitly and adapters work."""
    def test_none_backend_explains_requirement(self) -> None:
        with self.assertRaises(RealSC2UnavailableError): RealSC2MacroEnv(None)
    def test_protocol_backend_steps(self) -> None:
        env = RealSC2MacroEnv(Backend()); observation, _ = env.reset(); self.assertEqual(observation.shape, (env.observation_dim,))
        _, _, _, truncated, _ = env.step(MacroAction.NO_OP); self.assertTrue(truncated)
