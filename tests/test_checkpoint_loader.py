"""Strict deployable-checkpoint validation test."""

import tempfile
import unittest
from pathlib import Path

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig
from sc2wmrl.deployment.checkpoint_loader import CheckpointLoader
from sc2wmrl.envs.base_macro_env import MacroAction


class CheckpointLoaderTests(unittest.TestCase):
    """Reject dimension mismatches while accepting a compatible checkpoint."""
    def test_dimension_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"; agent = PPOAgent(4, PPOConfig(hidden_dim=8), device="cpu")
            agent.save(path, training_state={"macro_action_names": [action.name for action in MacroAction]})
            loaded = CheckpointLoader().load_ppo(path, expected_observation_dim=4, device="cpu")
            self.assertEqual(loaded.action_names[0], "NO_OP")
            with self.assertRaises(ValueError): CheckpointLoader().load_ppo(path, expected_observation_dim=5, device="cpu")
