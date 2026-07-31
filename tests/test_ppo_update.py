"""PPO mask, update, and checkpoint tests (skipped when PyTorch is absent)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig
from sc2wmrl.envs.base_macro_env import MacroAction


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class PPOTests(unittest.TestCase):
    """Exercise a small CPU PPO update and resumable checkpoint."""
    def test_action_mask_update_and_restore(self) -> None:
        agent = PPOAgent(4, PPOConfig(hidden_dim=16, epochs=1, minibatch_size=4)); masks = np.zeros((8, len(MacroAction)), dtype=np.bool_); masks[:, :2] = True
        action, _, _ = agent.act(np.zeros(4, dtype=np.float32), masks[0]); self.assertIn(action, (0, 1))
        stats = agent.update({"observations": np.zeros((8, 4), dtype=np.float32), "action_masks": masks, "actions": np.zeros(8, dtype=np.int64),
                              "old_log_probs": np.zeros(8, dtype=np.float32), "advantages": np.ones(8, dtype=np.float32), "returns": np.zeros(8, dtype=np.float32)})
        self.assertTrue(np.isfinite(list(stats.values())).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"; agent.save(path, training_state={"step": 2}); restored, metadata = PPOAgent.load(path)
        self.assertEqual(restored.observation_dim, 4); self.assertEqual(metadata["step"], 2)
