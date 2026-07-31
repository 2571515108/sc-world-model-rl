"""Imagined actor-critic smoke test with bounded finite trajectories."""

import unittest

import numpy as np
import torch

from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.critic import LatentCritic
from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.training.imagination_trainer import ImaginationConfig, ImaginationTrainer


class ImaginationTests(unittest.TestCase):
    """Ensure policy/critic can update from model-generated trajectories."""
    def test_rollout_and_update_are_finite(self) -> None:
        model = WorldModel(WorldModelConfig(observation_dim=4, deterministic_dim=8, stochastic_dim=4, hidden_dim=16, ensemble_size=2))
        trainer = ImaginationTrainer(model, LatentActor(12, 20, 16), LatentCritic(12, 16), ImaginationConfig(horizon=4, uncertainty_threshold=1e6))
        masks = torch.ones(3, 3, 20, dtype=torch.bool); rollout = trainer.rollout(torch.zeros(3, 3, 4), torch.zeros(3, 3, dtype=torch.long), masks)
        metrics = trainer.update(rollout); self.assertTrue(np.isfinite(list(metrics.values())).all()); self.assertLessEqual(metrics["imagined_length"], 4.0)
