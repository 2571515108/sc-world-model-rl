"""Uncertainty gating regression test."""

import unittest

import torch

from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.critic import LatentCritic
from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.training.imagination_trainer import ImaginationConfig, ImaginationTrainer


class UncertaintyGateTests(unittest.TestCase):
    """A zero threshold must terminate before accepting imagined reward."""
    def test_rollout_stops_when_disagreement_exceeds_threshold(self) -> None:
        model = WorldModel(WorldModelConfig(observation_dim=3, deterministic_dim=8, stochastic_dim=4, hidden_dim=16, ensemble_size=2))
        trainer = ImaginationTrainer(model, LatentActor(12, 20, 16), LatentCritic(12, 16), ImaginationConfig(horizon=5, uncertainty_threshold=-1.0))
        rollout = trainer.rollout(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, 20, dtype=torch.bool))
        self.assertEqual(float(rollout["alive"].sum()), 0.0)
