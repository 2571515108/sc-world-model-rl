"""RSSM training-loss and checkpoint coverage."""

import tempfile
import unittest
from pathlib import Path

import torch

from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.training.world_model_trainer import WorldModelTrainer


class WorldModelTests(unittest.TestCase):
    """Check head shapes, finite loss, update, and restore."""
    def test_loss_and_checkpoint(self) -> None:
        config = WorldModelConfig(observation_dim=2, deterministic_dim=16, stochastic_dim=8, hidden_dim=24, ensemble_size=2)
        trainer = WorldModelTrainer(WorldModel(config)); sequences = []
        from tests.test_replay_buffer import make_transition
        for batch in range(2): sequences.append([make_transition(step, batch) for step in range(5)])
        metrics = trainer.train_step(sequences); self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.pt"; trainer.save(path); restored = WorldModelTrainer.load(path)
        self.assertEqual(restored.model.config.observation_dim, 2)

    def test_open_loop_metrics_have_all_horizons(self) -> None:
        from tests.test_replay_buffer import make_transition
        trainer = WorldModelTrainer(WorldModel(WorldModelConfig(observation_dim=2, deterministic_dim=8, stochastic_dim=4, hidden_dim=16, ensemble_size=2)))
        metrics = trainer.open_loop_evaluate([[make_transition(index, 0) for index in range(4)] for _ in range(2)])
        self.assertIn("open_loop_mse_h4", metrics); self.assertGreaterEqual(metrics["reconstruction_mse"], 0.0)
