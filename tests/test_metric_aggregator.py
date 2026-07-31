"""Experiment logger, loader, and aligned aggregation tests."""

import tempfile
import unittest
from pathlib import Path

from sc2wmrl.reporting.experiment_logger import ExperimentLogger
from sc2wmrl.reporting.metric_aggregator import MetricAggregator
from sc2wmrl.reporting.run_loader import RunLoader


class MetricAggregatorTests(unittest.TestCase):
    """Verify persistent records aggregate on actual step coordinates."""
    def test_logger_round_trip_and_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            with ExperimentLogger(first, config={"seed": 0}) as logger:
                logger.log_scalar("episode_reward", 1.0, step=10, episode=1, phase="ppo"); logger.log_scalar("episode_reward", 3.0, step=30, episode=2, phase="ppo")
            with ExperimentLogger(second, config={"seed": 1}) as logger:
                logger.log_scalar("episode_reward", 2.0, step=10, episode=1, phase="ppo"); logger.log_scalar("episode_reward", 6.0, step=20, episode=2, phase="ppo")
            series = MetricAggregator().aggregate([RunLoader().load(first), RunLoader().load(second)], "episode_reward")
        self.assertEqual(series.steps, [10, 20, 30]); self.assertEqual(series.run_count, 2); self.assertEqual(series.mean[1], 4.0)
