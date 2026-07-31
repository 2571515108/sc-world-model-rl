"""Smoothing and aggregate-statistics tests."""

import unittest

import numpy as np

from sc2wmrl.reporting.smoothing import smooth
from sc2wmrl.reporting.statistics import aggregate_series


class StatisticsTests(unittest.TestCase):
    """Ensure display smoothing preserves source and aggregation is aligned."""
    def test_smoothing_methods(self) -> None:
        source = np.array([1.0, 3.0, 5.0]); self.assertTrue(np.array_equal(smooth(source), source)); self.assertEqual(smooth(source, "moving_average", window=2)[1], 2.0); self.assertLess(smooth(source, "ema", factor=0.5)[1], 3.0)
    def test_confidence_band(self) -> None:
        result = aggregate_series([(np.array([0, 2]), np.array([1.0, 3.0])), (np.array([0, 1, 2]), np.array([3.0, 3.0, 3.0]))])
        self.assertEqual(result.steps, [0, 1, 2]); self.assertEqual(result.mean[0], 2.0)
