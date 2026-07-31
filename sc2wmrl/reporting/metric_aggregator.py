"""Metric-specific run aggregation over actual steps rather than row indices."""

from __future__ import annotations

import numpy as np

from .run_loader import RunData
from .statistics import AggregateSeries, aggregate_series


class MetricAggregator:
    """Collects one metric from multiple independent runs and aligns their x-axis."""
    def aggregate(self, runs: list[RunData], metric_name: str, axis: str = "global_step") -> AggregateSeries:
        """Aggregate non-empty metric series and reject unsupported axes."""
        if axis not in {"global_step", "episode"}: raise ValueError("axis must be global_step or episode")
        series = []
        for run in runs:
            rows = sorted((row for row in run.metrics if row["metric_name"] == metric_name), key=lambda row: row[axis])
            if rows:
                steps = np.asarray([row[axis] for row in rows]); values = np.asarray([row["value"] for row in rows])
                unique, indices = np.unique(steps, return_index=True); series.append((unique, values[indices]))
        if not series: raise ValueError(f"metric {metric_name!r} is missing from every run")
        return aggregate_series(series)
