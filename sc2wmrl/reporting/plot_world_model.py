"""World-model plot entry points built on the shared experiment plotter."""

from __future__ import annotations

from pathlib import Path

from .plot_training import ExperimentPlotter
from .run_loader import RunData


def plot_world_model_metrics(runs: list[RunData], output_dir: Path) -> list[Path]:
    """Plot all available world-model loss and prediction metrics."""
    result: list[Path] = []; plotter = ExperimentPlotter()
    for metric, filename in (("total_world_model_loss", "world_model_loss_curve"), ("one_step_state_error", "multi_step_prediction_error"),
                             ("ensemble_disagreement", "uncertainty_vs_prediction_error"), ("uncertainty_error_correlation", "uncertainty_calibration_curve")):
        try: result += plotter.plot_metric(runs, metric, output_dir, filename=filename)
        except ValueError: continue
    return result
