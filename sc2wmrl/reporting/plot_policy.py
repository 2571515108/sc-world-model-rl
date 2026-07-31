"""Policy behavior plots from evaluation episode/action records."""

from __future__ import annotations

from pathlib import Path

from .plot_training import ExperimentPlotter
from .run_loader import RunData


def plot_policy_analysis(runs: list[RunData], output_dir: Path) -> list[Path]:
    """Plot available action/reward/macro behavior metrics without fabricated data."""
    result: list[Path] = []; plotter = ExperimentPlotter()
    for metric, filename in (("action_repeat_rate", "macro_action_distribution"), ("army_value", "army_value_progression"),
                             ("base_count", "base_count_progression"), ("map_control_score", "map_control_progression")):
        try: result += plotter.plot_metric(runs, metric, output_dir, filename=filename)
        except ValueError: continue
    return result
