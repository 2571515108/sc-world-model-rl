"""League-analysis plot entry points."""

from __future__ import annotations

from pathlib import Path

from .plot_training import ExperimentPlotter
from .run_loader import RunData


def plot_league_analysis(runs: list[RunData], output_dir: Path) -> list[Path]:
    """Plot available Elo and opponent win-rate series."""
    result: list[Path] = []; plotter = ExperimentPlotter()
    for metric, filename in (("agent_elo", "elo_progression"), ("worst_case_opponent_win_rate", "opponent_win_rate_heatmap"),
                             ("historical_policy_win_rate", "historical_policy_retention")):
        try: result += plotter.plot_metric(runs, metric, output_dir, filename=filename)
        except ValueError: continue
    return result
