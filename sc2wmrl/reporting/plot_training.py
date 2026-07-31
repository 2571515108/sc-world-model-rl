"""Matplotlib training and comparison plots with CSV/PNG/SVG artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metric_aggregator import MetricAggregator
from .run_loader import RunData
from .smoothing import smooth


class ExperimentPlotter:
    """Creates data-backed plots; missing metrics are reported as warnings, never invented."""
    def _save_series(self, name: str, steps: list[int], mean: list[float], lower: list[float], upper: list[float], output_dir: Path, label: str) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True); csv_path = output_dir / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("step", "mean", "lower", "upper")); writer.writeheader()
            writer.writerows({"step": x, "mean": y, "lower": lo, "upper": hi} for x, y, lo, hi in zip(steps, mean, lower, upper))
        figure, axis = plt.subplots(figsize=(8, 4.5)); axis.plot(steps, mean, label=label); axis.fill_between(steps, lower, upper, alpha=0.2)
        axis.set_xlabel("Environment steps"); axis.set_ylabel(name.replace("_", " ")); axis.set_title(name.replace("_", " ").title()); axis.legend(); axis.grid(alpha=0.25); figure.tight_layout()
        paths = [csv_path]
        for extension in ("png", "svg"):
            path = output_dir / f"{name}.{extension}"; figure.savefig(path, dpi=160 if extension == "png" else None); paths.append(path)
        plt.close(figure); return paths

    def plot_metric(self, runs: list[RunData], metric_name: str, output_dir: Path, *, filename: str | None = None,
                    smoothing_method: str = "raw") -> list[Path]:
        """Aggregate and plot a metric across runs with optional display smoothing."""
        series = MetricAggregator().aggregate(runs, metric_name); mean = smooth(series.mean, smoothing_method).tolist()
        return self._save_series(filename or metric_name, series.steps, mean, series.lower, series.upper, output_dir, f"{metric_name} (n={series.run_count})")

    def plot_training_curves(self, runs: list[RunData], output_dir: Path) -> list[Path]:
        """Generate available core training curves and omit unavailable metrics."""
        result: list[Path] = []
        for metric, filename in (("episode_reward", "training_reward_curve"), ("win_rate", "evaluation_win_rate_curve"),
                                 ("episode_length", "episode_length_curve"), ("policy_loss", "policy_value_loss_curve"),
                                 ("entropy", "entropy_curve"), ("learning_rate", "learning_rate_curve"), ("gradient_norm", "gradient_norm_curve")):
            try: result += self.plot_metric(runs, metric, output_dir, filename=filename)
            except ValueError: continue
        return result
