"""Build HTML, Markdown, and JSON experiment reports from persisted artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .html_template import render_html
from .plot_league import plot_league_analysis
from .plot_policy import plot_policy_analysis
from .plot_training import ExperimentPlotter
from .plot_world_model import plot_world_model_metrics
from .run_loader import RunData


class ReportBuilder:
    """Creates evidence-backed reports and lists unavailable inputs as warnings."""
    def build(self, runs: list[RunData], output_dir: str | Path) -> dict[str, Path]:
        """Generate figures plus HTML, JSON, and Markdown summary artifacts."""
        destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True); figures = destination / "figures"; figures.mkdir(exist_ok=True)
        warnings = [warning for run in runs for warning in run.warnings]
        generated = ExperimentPlotter().plot_training_curves(runs, figures)
        generated += plot_world_model_metrics(runs, figures) + plot_policy_analysis(runs, figures) + plot_league_analysis(runs, figures)
        summary = {"run_count": len(runs), "run_ids": [run.path.name for run in runs], "metric_record_count": sum(len(run.metrics) for run in runs),
                   "generated_figure_count": len([path for path in generated if path.suffix == ".png"])}
        summary_path = destination / "summary.json"; summary_path.write_text(json.dumps({"summary": summary, "warnings": warnings}, indent=2), encoding="utf-8")
        markdown = "# Experiment Summary\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in summary.items()) + "\n\n## Warnings\n\n" + ("\n".join(f"- {warning}" for warning in warnings) or "- None") + "\n"
        markdown_path = destination / "summary.md"; markdown_path.write_text(markdown, encoding="utf-8")
        relative_figures = [str(path.relative_to(destination)).replace("\\", "/") for path in generated]
        html_path = destination / "report.html"; html_path.write_text(render_html("SC2 World Model RL Experiment Report", summary, relative_figures, warnings), encoding="utf-8")
        return {"html": html_path, "summary_json": summary_path, "summary_markdown": markdown_path}
