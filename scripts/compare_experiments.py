"""Aggregate multiple run directories into an aligned comparison report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sc2wmrl.reporting.report_builder import ReportBuilder
from sc2wmrl.reporting.run_loader import RunLoader


def main() -> None:
    """Build a comparison report and explicit final metrics table for valid runs."""
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", nargs="+", required=True); parser.add_argument("--group-by", default="algorithm"); parser.add_argument("--output", required=True); args = parser.parse_args()
    runs = [RunLoader().load(path) for path in args.runs]; output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with (output / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "metric_name", "value", "global_step")); writer.writeheader()
        for run in runs:
            latest: dict[str, dict] = {}
            for row in run.metrics:
                if row["metric_name"] not in latest or row["global_step"] >= latest[row["metric_name"]]["global_step"]: latest[row["metric_name"]] = row
            writer.writerows({"run": run.path.name, "metric_name": name, "value": row["value"], "global_step": row["global_step"]} for name, row in latest.items())
    print({key: str(value) for key, value in ReportBuilder().build(runs, output).items()})


if __name__ == "__main__": main()
