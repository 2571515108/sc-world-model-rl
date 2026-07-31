"""Validate a run directory and generate all available plots and report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.reporting.report_builder import ReportBuilder
from sc2wmrl.reporting.run_loader import RunLoader


def main() -> None:
    """Run local post-processing for one completed run."""
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", required=True); args = parser.parse_args(); run = RunLoader().load(args.run_dir)
    if not run.metrics: raise ValueError("run contains no metrics; refusing to generate an empty report")
    print({key: str(value) for key, value in ReportBuilder().build([run], Path(args.run_dir) / "report").items()})


if __name__ == "__main__": main()
