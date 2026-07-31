"""Generate a data-backed HTML experiment report from one or more run directories."""

from __future__ import annotations

import argparse

from sc2wmrl.reporting.report_builder import ReportBuilder
from sc2wmrl.reporting.run_loader import RunLoader


def main() -> None:
    """Load independent runs and create report artifacts without fabricating missing metrics."""
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", nargs="+", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    runs = [RunLoader().load(path) for path in args.runs]; print({key: str(value) for key, value in ReportBuilder().build(runs, args.output).items()})


if __name__ == "__main__": main()
