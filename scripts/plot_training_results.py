"""Create training figures for one or more completed run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.reporting.plot_training import ExperimentPlotter
from sc2wmrl.reporting.run_loader import RunLoader


def main() -> None:
    """Generate available training curves and print their paths."""
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", nargs="+", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    paths = ExperimentPlotter().plot_training_curves([RunLoader().load(path) for path in args.runs], Path(args.output)); print([str(path) for path in paths])


if __name__ == "__main__": main()
