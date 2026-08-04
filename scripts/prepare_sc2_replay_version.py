"""Request the current SC2 client to prepare data for an exact replay build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc2wmrl.replays.extractor import _require_sc2
from sc2wmrl.replays.metadata import inspect_replay_metadata
from sc2wmrl.replays.versioning import prepare_replay_version


def main() -> None:
    """Run the official replay-info download request and write its audit report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay = Path(args.replay).expanduser().resolve()
    metadata = inspect_replay_metadata(replay)
    run_configs, _, sc_pb = _require_sc2()
    report = prepare_replay_version(replay, metadata, run_configs=run_configs, sc_pb=sc_pb)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    if report.replay_info_error or not report.executable_present_after:
        raise RuntimeError(
            f"Base{report.base_build} is not ready for replay conversion; see {output} for the exact client response."
        )


if __name__ == "__main__":
    main()
