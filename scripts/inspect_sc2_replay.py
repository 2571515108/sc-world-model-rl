"""Inspect replay metadata and select candidate expert viewpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc2wmrl.replays.metadata import inspect_replay_metadata, select_expert_players


def main() -> None:
    """Print engine-free metadata for one or more ``.SC2Replay`` files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+")
    parser.add_argument("--race", default="Terran")
    parser.add_argument("--min-apm", type=float, default=0.0)
    parser.add_argument("--include-losses", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    reports = []
    for replay in args.replays:
        metadata = inspect_replay_metadata(replay)
        selected = select_expert_players(metadata, target_race=args.race, winners_only=not args.include_losses,
                                         min_apm=args.min_apm)
        report = metadata.to_dict(); report["selected_player_ids"] = [player.player_id for player in selected]
        reports.append(report); print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        output = Path(args.json_output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
