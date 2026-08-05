"""Convert each P/T/Z replay participant into an isolated expert episode."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from sc2wmrl.envs.reward import RewardConfig
from sc2wmrl.replays.extractor import ReplayExtractionConfig
from sc2wmrl.replays.metadata import inspect_replay_metadata
from sc2wmrl.replays.multirace import extract_multirace_viewpoint, save_multirace_dataset
from sc2wmrl.utils.config import load_yaml


def _paths(values: list[str]) -> list[Path]:
    """Expand replay paths deterministically and de-duplicate resolved files."""
    result: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            result.extend(sorted(path.rglob("*.SC2Replay")))
        elif any(token in value for token in "*?[]"):
            result.extend(Path(item) for item in sorted(glob.glob(value, recursive=True)))
        else:
            result.append(path)
    return list(dict.fromkeys(path.resolve() for path in result))


def main() -> None:
    """Build a joint P/T/Z dataset while retaining both local viewpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/replays/multirace_expert.yaml")
    parser.add_argument("--replays", nargs="*")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_yaml(args.config); configured = config.get("replays", [])
    paths = _paths(args.replays or ([configured] if isinstance(configured, str) else list(configured)))
    if not paths:
        raise ValueError("no .SC2Replay paths matched")
    extraction = ReplayExtractionConfig(step_mul=int(config.get("step_mul", 32)), minimum_label_confidence=float(config.get("minimum_label_confidence", 0.70)),
        include_no_op_for_behavior_cloning=False, download_missing_version=bool(config.get("download_missing_version", True)),
        map_file=None if config.get("map_file") is None else str(config["map_file"]), sc2_path=None if config.get("sc2_path") is None else str(config["sc2_path"]), reward=RewardConfig(**dict(config.get("reward", {}))))
    races = {str(value).strip().title() for value in config.get("races", ["Terran", "Protoss", "Zerg"])}
    minimum_apm = float(config.get("min_apm", 0.0)); transitions = []; summaries = []; failures = []; episode_id = 0
    for replay_id, replay_path in enumerate(paths):
        try:
            metadata = inspect_replay_metadata(replay_path)
            players = [player for player in metadata.players if player.assigned_race in races and player.apm >= minimum_apm]
            if len(players) != 2:
                raise ValueError(f"expected two P/T/Z players, found {len(players)}")
            for player in players:
                items, summary = extract_multirace_viewpoint(replay_path, player, episode_id=episode_id, replay_id=replay_id, config=extraction, metadata=metadata)
                if not items:
                    raise RuntimeError("replay viewpoint produced no transitions")
                transitions.extend(items); summaries.append(summary); episode_id += 1
                print({"replay": replay_path.name, "player_id": player.player_id, "race": player.assigned_race, "steps": len(items), "valid_labels": summary.valid_universal_labels})
        except Exception as exc:
            failures.append({"path": str(replay_path), "error": f"{type(exc).__name__}: {exc}"})
            if bool(config.get("fail_fast", False)):
                raise
    if not transitions:
        raise RuntimeError("no multi-race replay transitions were converted")
    output = Path(args.output or config["output_path"]); save_multirace_dataset(output, transitions, summaries)
    output.with_suffix(output.suffix + ".conversion.json").write_text(json.dumps({"episodes": episode_id, "transitions": len(transitions), "summaries": [item.to_dict() for item in summaries], "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"dataset": str(output), "episodes": episode_id, "transitions": len(transitions), "failures": len(failures)})


if __name__ == "__main__":
    main()
