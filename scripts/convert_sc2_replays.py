"""Convert selected replay viewpoints into persistent macro transitions."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from sc2wmrl.envs.reward import RewardConfig
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replays.extractor import ReplayExtractionConfig, extract_replay_viewpoint
from sc2wmrl.replays.metadata import inspect_replay_metadata, select_expert_players
from sc2wmrl.utils.config import load_yaml


def _paths(values: list[str]) -> list[Path]:
    """Expand files, directories, and glob expressions deterministically."""
    resolved: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.SC2Replay")))
        elif any(token in value for token in "*?[]"):
            resolved.extend(Path(item) for item in sorted(glob.glob(value, recursive=True)))
        else:
            resolved.append(path)
    unique: list[Path] = []
    for path in resolved:
        absolute = path.resolve()
        if absolute not in unique:
            unique.append(absolute)
    return unique


def main() -> None:
    """Extract human macro transitions while preserving a single-player fog view."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/replays/expert_terran.yaml")
    parser.add_argument("--replays", nargs="*")
    parser.add_argument("--output")
    parser.add_argument("--player-id", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    configured = config.get("replays", [])
    configured = [configured] if isinstance(configured, str) else list(configured)
    replay_paths = _paths(args.replays or configured)
    if not replay_paths:
        raise ValueError("no .SC2Replay paths matched")
    extraction = ReplayExtractionConfig(
        step_mul=int(config.get("step_mul", 32)), minimum_label_confidence=float(config.get("minimum_label_confidence", 0.70)),
        include_no_op_for_behavior_cloning=bool(config.get("include_no_op_for_behavior_cloning", False)),
        download_missing_version=bool(config.get("download_missing_version", True)),
        reward=RewardConfig(**dict(config.get("reward", {}))),
    )
    output = Path(args.output or config["output_path"])
    buffer = ReplayBuffer(int(config.get("capacity", 5_000_000)), seed=int(config.get("seed", 7)))
    summaries = []; failures = []; episode_id = 0
    configured_player = args.player_id if args.player_id is not None else config.get("player_id")
    for replay_path in replay_paths:
        try:
            metadata = inspect_replay_metadata(replay_path)
            players = select_expert_players(
                metadata, target_race=str(config.get("target_race", "Terran")),
                winners_only=bool(config.get("winners_only", True)), min_apm=float(config.get("min_apm", 0.0)),
                player_id=None if configured_player is None else int(configured_player),
            )
            if not players:
                raise ValueError("no player viewpoint passed the expert filters")
            for player in players:
                transitions, summary = extract_replay_viewpoint(replay_path, player, episode_id=episode_id,
                                                                config=extraction, metadata=metadata)
                if not transitions:
                    raise RuntimeError("replay viewpoint produced no macro transitions")
                buffer.extend(transitions); summaries.append(summary.to_dict()); episode_id += 1
                print({"converted": str(replay_path), "player_id": player.player_id, "steps": len(transitions),
                       "behavior_cloning_labels": summary.behavior_cloning_labels})
        except Exception as exc:
            failures.append({"path": str(replay_path), "error": f"{type(exc).__name__}: {exc}"})
            if bool(config.get("fail_fast", True)):
                raise
    if not len(buffer):
        raise RuntimeError("no replay transitions were converted")
    buffer.save(output)
    report = {"format": "expert_sc2_replay_dataset_v2", "output_path": str(output), "episodes": episode_id,
              "transitions": len(buffer), "summaries": summaries, "failures": failures}
    output.with_suffix(output.suffix + ".summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print({"dataset": str(output), "episodes": episode_id, "transitions": len(buffer), "failures": len(failures)})


if __name__ == "__main__":
    main()
