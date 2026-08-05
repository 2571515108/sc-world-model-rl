"""Audit converted expert datasets without reconstructing all observations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction


def main() -> None:
    """Report action coverage, BC coverage, and unknown-opponent safeguards."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-passive-ratio", type=float, default=0.85)
    args = parser.parse_args()
    replay_path = Path(args.replay)
    metadata = json.loads(replay_path.with_suffix(replay_path.suffix + ".json").read_text(encoding="utf-8"))
    with np.load(replay_path, allow_pickle=False) as arrays:
        actions = np.asarray(arrays["actions"], dtype=np.int64)
        valid = np.asarray(arrays["opponent_action_valid"], dtype=np.bool_) if "opponent_action_valid" in arrays.files else np.asarray(
            [value != "real_sc2" for value in metadata.get("environment_types", ["synthetic"] * len(actions))], dtype=np.bool_
        )
    if not len(actions):
        raise ValueError("expert replay contains no transitions")
    infos = metadata["infos"]
    histogram = Counter(MacroAction(int(action)).name for action in actions)
    labels = [info for info in infos if bool(info.get("expert_label", False))]
    confidences = [float(info.get("label_confidence", 0.0)) for info in labels]
    passive = histogram[MacroAction.NO_OP.name] + histogram[MacroAction.SCOUT_ENEMY_MAIN.name] + histogram[MacroAction.SCOUT_EXPANSION.name]
    report = {
        "transitions": len(actions), "episodes": len(set(metadata["episode_ids"])),
        "action_histogram": {action.name: histogram[action.name] for action in MacroAction},
        "passive_ratio": passive / len(actions), "behavior_cloning_labels": len(labels),
        "behavior_cloning_fraction": len(labels) / len(actions),
        "mean_label_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "augmented_action_masks": sum(bool(info.get("mask_augmented_by_expert_command", False)) for info in infos),
        "unknown_opponent_action_fraction": float((~valid).mean()),
        "source_replays": sorted({str(info.get("source_replay", "unknown")) for info in infos}),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False); print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    required = {
        "BUILD_BARRACKS": (MacroAction.BUILD_BARRACKS,),
        "TRAIN_BASIC_ARMY": (MacroAction.TRAIN_BASIC_ARMY,),
        # Replay target points often describe a flank or nearby unit rather
        # than the fixed enemy-main coordinate. Either combat label proves the
        # expert dataset contains offensive behavior at this action granularity.
        "OFFENSIVE_COMBAT": (MacroAction.ATTACK_ENEMY_MAIN, MacroAction.HARASS),
    }
    missing = [name for name, alternatives in required.items() if not any(histogram[action.name] for action in alternatives)]
    if missing:
        raise RuntimeError(f"expert dataset is missing required macro coverage: {', '.join(missing)}")
    if report["passive_ratio"] > args.max_passive_ratio:
        raise RuntimeError("expert dataset is dominated by NO_OP/scouting labels")
    if not labels:
        raise RuntimeError("expert dataset has no behavior-cloning labels")


if __name__ == "__main__":
    main()
