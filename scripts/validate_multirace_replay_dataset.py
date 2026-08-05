"""Fail-fast structural and leakage checks for a multi-race replay dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sc2wmrl.envs.base_macro_env import ACTION_COUNT
from sc2wmrl.replays.multirace import MULTIRACE_SCHEMA, MultiRaceFeatureExtractor


def main() -> None:
    """Validate arrays needed by the shared world model and Terran actor route."""
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="outputs/replays/expert_multirace.npz"); args = parser.parse_args()
    path = Path(args.dataset); metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 3 or metadata.get("schema") != MULTIRACE_SCHEMA:
        raise ValueError("dataset is not a sc2wmrl multi-race v3 replay")
    with np.load(path, allow_pickle=False) as data:
        required = {"observations", "next_observations", "feature_valid_masks", "next_feature_valid_masks", "actions", "universal_intents", "universal_intent_valid", "race_actions", "race_action_valid", "label_confidences", "episode_ids", "game_loops", "replay_ids", "player_ids", "player_races", "opponent_races", "map_names", "paired_view_indices", "terran_macro_actions", "terran_macro_action_valid", "terran_action_masks"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dataset misses required arrays: {sorted(missing)}")
        count, dimension = data["observations"].shape
        if dimension != MultiRaceFeatureExtractor(200, 200).dimension or data["next_observations"].shape != (count, dimension):
            raise ValueError("observation schema shape mismatch")
        if data["feature_valid_masks"].shape != (count, dimension) or data["next_feature_valid_masks"].shape != (count, dimension):
            raise ValueError("feature validity masks do not match observations")
        if not np.isfinite(data["observations"]).all() or not np.isfinite(data["next_observations"]).all():
            raise ValueError("observations contain NaN or Inf")
        if not np.all((0 <= data["actions"]) & (data["actions"] < 14)) or not np.array_equal(data["actions"], data["universal_intents"]):
            raise ValueError("world-model actions must equal 14-class universal intents")
        if data["terran_action_masks"].shape != (count, ACTION_COUNT):
            raise ValueError("Terran action-mask shape mismatch")
        terran_valid = data["terran_macro_action_valid"].astype(bool)
        if terran_valid.any() and not np.all(data["player_races"][terran_valid] == 1):
            raise ValueError("a non-Terran viewpoint was routed to the Terran actor")
        pairs = data["paired_view_indices"]
        for index, peer in enumerate(pairs):
            if peer < 0:
                continue
            if peer >= count or pairs[peer] != index or data["replay_ids"][peer] != data["replay_ids"][index] or data["player_ids"][peer] == data["player_ids"][index]:
                raise ValueError("invalid paired-view index")
        episodes = np.unique(data["episode_ids"])
        if len(episodes) < 2:
            raise ValueError("at least two local-view episodes are required for episode-level validation")
        present_races = sorted(set(int(value) for value in data["player_races"]))
    report = {"dataset": str(path), "transitions": int(count), "episodes": int(len(episodes)), "observation_dim": int(dimension), "player_race_ids": present_races,
              "paired_transitions": int((pairs >= 0).sum()), "valid_terran_actor_labels": int(terran_valid.sum())}
    print(report)


if __name__ == "__main__":
    main()
