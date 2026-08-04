"""Merge replay datasets while preserving episode-boundary integrity."""

from __future__ import annotations

import argparse
from dataclasses import replace

from sc2wmrl.replay.replay_buffer import ReplayBuffer


def main() -> None:
    """Remap episode IDs so arrays from multiple replay files remain disjoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sources = [ReplayBuffer.load(path, seed=args.seed + index) for index, path in enumerate(args.inputs)]
    merged = ReplayBuffer(sum(len(source) for source in sources), seed=args.seed)
    next_episode = 0
    for source in sources:
        mapping: dict[int, int] = {}
        for item in source.transitions():
            if item.episode_id not in mapping:
                mapping[item.episode_id] = next_episode; next_episode += 1
            merged.append(replace(item, episode_id=mapping[item.episode_id]))
    merged.save(args.output)
    print({"output": args.output, "transitions": len(merged), "episodes": next_episode})


if __name__ == "__main__":
    main()
