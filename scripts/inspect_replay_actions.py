"""Validate that a collected replay contains useful macro-action coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction


def main() -> None:
    """Report stable action counts and reject demonstrably degenerate replays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--require", action="append", default=[], choices=[action.name for action in MacroAction])
    parser.add_argument("--max-passive-ratio", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.max_passive_ratio <= 1.0:
        raise ValueError("max-passive-ratio must be in [0, 1]")
    # Only the compact action member is needed for this validation.  Avoid
    # reconstructing observations and transition objects for a large replay.
    with np.load(Path(args.replay), allow_pickle=False) as replay:
        actions = np.asarray(replay["actions"], dtype=np.int64)
    counts = Counter(MacroAction(int(action)).name for action in actions)
    total = len(actions)
    if total == 0:
        raise ValueError("replay contains no transitions")
    passive = counts[MacroAction.NO_OP.name] + counts[MacroAction.SCOUT_ENEMY_MAIN.name] + counts[MacroAction.SCOUT_EXPANSION.name]
    missing = [name for name in args.require if counts[name] == 0]
    summary = {action.name: counts[action.name] for action in MacroAction}
    print({"replay": args.replay, "transitions": total, "passive_ratio": passive / total, "action_counts": summary})
    if missing:
        raise RuntimeError(f"replay is missing required actions: {', '.join(missing)}")
    if passive / total > args.max_passive_ratio:
        raise RuntimeError(f"replay passive-action ratio {passive / total:.3f} exceeds {args.max_passive_ratio:.3f}")


if __name__ == "__main__":
    main()
