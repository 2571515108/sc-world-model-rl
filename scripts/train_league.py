"""Initialize and serialize a small scripted/snapshot League state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc2wmrl.league.league import League
from sc2wmrl.utils.config import load_yaml


def main() -> None:
    """Create a reproducible baseline opponent pool; training remains caller-owned."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/league/small_league.yaml"); parser.add_argument("--output", default="outputs/league.json"); parser.add_argument("--checkpoint"); args = parser.parse_args()
    config = load_yaml(args.config); league = League(int(config["seed"])); league.add_scripted_opponents()
    if args.checkpoint:
        snapshot = league.add_snapshot(args.checkpoint, 1000.0, {"source": "train_league"}); league.add_exploiter(args.checkpoint, snapshot.snapshot_id)
    payload = {"opponents": [record.__dict__ for record in league.pool.records()], "payoff_labels": league.payoff_matrix()[0]}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2), encoding="utf-8"); print({"opponents": len(payload["opponents"]), "output": str(path)})


if __name__ == "__main__": main()
