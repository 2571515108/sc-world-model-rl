"""Print recorded real-time decisions from a completed game JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Replay decisions as readable records without claiming to replay SC2 simulation."""
    parser = argparse.ArgumentParser(); parser.add_argument("--decisions", required=True); args = parser.parse_args()
    for line in Path(args.decisions).read_text(encoding="utf-8").splitlines(): print(json.loads(line))


if __name__ == "__main__": main()
