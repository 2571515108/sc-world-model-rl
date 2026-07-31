"""Per-decision JSONL, CSV summary, NPZ trajectory, and game summary writer."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DecisionRecord:
    """Replayable macro-decision record emitted by a real-time agent."""
    game_id: str
    game_loop: int
    observation: list[float]
    action_mask: list[int]
    selected_action: int
    selected_action_name: str
    action_probability: float
    top_actions: list[dict[str, Any]]
    predicted_value: float
    policy_entropy: float
    opponent_probabilities: list[float]
    world_model_uncertainty: float | None
    inference_duration_ms: float
    skill_result: str | None
    fallback_used: bool


class DecisionRecorder:
    """Persists every decision without relying on an SC2 replay file."""
    def __init__(self, game_dir: str | Path, game_id: str) -> None:
        self.game_dir = Path(game_dir); self.game_dir.mkdir(parents=True, exist_ok=True); self.game_id = game_id; self.records: list[DecisionRecord] = []
        self._jsonl = (self.game_dir / "decisions.jsonl").open("w", encoding="utf-8")

    def record(self, record: DecisionRecord) -> None:
        """Append one full decision record to memory and JSONL."""
        self.records.append(record); self._jsonl.write(json.dumps(asdict(record), allow_nan=False) + "\n"); self._jsonl.flush()

    def close(self, summary: dict[str, Any]) -> None:
        """Write game summary, macro timeline, episode metrics, and NPZ trajectory."""
        self._jsonl.close(); (self.game_dir / "game_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        with (self.game_dir / "macro_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("game_loop", "selected_action", "selected_action_name", "inference_duration_ms", "fallback_used", "skill_result")); writer.writeheader()
            writer.writerows({key: getattr(record, key) for key in writer.fieldnames} for record in self.records)
        (self.game_dir / "episode_metrics.csv").write_text("decision_count\n" + str(len(self.records)) + "\n", encoding="utf-8")
        if self.records:
            np.savez_compressed(self.game_dir / "trajectory.npz", observations=np.asarray([r.observation for r in self.records], dtype=np.float32),
                                actions=np.asarray([r.selected_action for r in self.records], dtype=np.int64), masks=np.asarray([r.action_mask for r in self.records], dtype=np.int8))
