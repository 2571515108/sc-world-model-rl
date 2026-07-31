"""Best-effort SC2 replay persistence with explicit failure records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class ReplaySource(Protocol):
    """A real SC2 integration that can attempt saving its current replay."""
    def save_replay(self, path: Path) -> None:
        """Save replay to the requested destination or raise an informative error."""


class ReplayManager:
    """Never reports a replay as saved unless the backend actually created it."""
    def save(self, source: ReplaySource | None, game_dir: str | Path) -> tuple[Path | None, str | None]:
        """Attempt replay persistence and record failure reason when unavailable."""
        directory = Path(game_dir); directory.mkdir(parents=True, exist_ok=True); path = directory / "replay.SC2Replay"
        if source is None:
            reason = "SC2 backend does not expose replay saving"
        else:
            try:
                source.save_replay(path)
                if path.exists(): return path, None
                reason = "backend returned without creating replay file"
            except Exception as exc:
                reason = f"replay save failed: {type(exc).__name__}: {exc}"
        (directory / "replay_save_failure.json").write_text(json.dumps({"reason": reason}), encoding="utf-8"); return None, reason
