"""Unified TensorBoard, CSV, and JSONL experiment metric logging."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricRecord:
    """A single numeric observation emitted by a training or evaluation phase."""

    run_id: str
    phase: str
    global_step: int
    episode: int
    metric_name: str
    value: float
    timestamp: float
    metadata: dict[str, Any]


def _git_commit(project_dir: Path) -> str | None:
    """Return the current commit when Git metadata is available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_dir, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run_directory(base_dir: str | Path, algorithm: str, seed: int) -> Path:
    """Create the documented independent run directory and required subfolders."""
    run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{algorithm}_seed{seed}"
    run_dir = Path(base_dir) / run_id
    for relative in ("checkpoints", "evaluation", "figures", "replays", "report"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir


class ExperimentLogger:
    """Writes lossless scalar records to CSV/JSONL and optional TensorBoard."""

    COLUMNS = ("run_id", "phase", "global_step", "episode", "metric_name", "value", "timestamp", "metadata")

    def __init__(self, run_dir: str | Path, *, run_id: str | None = None, config: dict[str, Any] | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name; self._records: list[MetricRecord] = []
        self._csv_file = (self.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.COLUMNS)
        if self._csv_file.tell() == 0: self._csv_writer.writeheader()
        self._jsonl_file = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.run_dir))
        except (ImportError, OSError):
            self._writer = None
        if config is not None: (self.run_dir / "config.yaml").write_text(self._yaml(config), encoding="utf-8")
        merged = self.runtime_metadata() | (metadata or {})
        (self.run_dir / "metadata.json").write_text(json.dumps(merged, indent=2, sort_keys=True, default=str), encoding="utf-8")

    @staticmethod
    def _yaml(value: dict[str, Any]) -> str:
        """Serialize configuration with PyYAML when available, then JSON fallback."""
        try:
            import yaml
            return yaml.safe_dump(value, sort_keys=True)
        except ImportError:
            return json.dumps(value, indent=2, sort_keys=True)

    @staticmethod
    def runtime_metadata() -> dict[str, Any]:
        """Capture runtime information without claiming unavailable SC2 details."""
        try:
            import torch
            torch_data = {"pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
                          "device": "cuda" if torch.cuda.is_available() else "cpu"}
        except ImportError:
            torch_data = {"pytorch_version": None, "cuda_version": None, "device": "unknown"}
        return {"python_version": sys.version, "platform": platform.platform(), "start_time": time.time(), **torch_data}

    def log_scalar(self, name: str, value: float, *, step: int, episode: int, phase: str,
                   metadata: dict[str, Any] | None = None) -> None:
        """Record one finite scalar in all locally available log formats."""
        import math
        if not name or step < 0 or episode < 0 or not math.isfinite(float(value)):
            raise ValueError("metric name, step, episode, and value must be valid")
        record = MetricRecord(self.run_id, phase, step, episode, name, float(value), time.time(), metadata or {})
        self._records.append(record); row = asdict(record); row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
        self._csv_writer.writerow(row); self._jsonl_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        if self._writer is not None: self._writer.add_scalar(f"{phase}/{name}", value, step)

    def flush(self) -> None:
        """Flush every sink so interrupted jobs retain completed metric records."""
        self._csv_file.flush(); self._jsonl_file.flush()
        if self._writer is not None: self._writer.flush()

    def close(self) -> None:
        """Flush and close file resources exactly once."""
        self.flush(); self._csv_file.close(); self._jsonl_file.close()
        if self._writer is not None: self._writer.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
