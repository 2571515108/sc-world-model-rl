"""Structured loading of persistent experiment metrics and metadata."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunData:
    """A run directory with parsed metric rows and explicit load warnings."""
    path: Path
    metadata: dict[str, Any]
    config: dict[str, Any]
    metrics: list[dict[str, Any]]
    warnings: tuple[str, ...] = ()


class RunLoader:
    """Loads valid runs independently so one damaged run does not block a comparison."""
    def load(self, path: str | Path) -> RunData:
        """Load CSV metrics and JSON metadata, retaining missing-data warnings."""
        run_dir = Path(path); warnings: list[str] = []
        if not run_dir.is_dir(): raise FileNotFoundError(run_dir)
        metadata_path, config_path, metrics_path = run_dir / "metadata.json", run_dir / "config.yaml", run_dir / "metrics.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not metadata_path.exists(): warnings.append("missing metadata.json")
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {} if config_path.exists() else {}
        except ImportError:
            config = {}
        if not config_path.exists(): warnings.append("missing config.yaml")
        metrics: list[dict[str, Any]] = []
        if metrics_path.exists():
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["global_step"], row["episode"], row["value"] = int(row["global_step"]), int(row["episode"]), float(row["value"])
                    row["metadata"] = json.loads(row.get("metadata") or "{}")
                    metrics.append(row)
        else: warnings.append("missing metrics.csv")
        return RunData(run_dir, metadata, config, metrics, tuple(warnings))
