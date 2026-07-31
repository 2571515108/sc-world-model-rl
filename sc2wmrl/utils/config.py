"""Small YAML loader with explicit dependency failures and typed conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a mapping-only YAML document without silently accepting malformed configs."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configuration requires PyYAML; install requirements.txt") from exc
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration {path} must contain a mapping")
    return value
