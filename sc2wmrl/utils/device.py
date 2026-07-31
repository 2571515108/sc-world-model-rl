"""Device selection with CUDA-first defaults and explicit fallback behavior."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = "auto") -> torch.device:
    """Select CUDA when available unless a concrete device was requested."""
    choice = (requested or "auto").lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU")
    return device
