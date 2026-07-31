"""Uncertainty calibration metrics for ensemble world-model predictions."""

from __future__ import annotations

import numpy as np


def uncertainty_error_correlation(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """Return Pearson correlation, safely zero for degenerate finite inputs."""
    uncertainty, errors = np.asarray(uncertainty, dtype=float).reshape(-1), np.asarray(errors, dtype=float).reshape(-1)
    if uncertainty.shape != errors.shape or not np.isfinite(uncertainty).all() or not np.isfinite(errors).all(): raise ValueError("invalid calibration inputs")
    if len(uncertainty) < 2 or uncertainty.std() == 0 or errors.std() == 0: return 0.0
    return float(np.corrcoef(uncertainty, errors)[0, 1])
