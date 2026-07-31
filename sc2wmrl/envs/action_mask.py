"""Single source of truth for macro-action legality."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base_macro_env import ACTION_COUNT, MacroAction


def action_mask(state: Mapping[str, Any]) -> np.ndarray:
    """Compute legal actions from the current structured state.

    This deliberately enforces only prerequisites observable at the macro layer.
    The skill executor may still report a transient execution failure; those are
    recorded in ``info`` but never turn an illegal policy action into success.
    """
    minerals, vespene = float(state.get("minerals", 0)), float(state.get("vespene", 0))
    supply, supply_cap = int(state.get("current_supply", 0)), int(state.get("maximum_supply", 0))
    workers, bases = int(state.get("worker_count", 0)), int(state.get("base_count", 0))
    army = float(state.get("army_value", 0))
    b = state.get("buildings", {})
    barracks, factory, starport = int(b.get("barracks", 0)), int(b.get("factory", 0)), int(b.get("starport", 0))
    refiners, upgrades = int(b.get("refinery", 0)), state.get("completed_upgrade_flags", [])
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    mask[MacroAction.NO_OP] = True
    mask[MacroAction.TRAIN_WORKERS] = minerals >= 50 and workers < min(80, supply_cap) and bases > 0
    mask[MacroAction.BUILD_SUPPLY] = minerals >= 100 and supply_cap < 200
    mask[MacroAction.BUILD_BARRACKS] = minerals >= 150
    mask[MacroAction.BUILD_REFINERY] = minerals >= 75 and refiners < bases * 2
    mask[MacroAction.BUILD_FACTORY] = minerals >= 150 and vespene >= 100 and barracks > 0
    mask[MacroAction.BUILD_STARPORT] = minerals >= 150 and vespene >= 100 and factory > 0
    mask[MacroAction.EXPAND] = minerals >= 400 and bases < 8
    mask[MacroAction.TRAIN_BASIC_ARMY] = minerals >= 50 and supply < supply_cap and barracks > 0
    mask[MacroAction.TRAIN_ANTI_GROUND] = minerals >= 100 and vespene >= 25 and supply < supply_cap and barracks > 0
    mask[MacroAction.TRAIN_ANTI_AIR] = minerals >= 100 and vespene >= 100 and supply < supply_cap and starport > 0
    mask[MacroAction.RESEARCH_UPGRADE] = minerals >= 100 and vespene >= 100 and barracks > 0 and not all(upgrades)
    mask[MacroAction.SCOUT_ENEMY_MAIN] = workers > 0 or army > 0
    mask[MacroAction.SCOUT_EXPANSION] = workers > 0 or army > 0
    mask[MacroAction.DEFEND_MAIN] = army > 0
    mask[MacroAction.DEFEND_NATURAL] = army > 0 and bases > 1
    mask[MacroAction.HARASS] = army >= 100
    mask[MacroAction.ATTACK_ENEMY_NATURAL] = army >= 150
    mask[MacroAction.ATTACK_ENEMY_MAIN] = army >= 250
    mask[MacroAction.RETREAT] = army > 0
    if not mask.any():  # Defensive invariant, primarily protects future edits.
        mask[MacroAction.NO_OP] = True
    return mask


def masked_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return logits with illegal choices set to ``-inf`` after strict checks."""
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    if logits.shape != (ACTION_COUNT,) or mask.shape != (ACTION_COUNT,):
        raise ValueError("logits and action mask must both have ACTION_COUNT elements")
    if not np.isfinite(logits).all() or not mask.any():
        raise ValueError("invalid logits or empty action mask")
    return np.where(mask, logits, -np.inf)
