"""Helpers for adapting replay-controller protobuf observations."""

from __future__ import annotations

from typing import Any

import numpy as np


def _raw_unit_dict(unit: Any) -> dict[str, Any]:
    """Normalize controller protobuf units to the live adapter's raw schema."""
    position = getattr(unit, "pos", None)
    return {
        "alliance": int(unit.alliance), "unit_type": int(unit.unit_type), "tag": int(unit.tag),
        "x": float(getattr(position, "x", 0.0)), "y": float(getattr(position, "y", 0.0)),
        "health": float(unit.health), "health_max": float(unit.health_max),
        # PySC2 feature observations scale completion to 0--100; maintain the
        # same canonical value for the shared state adapter.
        "build_progress": float(unit.build_progress) * 100.0,
        "order_length": len(unit.orders), "add_on_tag": int(unit.add_on_tag),
    }


def controller_observation_dict(response: Any) -> dict[str, Any]:
    """Convert a controller observation to the state adapter's input schema."""
    observation = response.observation
    common = observation.player_common
    return {
        "raw_units": [_raw_unit_dict(unit) for unit in observation.raw_data.units],
        "player": np.asarray([
            common.player_id, common.minerals, common.vespene, common.food_used, common.food_cap,
            common.food_army, common.food_workers, common.idle_worker_count, common.army_count,
            common.warp_gate_count, common.larva_count,
        ], dtype=np.float32),
        "game_loop": np.asarray([observation.game_loop], dtype=np.int64),
    }


def player_outcome(response: Any, player_id: int) -> str | None:
    """Return a terminal result from a replay-controller response, when present."""
    for result in response.player_result:
        if int(result.player_id) == int(player_id):
            return {1: "win", 2: "loss", 3: "draw"}.get(int(result.result))
    return None


def raw_unit_index(response: Any) -> dict[int, Any]:
    """Index visible units by their stable raw-unit tag."""
    return {int(unit.tag): unit for unit in response.observation.raw_data.units}
