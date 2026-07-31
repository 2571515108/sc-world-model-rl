"""Fixed, normalized structured state representation for macro policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


REGIONS = (
    "own_main", "own_natural", "map_center", "enemy_natural", "enemy_main",
    "left_route", "right_route", "expansion_regions",
)
UNIT_TYPES = ("marine", "marauder", "reaper", "hellion", "tank", "medivac", "viking", "battlecruiser")
BUILDINGS = ("command_center", "barracks", "factory", "starport", "refinery", "engineering_bay", "tech_lab", "reactor")
STRATEGIES = ("rush", "economy", "defensive", "ground_tech", "air_tech", "unknown")


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable schema allowing replay/checkpoint compatibility checks."""

    names: tuple[str, ...]

    @property
    def dimension(self) -> int:
        """Number of scalar network inputs."""
        return len(self.names)


class FeatureExtractor:
    """Converts raw dictionary state into finite, range-bounded float32 features.

    Unknown enemy positions have an explicit present bit, preventing a coordinate
    value of zero from being confused with an unobserved enemy.
    """

    def __init__(self, map_width: float = 200.0, map_height: float = 200.0) -> None:
        if map_width <= 0 or map_height <= 0:
            raise ValueError("map dimensions must be positive")
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.spec = FeatureSpec(self._feature_names())

    @staticmethod
    def _feature_names() -> tuple[str, ...]:
        names = [
            "game_time", "minerals", "vespene", "current_supply", "maximum_supply",
            "worker_count", "base_count", "idle_worker_count", "army_value",
            "lost_army_value", "enemy_army_value_estimate",
        ]
        names += [f"building_{name}" for name in BUILDINGS]
        names += [f"upgrade_{index}" for index in range(6)]
        names += [f"queue_{index}" for index in range(5)]
        names += [f"unit_{name}" for name in UNIT_TYPES]
        names += ["average_army_health", "army_center_x", "army_center_y", "number_of_army_groups"]
        names += [f"enemy_observed_{name}" for name in UNIT_TYPES]
        names += [f"enemy_building_{name}" for name in ("base", "production", "ground_tech", "air_tech")]
        names += ["enemy_position_x", "enemy_position_y", "enemy_position_present", "time_since_last_scout",
                  "estimated_enemy_army_value", "estimated_enemy_worker_count"]
        names += [f"enemy_strategy_{name}" for name in STRATEGIES]
        for region in REGIONS:
            names += [f"{region}_{key}" for key in ("friendly_power", "visible_enemy_power", "visibility", "control_score", "last_scout_time")]
        return tuple(names)

    @property
    def dimension(self) -> int:
        """Fixed observation dimension."""
        return self.spec.dimension

    @staticmethod
    def _log(value: Any, scale: float) -> float:
        return float(np.clip(np.log1p(max(0.0, float(value))) / np.log1p(scale), 0.0, 1.0))

    @staticmethod
    def _ratio(value: Any, scale: float) -> float:
        return float(np.clip(float(value) / scale, 0.0, 1.0))

    def extract(self, state: Mapping[str, Any]) -> np.ndarray:
        """Return a fixed-shape finite array; absent values receive safe encodings."""
        values: list[float] = []
        values += [
            self._log(state.get("game_time", 0), 3600), self._log(state.get("minerals", 0), 4000),
            self._log(state.get("vespene", 0), 3000), self._ratio(state.get("current_supply", 0), 200),
            self._ratio(state.get("maximum_supply", 0), 200), self._ratio(state.get("worker_count", 0), 80),
            self._ratio(state.get("base_count", 0), 8), self._ratio(state.get("idle_worker_count", 0), 80),
            self._log(state.get("army_value", 0), 12000), self._log(state.get("lost_army_value", 0), 12000),
            self._log(state.get("enemy_army_value_estimate", 0), 12000),
        ]
        buildings = state.get("buildings", {})
        values += [self._ratio(buildings.get(name, 0), 20) for name in BUILDINGS]
        upgrades = state.get("completed_upgrade_flags", [0] * 6)
        values += [float(bool(upgrades[i])) if i < len(upgrades) else 0.0 for i in range(6)]
        queues = state.get("production_queue_summary", [0] * 5)
        values += [self._ratio(queues[i], 20) if i < len(queues) else 0.0 for i in range(5)]
        units = state.get("units", {})
        values += [self._ratio(units.get(name, 0), 200) for name in UNIT_TYPES]
        values += [self._ratio(state.get("average_army_health", 0), 1), self._ratio(state.get("army_center_x", 0), self.map_width),
                   self._ratio(state.get("army_center_y", 0), self.map_height), self._ratio(state.get("number_of_army_groups", 0), 10)]
        enemy = state.get("enemy", {})
        observed = enemy.get("observed_unit_counts", {})
        values += [self._ratio(observed.get(name, 0), 200) for name in UNIT_TYPES]
        enemy_buildings = enemy.get("observed_buildings", {})
        values += [self._ratio(enemy_buildings.get(name, 0), 20) for name in ("base", "production", "ground_tech", "air_tech")]
        position = enemy.get("last_seen_army_position")
        if position is None:
            values += [0.0, 0.0, 0.0]
        else:
            values += [self._ratio(position[0], self.map_width), self._ratio(position[1], self.map_height), 1.0]
        values += [self._ratio(enemy.get("time_since_last_scout", 3600), 3600), self._log(enemy.get("estimated_army_value", 0), 12000), self._ratio(enemy.get("estimated_worker_count", 0), 80)]
        probabilities = enemy.get("strategy_probabilities", {})
        values += [self._ratio(probabilities.get(name, 0), 1) for name in STRATEGIES]
        map_control = state.get("map_control", {})
        for region in REGIONS:
            entry = map_control.get(region, {})
            values += [self._log(entry.get("friendly_power", 0), 12000), self._log(entry.get("visible_enemy_power", 0), 12000),
                       self._ratio(entry.get("visibility", 0), 1), float(np.clip((float(entry.get("control_score", 0)) + 1) / 2, 0, 1)),
                       self._ratio(entry.get("last_scout_time", 3600), 3600)]
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.dimension,) or not np.isfinite(observation).all():
            raise ValueError("feature extraction produced invalid observation")
        return observation
