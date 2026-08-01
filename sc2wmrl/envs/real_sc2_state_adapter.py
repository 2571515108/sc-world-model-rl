"""Translate visible python-sc2 state into the shared macro raw-state schema.

The adapter intentionally uses only information normally exposed to a BotAI:
owned units/buildings, visible enemy units/buildings, and a persistent local
scouting estimate.  It never queries hidden enemy state from the game engine.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Protocol

from .feature_extractor import REGIONS, STRATEGIES


class UnitLike(Protocol):
    """Small structural subset shared by python-sc2 units and test doubles."""

    name: str
    position: Any
    health_percentage: float


_UNIT_ALIASES = {
    "marine": "MARINE", "marauder": "MARAUDER", "reaper": "REAPER", "hellion": "HELLION",
    "tank": "SIEGETANK", "medivac": "MEDIVAC", "viking": "VIKINGFIGHTER", "battlecruiser": "BATTLECRUISER",
}
_BUILDING_ALIASES = {
    "command_center": "COMMANDCENTER", "barracks": "BARRACKS", "factory": "FACTORY", "starport": "STARPORT",
    "refinery": "REFINERY", "engineering_bay": "ENGINEERINGBAY", "tech_lab": "TECHLAB", "reactor": "REACTOR",
}


def _name(item: Any) -> str:
    value = getattr(item, "type_id", getattr(item, "name", item))
    return str(getattr(value, "name", value)).upper()


def _items(value: Any) -> list[Any]:
    """Convert python-sc2 Units or a test iterable into a concrete list."""
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _position(item: Any) -> tuple[float, float] | None:
    point = getattr(item, "position", None)
    if point is None:
        return None
    try:
        return float(point.x), float(point.y)
    except (AttributeError, TypeError, ValueError):
        return None


class RealSC2StateAdapter:
    """Build canonical macro state and maintain non-cheating scouting memory."""

    def __init__(self, map_width: float = 200.0, map_height: float = 200.0) -> None:
        self.map_width, self.map_height = float(map_width), float(map_height)
        self._last_enemy_counts: Counter[str] = Counter()
        self._last_enemy_buildings: Counter[str] = Counter()
        self._last_enemy_position: tuple[float, float] | None = None
        self._last_seen_time: float | None = None

    def reset(self) -> None:
        """Forget observations between games so no enemy information leaks."""
        self._last_enemy_counts.clear(); self._last_enemy_buildings.clear()
        self._last_enemy_position = None; self._last_seen_time = None

    def extract_raw_state(self, bot: Any, *, pending_actions: dict[str, bool] | None = None) -> dict[str, Any]:
        """Extract a state from a live BotAI or a structurally equivalent mock."""
        game_time = float(getattr(bot, "time", 0.0))
        own_units = _items(getattr(bot, "units", None))
        own_structures = _items(getattr(bot, "structures", None))
        workers = _items(getattr(bot, "workers", None))
        enemy_units = _items(getattr(bot, "enemy_units", None))
        enemy_structures = _items(getattr(bot, "enemy_structures", None))
        unit_counts = Counter(_name(unit) for unit in own_units)
        structure_counts = Counter(_name(unit) for unit in own_structures)
        visible_enemy = enemy_units + enemy_structures
        if visible_enemy:
            self._last_enemy_counts = Counter(_name(unit) for unit in enemy_units)
            self._last_enemy_buildings = Counter(_name(unit) for unit in enemy_structures)
            points = [point for unit in enemy_units for point in [_position(unit)] if point is not None]
            if points:
                self._last_enemy_position = (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))
            self._last_seen_time = game_time
        state = {
            "game_time": game_time,
            "minerals": float(getattr(bot, "minerals", 0.0)),
            "vespene": float(getattr(bot, "vespene", 0.0)),
            "current_supply": float(getattr(bot, "supply_used", 0.0)),
            "maximum_supply": float(getattr(bot, "supply_cap", 0.0)),
            "worker_count": len(workers),
            "base_count": self._count(structure_counts, "COMMANDCENTER", "ORBITALCOMMAND", "PLANETARYFORTRESS"),
            "idle_worker_count": sum(bool(getattr(worker, "is_idle", False)) for worker in workers),
            "army_value": self._army_value(bot, own_units),
            "lost_army_value": float(getattr(getattr(bot, "state", None), "score", getattr(bot, "lost_army_value", 0.0)).lost_minerals_army if hasattr(getattr(getattr(bot, "state", None), "score", None), "lost_minerals_army") else getattr(bot, "lost_army_value", 0.0)),
            "buildings": {key: self._count(structure_counts, alias) for key, alias in _BUILDING_ALIASES.items()},
            "completed_upgrade_flags": self._upgrades(bot),
            "production_queue_summary": self._queues(own_structures),
            "units": {key: self._count(unit_counts, alias) for key, alias in _UNIT_ALIASES.items()},
            "average_army_health": self._average_health(own_units),
            "army_center_x": self._army_center(own_units)[0],
            "army_center_y": self._army_center(own_units)[1],
            "number_of_army_groups": int(bool(own_units)),
            "enemy_army_value_estimate": self._estimated_enemy_value(bot),
            "enemy": self._enemy_state(game_time),
            "map_control": self._map_control(own_units, enemy_units, game_time),
            "pending_actions": dict(pending_actions or {}),
            "available": self._availability(bot, workers, own_structures, own_units),
        }
        return state

    @staticmethod
    def _count(counts: Counter[str], *names: str) -> int:
        return sum(counts.get(name, 0) for name in names)

    @staticmethod
    def _average_health(units: Iterable[Any]) -> float:
        values = [float(getattr(unit, "health_percentage", 1.0)) for unit in units]
        return sum(values) / len(values) if values else 1.0

    @staticmethod
    def _army_center(units: Iterable[Any]) -> tuple[float, float]:
        points = [point for unit in units for point in [_position(unit)] if point is not None]
        return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)) if points else (0.0, 0.0)

    @staticmethod
    def _upgrades(bot: Any) -> list[int]:
        upgrades = list(getattr(getattr(bot, "state", None), "upgrades", set()))
        return [int(index < len(upgrades)) for index in range(6)]

    @staticmethod
    def _queues(structures: Iterable[Any]) -> list[int]:
        names = ("BARRACKS", "FACTORY", "STARPORT", "REFINERY", "COMMANDCENTER")
        counts = Counter(_name(item) for item in structures)
        return [counts.get(name, 0) for name in names]

    def _army_value(self, bot: Any, units: Iterable[Any]) -> float:
        total = 0.0
        for unit in units:
            try:
                value = bot.calculate_unit_value(unit.type_id)
                total += float(value.minerals + value.vespene)
            except (AttributeError, TypeError, ValueError):
                total += 50.0
        return total

    def _estimated_enemy_value(self, bot: Any) -> float:
        total = 0.0
        for name, count in self._last_enemy_counts.items():
            try:
                unit_type = next(unit.type_id for unit in _items(getattr(bot, "enemy_units", None)) if _name(unit) == name)
                value = bot.calculate_unit_value(unit_type); total += count * float(value.minerals + value.vespene)
            except (AttributeError, StopIteration, TypeError, ValueError):
                total += count * 50.0
        return total

    def _enemy_state(self, game_time: float) -> dict[str, Any]:
        reverse_units = {alias: key for key, alias in _UNIT_ALIASES.items()}
        observed = {reverse_units[name]: count for name, count in self._last_enemy_counts.items() if name in reverse_units}
        building_groups = {"base": ("COMMANDCENTER", "NEXUS", "HATCHERY"), "production": ("BARRACKS", "GATEWAY", "SPAWNINGPOOL"), "ground_tech": ("FACTORY", "ROBOTICSFACILITY"), "air_tech": ("STARPORT", "STARGATE", "SPIRE")}
        return {
            "observed_unit_counts": observed,
            "observed_buildings": {key: self._count(self._last_enemy_buildings, *names) for key, names in building_groups.items()},
            "last_seen_army_position": self._last_enemy_position,
            "time_since_last_scout": 3600.0 if self._last_seen_time is None else max(0.0, game_time - self._last_seen_time),
            "estimated_army_value": self._estimated_enemy_value_from_counts(),
            "estimated_worker_count": float(self._last_enemy_counts.get("SCV", 0) + self._last_enemy_counts.get("PROBE", 0) + self._last_enemy_counts.get("DRONE", 0)),
            "strategy_probabilities": {name: 1.0 if name == "unknown" else 0.0 for name in STRATEGIES},
        }

    def _estimated_enemy_value_from_counts(self) -> float:
        return float(sum(self._last_enemy_counts.values()) * 50)

    def _map_control(self, own_units: list[Any], enemy_units: list[Any], game_time: float) -> dict[str, dict[str, float]]:
        own_power, enemy_power = float(len(own_units) * 50), float(len(enemy_units) * 50)
        result: dict[str, dict[str, float]] = {}
        for region in REGIONS:
            visible = 1.0 if region.startswith("own") or (region.startswith("enemy") and bool(enemy_units)) else 0.0
            result[region] = {"friendly_power": own_power if region.startswith("own") else 0.0, "visible_enemy_power": enemy_power if region.startswith("enemy") else 0.0, "visibility": visible, "control_score": max(-1.0, min(1.0, (own_power - enemy_power) / 1000.0)), "last_scout_time": 0.0 if visible else game_time}
        return result

    @staticmethod
    def _availability(bot: Any, workers: list[Any], structures: list[Any], units: list[Any]) -> dict[str, bool]:
        names = Counter(_name(item) for item in structures)
        idle = lambda name: any(_name(item) == name and bool(getattr(item, "is_idle", True)) for item in structures)
        starts = _items(getattr(bot, "enemy_start_locations", None))
        return {"worker": bool(workers), "command_center": idle("COMMANDCENTER") or idle("ORBITALCOMMAND"), "barracks": idle("BARRACKS"), "factory": idle("FACTORY"), "starport": idle("STARPORT"), "scout": bool(workers or units), "defend_target": bool(units), "natural_target": names.get("COMMANDCENTER", 0) > 1, "enemy_target": bool(starts), "enemy_natural_target": bool(starts), "enemy_main_target": bool(starts)}
