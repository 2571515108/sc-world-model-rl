"""PySC2 implementation of the synchronous real-SC2 backend contract.

The backend deliberately uses PySC2's raw-unit and raw-action interface.  It
does not depend on ``BotAI`` or run an asyncio loop, so every macro step maps
directly to one ``SC2Env.step`` call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .action_mask import action_mask
from .base_macro_env import MacroAction


# StarCraft II unit type IDs used by PySC2's raw-unit observations.  These are
# stable game-data identifiers, not positions in PySC2's action table.
_UNIT_IDS = {"COMMANDCENTER": 18, "SUPPLYDEPOT": 19, "REFINERY": 20,
             "BARRACKS": 21, "ENGINEERINGBAY": 22, "FACTORY": 27,
             "STARPORT": 28, "SIEGETANK": 33, "VIKINGFIGHTER": 35,
             "SCV": 45, "MARINE": 48, "REAPER": 49, "MARAUDER": 51,
             "HELLION": 53, "MEDIVAC": 54, "BATTLECRUISER": 57,
             "PLANETARYFORTRESS": 130, "ORBITALCOMMAND": 132}
_BUILDING_IDS = {_UNIT_IDS[name] for name in ("COMMANDCENTER", "SUPPLYDEPOT", "REFINERY", "BARRACKS", "ENGINEERINGBAY", "FACTORY", "STARPORT", "PLANETARYFORTRESS", "ORBITALCOMMAND")}
_ARMY_IDS = {_UNIT_IDS[name] for name in ("MARINE", "MARAUDER", "REAPER", "HELLION", "SIEGETANK", "MEDIVAC", "VIKINGFIGHTER", "BATTLECRUISER")}
_UNIT_VALUES = {_UNIT_IDS["MARINE"]: 50.0, _UNIT_IDS["MARAUDER"]: 125.0, _UNIT_IDS["REAPER"]: 50.0,
                _UNIT_IDS["HELLION"]: 100.0, _UNIT_IDS["SIEGETANK"]: 275.0, _UNIT_IDS["MEDIVAC"]: 200.0,
                _UNIT_IDS["VIKINGFIGHTER"]: 225.0, _UNIT_IDS["BATTLECRUISER"]: 700.0}
_BUILD_ABILITY_IDS = {MacroAction.BUILD_SUPPLY: 319, MacroAction.BUILD_BARRACKS: 321,
                      MacroAction.BUILD_FACTORY: 328, MacroAction.BUILD_STARPORT: 329,
                      MacroAction.RESEARCH_UPGRADE: 322}


@dataclass(frozen=True)
class PySC2BackendConfig:
    """PySC2 launch settings for a one-agent versus built-in-bot match."""

    map_name: str
    map_file: str | None = None
    race: str = "Terran"
    opponent_type: str = "builtin"
    opponent_race: str = "Terran"
    opponent_difficulty: str = "Medium"
    realtime: bool = False
    timeout_seconds: float = 120.0
    command_cooldown_game_loops: int = 64
    map_width: float = 200.0
    map_height: float = 200.0


def _field(unit: Any, name: str, default: int | float = 0) -> int | float:
    """Read a raw-unit field from PySC2 or a simple test double."""
    if isinstance(unit, dict):
        return unit.get(name, default)
    try:
        return unit[name]
    except (IndexError, KeyError, TypeError):
        return getattr(unit, name, default)


def _units(observation: dict[str, Any]) -> list[Any]:
    value = observation.get("raw_units", [])
    return list(value) if value is not None else []


class PySC2StateAdapter:
    """Convert visible PySC2 raw units into the project's canonical state."""

    SELF, ENEMY = 1, 4

    def __init__(self, map_width: float, map_height: float) -> None:
        self.map_width, self.map_height = map_width, map_height
        self.reset()

    def reset(self) -> None:
        """Clear local scouting memory at an episode boundary."""
        self._enemy_counts: Counter[int] = Counter()
        self._enemy_position: tuple[float, float] | None = None
        self._last_seen_loop: int | None = None
        self._previous_own_values: Counter[int] = Counter()
        self._lost_army_value = 0.0

    def extract(self, observation: dict[str, Any], pending_actions: dict[str, bool]) -> dict[str, Any]:
        """Build state only from the agent's raw observation and local memory."""
        raw_units = _units(observation)
        own = [unit for unit in raw_units if int(_field(unit, "alliance")) == self.SELF]
        enemy = [unit for unit in raw_units if int(_field(unit, "alliance")) == self.ENEMY]
        if enemy:
            self._enemy_counts = Counter(int(_field(unit, "unit_type")) for unit in enemy)
            points = [(float(_field(unit, "x")), float(_field(unit, "y"))) for unit in enemy]
            self._enemy_position = (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))
            self._last_seen_loop = self._game_loop(observation)
        counts = Counter(int(_field(unit, "unit_type")) for unit in own)
        player = np.asarray(observation.get("player", []), dtype=np.float32)
        minerals = float(player[1]) if player.size > 1 else 0.0
        vespene = float(player[2]) if player.size > 2 else 0.0
        supply_used = float(player[3]) if player.size > 3 else 0.0
        supply_cap = float(player[4]) if player.size > 4 else 0.0
        workers = [unit for unit in own if int(_field(unit, "unit_type")) == _UNIT_IDS["SCV"]]
        army = [unit for unit in own if int(_field(unit, "unit_type")) in _ARMY_IDS]
        game_loop = self._game_loop(observation)
        army_value = sum(_UNIT_VALUES.get(int(_field(unit, "unit_type")), 50.0) for unit in army)
        army_counts = Counter(int(_field(unit, "unit_type")) for unit in army)
        for unit_type, previous_count in self._previous_own_values.items():
            removed = max(0, previous_count - army_counts.get(unit_type, 0))
            self._lost_army_value += removed * _UNIT_VALUES.get(unit_type, 50.0)
        self._previous_own_values = army_counts
        enemy_value = sum(_UNIT_VALUES.get(unit_type, 50.0) * count for unit_type, count in self._enemy_counts.items())
        own_center = self._center(army)
        return {
            "game_time": game_loop / 22.4, "minerals": minerals, "vespene": vespene,
            "current_supply": supply_used, "maximum_supply": supply_cap,
            "worker_count": len(workers), "base_count": self._count(counts, "COMMANDCENTER", "ORBITALCOMMAND", "PLANETARYFORTRESS"),
            "idle_worker_count": sum(int(_field(unit, "order_length")) == 0 for unit in workers),
            "army_value": army_value, "lost_army_value": self._lost_army_value,
            "buildings": {"command_center": self._count(counts, "COMMANDCENTER", "ORBITALCOMMAND", "PLANETARYFORTRESS"),
                          "supply_depot": self._count(counts, "SUPPLYDEPOT"),
                          "barracks": self._count(counts, "BARRACKS"), "factory": self._count(counts, "FACTORY"),
                          "starport": self._count(counts, "STARPORT"), "refinery": self._count(counts, "REFINERY"),
                          "engineering_bay": self._count(counts, "ENGINEERINGBAY"), "tech_lab": counts[5], "reactor": counts[6]},
            "completed_upgrade_flags": [int(value) for value in np.asarray(observation.get("upgrades", []), dtype=np.int32)[:6]],
            "production_queue_summary": [sum(int(_field(unit, "order_length")) for unit in own if int(_field(unit, "unit_type")) == _UNIT_IDS[name]) for name in ("BARRACKS", "FACTORY", "STARPORT", "REFINERY", "COMMANDCENTER")],
            "units": {"marine": counts[_UNIT_IDS["MARINE"]], "marauder": counts[_UNIT_IDS["MARAUDER"]], "reaper": counts[_UNIT_IDS["REAPER"]],
                      "hellion": counts[_UNIT_IDS["HELLION"]], "tank": counts[_UNIT_IDS["SIEGETANK"]], "medivac": counts[_UNIT_IDS["MEDIVAC"]],
                      "viking": counts[_UNIT_IDS["VIKINGFIGHTER"]], "battlecruiser": counts[_UNIT_IDS["BATTLECRUISER"]]},
            "average_army_health": self._average_health(army), "army_center_x": own_center[0], "army_center_y": own_center[1],
            "number_of_army_groups": int(bool(army)), "enemy_army_value_estimate": enemy_value,
            "enemy": {"observed_unit_counts": {str(unit_type): count for unit_type, count in self._enemy_counts.items()}, "observed_buildings": {"base": sum(count for unit_type, count in self._enemy_counts.items() if unit_type in {18, 59, 86, 100, 101, 130, 132}), "production": sum(count for unit_type, count in self._enemy_counts.items() if unit_type in {21, 27, 28, 62, 67, 71, 89, 91, 92}), "ground_tech": 0, "air_tech": 0},
                      "last_seen_army_position": self._enemy_position, "time_since_last_scout": 3600.0 if self._last_seen_loop is None else max(0.0, (game_loop - self._last_seen_loop) / 22.4),
                      "estimated_army_value": enemy_value, "estimated_worker_count": float(sum(count for unit_type, count in self._enemy_counts.items() if unit_type in {45, 84, 104})), "strategy_probabilities": {"unknown": 1.0}},
            "map_control": {}, "pending_actions": dict(pending_actions),
            # Gathering workers remain valid construction candidates; raw SC2
            # commands replace their harvest order when they start building.
            "available": {"worker": bool(workers), "command_center": self._idle(own, _UNIT_IDS["COMMANDCENTER"]),
                          "barracks": self._idle(own, _UNIT_IDS["BARRACKS"]), "factory": self._idle(own, _UNIT_IDS["FACTORY"]),
                          "supply_depot": self._idle(own, _UNIT_IDS["SUPPLYDEPOT"]),
                          "starport": self._idle(own, _UNIT_IDS["STARPORT"]),
                          "engineering_bay": self._idle(own, _UNIT_IDS["ENGINEERINGBAY"]), "scout": bool(army),
                          "defend_target": bool(army), "natural_target": self._count(counts, "COMMANDCENTER") > 1,
                          "enemy_target": self._enemy_position is not None, "enemy_natural_target": self._enemy_position is not None,
                          # Every ladder game has a known opposing spawn.  The
                          # executor falls back to the mirrored start point
                          # until scouting supplies a more precise position.
                          "enemy_main_target": True},
        }

    @staticmethod
    def _game_loop(observation: dict[str, Any]) -> int:
        value = np.asarray(observation.get("game_loop", [0])).reshape(-1)
        return int(value[0]) if value.size else 0

    @staticmethod
    def _average_health(units: list[Any]) -> float:
        if not units:
            return 1.0
        return float(np.mean([float(_field(unit, "health")) / max(1.0, float(_field(unit, "health_max", 1))) for unit in units]))

    @staticmethod
    def _center(units: list[Any]) -> tuple[float, float]:
        if not units:
            return 0.0, 0.0
        return float(np.mean([_field(unit, "x") for unit in units])), float(np.mean([_field(unit, "y") for unit in units]))

    @staticmethod
    def _idle(units: list[Any], unit_type: int) -> bool:
        # Feature-layer raw units use a 0--100 build scale, while replay
        # controller protobufs use 0--1.  Support both without accepting an
        # in-progress structure as an available producer/prerequisite.
        def completed(unit: Any) -> bool:
            progress = float(_field(unit, "build_progress", 100.0))
            return progress >= 100.0 or (0.0 <= progress <= 1.0 and progress >= 0.999)
        return any(int(_field(unit, "unit_type")) == unit_type and int(_field(unit, "order_length")) == 0 and completed(unit) for unit in units)

    @staticmethod
    def _count(counts: Counter[int], *names: str) -> int:
        return sum(counts[_UNIT_IDS[name]] for name in names)


class PySC2MacroActionExecutor:
    """Translate Terran macro actions into legal PySC2 raw function calls."""

    def __init__(self, width: float, height: float, cooldown_game_loops: int = 64) -> None:
        self.width, self.height = width, height
        self.cooldown_game_loops = max(1, int(cooldown_game_loops))
        self.pending_actions: dict[str, bool] = {}
        self._pending_until: dict[MacroAction, int] = {}

    def clear_completed(self) -> None:
        """Clear all game-local command reservations at an episode boundary."""
        self.pending_actions.clear()
        self._pending_until.clear()

    def advance(self, game_loop: int) -> None:
        """Release only requests whose cooldown survived a later observation."""
        for action, until in list(self._pending_until.items()):
            if game_loop >= until:
                self._pending_until.pop(action, None)
                self.pending_actions.pop(action.name, None)

    def is_ready(self, action: MacroAction) -> bool:
        """Expose command reservations to the backend action mask."""
        return action not in self._pending_until

    def reserve(self, action: MacroAction, issued_loop: int, duration: int) -> None:
        """Block duplicate requests until SC2 has advanced through a cooldown."""
        self._pending_until[action] = int(issued_loop) + max(int(duration), self.cooldown_game_loops)

    def action(self, macro: MacroAction, observation: dict[str, Any], enemy_position: tuple[float, float] | None,
               game_loop: int, duration: int, build_point: tuple[float, float] | None = None) -> tuple[Any, dict[str, Any]]:
        """Return one raw action and execution metadata; failures become no-ops."""
        try:
            from pysc2.lib import actions
        except ImportError as exc:
            raise RuntimeError("PySC2 is required for the real-SC2 backend. Install it in a Python 3.10 environment.") from exc
        raw = actions.RAW_FUNCTIONS
        if macro == MacroAction.NO_OP:
            return raw.no_op(), self._result(True, macro)
        if not self.is_ready(macro):
            return raw.no_op(), self._result(False, macro, "cooldown_active")
        units = _units(observation)
        own = [unit for unit in units if int(_field(unit, "alliance")) == PySC2StateAdapter.SELF]
        worker = self._first(own, _UNIT_IDS["SCV"])
        army = [int(_field(unit, "tag")) for unit in own if int(_field(unit, "unit_type")) in _ARMY_IDS]
        target = enemy_position or self._opposite(self._first_position(own, _UNIT_IDS["COMMANDCENTER"]))
        try:
            if macro == MacroAction.TRAIN_WORKERS:
                return self._quick(raw, "Train_SCV_quick", self._first_tag(own, _UNIT_IDS["COMMANDCENTER"]), macro)
            if macro == MacroAction.BUILD_SUPPLY:
                return self._building_point(raw, "Build_SupplyDepot_pt", worker, build_point, macro)
            if macro == MacroAction.BUILD_BARRACKS:
                return self._building_point(raw, "Build_Barracks_pt", worker, build_point, macro)
            if macro == MacroAction.BUILD_REFINERY:
                geyser = self._first_neutral_geyser(units, own)
                if geyser is None:
                    return raw.no_op(), self._result(False, macro, "no_available_geyser")
                # PySC2 labels this ability ``_pt``, but its raw signature
                # targets the neutral geyser's unit tag.
                return self._unit(raw, "Build_Refinery_pt", worker, geyser, macro)
            if macro == MacroAction.BUILD_FACTORY:
                return self._building_point(raw, "Build_Factory_pt", worker, build_point, macro)
            if macro == MacroAction.BUILD_STARPORT:
                return self._building_point(raw, "Build_Starport_pt", worker, build_point, macro)
            if macro == MacroAction.EXPAND:
                return self._point(raw, "Build_CommandCenter_pt", worker, self._toward_map_center(self._first_position(own, _UNIT_IDS["COMMANDCENTER"]), 28.0), macro)
            if macro == MacroAction.TRAIN_BASIC_ARMY:
                return self._quick(raw, "Train_Marine_quick", self._first_tag(own, _UNIT_IDS["BARRACKS"]), macro)
            if macro == MacroAction.TRAIN_ANTI_GROUND:
                barracks = self._first(own, _UNIT_IDS["BARRACKS"])
                if barracks is None:
                    return raw.no_op(), self._result(False, macro, "no_producer")
                if int(_field(barracks, "add_on_tag")) == 0:
                    return self._quick(raw, "Build_TechLab_Barracks_quick", int(_field(barracks, "tag")), macro, skill_phase="build_tech_lab")
                return self._quick(raw, "Train_Marauder_quick", int(_field(barracks, "tag")), macro, skill_phase="train_marauder")
            if macro == MacroAction.TRAIN_ANTI_AIR:
                return self._quick(raw, "Train_VikingFighter_quick", self._first_tag(own, _UNIT_IDS["STARPORT"]), macro)
            if macro == MacroAction.RESEARCH_UPGRADE:
                engineering_bay = self._first(own, _UNIT_IDS["ENGINEERINGBAY"])
                if engineering_bay is None:
                    return self._building_point(raw, "Build_EngineeringBay_pt", worker, build_point, macro, skill_phase="build_engineering_bay")
                return self._quick(raw, "Research_TerranInfantryWeaponsLevel1_quick", int(_field(engineering_bay, "tag")), macro, skill_phase="research_infantry_weapons")
            if macro in (MacroAction.SCOUT_ENEMY_MAIN, MacroAction.SCOUT_EXPANSION):
                scout = self._scout_unit(own)
                if scout is None:
                    return raw.no_op(), self._result(False, macro, "no_army_scout")
                return self._point(raw, "Move_pt", scout, target, macro)
            if macro in (MacroAction.DEFEND_MAIN, MacroAction.DEFEND_NATURAL, MacroAction.RETREAT):
                return self._army_point(raw, army, self._first_position(own, _UNIT_IDS["COMMANDCENTER"]), macro, move=True)
            if macro in (MacroAction.HARASS, MacroAction.ATTACK_ENEMY_NATURAL, MacroAction.ATTACK_ENEMY_MAIN):
                return self._army_point(raw, army, target, macro, move=False)
            return raw.no_op(), self._result(False, macro, "unsupported_action")
        except (AttributeError, KeyError, ValueError, TypeError) as exc:
            return raw.no_op(), self._result(False, macro, "command_unavailable", exception=type(exc).__name__)

    def _quick(self, raw: Any, name: str, tag: int | None, macro: MacroAction, **details: Any) -> tuple[Any, dict[str, Any]]:
        if tag is None:
            return raw.no_op(), self._result(False, macro, "no_producer")
        return getattr(raw, name)("now", [tag]), self._result(True, macro, pending=True, **details)

    def _point(self, raw: Any, name: str, unit: Any | None, point: tuple[float, float], macro: MacroAction, **details: Any) -> tuple[Any, dict[str, Any]]:
        if unit is None:
            return raw.no_op(), self._result(False, macro, "no_worker")
        # Raw-unit positions are already represented in PySC2's raw-action
        # coordinate system.  Transforming them a second time sends commands
        # to a different point on the map.
        return getattr(raw, name)("now", [int(_field(unit, "tag"))], point), self._result(
            True, macro, pending=True, target_position=(float(point[0]), float(point[1])), **details
        )

    def _building_point(self, raw: Any, name: str, worker: Any | None, point: tuple[float, float] | None,
                        macro: MacroAction, **details: Any) -> tuple[Any, dict[str, Any]]:
        """Issue a placement command only after the backend found a legal tile."""
        if point is None:
            return raw.no_op(), self._result(False, macro, "no_legal_build_location", **details)
        return self._point(raw, name, worker, point, macro, **details)

    def _unit(self, raw: Any, name: str, unit: Any | None, target: Any | None, macro: MacroAction) -> tuple[Any, dict[str, Any]]:
        if unit is None:
            return raw.no_op(), self._result(False, macro, "no_worker")
        if target is None:
            return raw.no_op(), self._result(False, macro, "no_available_geyser")
        return getattr(raw, name)("now", [int(_field(unit, "tag"))], int(_field(target, "tag"))), self._result(True, macro, pending=True)

    def _army_point(self, raw: Any, tags: list[int], point: tuple[float, float], macro: MacroAction, *, move: bool) -> tuple[Any, dict[str, Any]]:
        if not tags:
            return raw.no_op(), self._result(False, macro, "no_army")
        function = raw.Move_pt if move else raw.Attack_pt
        return function("now", tags, point), self._result(True, macro, pending=True)

    @staticmethod
    def _first(units: list[Any], unit_type: int) -> Any | None:
        return next((unit for unit in units if int(_field(unit, "unit_type")) == unit_type), None)

    def _first_tag(self, units: list[Any], unit_type: int) -> int | None:
        unit = self._first(units, unit_type)
        return int(_field(unit, "tag")) if unit is not None else None

    def _first_position(self, units: list[Any], unit_type: int) -> tuple[float, float]:
        unit = self._first(units, unit_type)
        return self._position(unit) if unit is not None else (self.width * 0.2, self.height * 0.2)

    @staticmethod
    def _position(unit: Any) -> tuple[float, float]:
        """Return a raw unit's map point in PySC2 point-action format."""
        return float(_field(unit, "x")), float(_field(unit, "y"))

    def _first_neutral_geyser(self, units: list[Any], own: list[Any]) -> Any | None:
        """Choose an unoccupied visible geyser nearest the first Command Center."""
        geysers = [unit for unit in units if int(_field(unit, "alliance")) == 3 and int(_field(unit, "unit_type")) == 342]
        if not geysers:
            return None
        occupied = [unit for unit in own if int(_field(unit, "unit_type")) == _UNIT_IDS["REFINERY"]]
        base = self._first_position(own, _UNIT_IDS["COMMANDCENTER"])
        def score(geyser: Any) -> float:
            x, y = float(_field(geyser, "x")), float(_field(geyser, "y"))
            if any((x - float(_field(refinery, "x"))) ** 2 + (y - float(_field(refinery, "y"))) ** 2 < 4.0 for refinery in occupied):
                return float("inf")
            return (x - base[0]) ** 2 + (y - base[1]) ** 2
        candidate = min(geysers, key=score)
        return None if score(candidate) == float("inf") else candidate

    @staticmethod
    def _scout_unit(units: list[Any]) -> Any | None:
        """Use a mobile army unit for scouting instead of sacrificing an SCV."""
        preferred = (_UNIT_IDS["REAPER"], _UNIT_IDS["MARINE"])
        for unit_type in preferred:
            unit = next((item for item in units if int(_field(item, "unit_type")) == unit_type), None)
            if unit is not None:
                return unit
        return next((item for item in units if int(_field(item, "unit_type")) in _ARMY_IDS), None)

    def _build_point(self, units: list[Any]) -> tuple[float, float]:
        x, y = self._first_position(units, _UNIT_IDS["COMMANDCENTER"])
        # Ladder starts can be in any corner.  Route structures inward, but
        # use separate slots so a Barracks cannot be placed on its Depot.
        dx, dy = self.width * 0.5 - x, self.height * 0.5 - y
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        building_count = sum(int(_field(unit, "unit_type")) in _BUILDING_IDS for unit in units)
        slot = building_count % 4
        lateral_offset = (slot - 1.5) * 8.0
        forward_distance = 16.0 + (building_count // 4) * 8.0
        lateral_x, lateral_y = -dy / length, dx / length
        return (min(self.width - 2.0, max(2.0, x + forward_distance * dx / length + lateral_offset * lateral_x)),
                min(self.height - 2.0, max(2.0, y + forward_distance * dy / length + lateral_offset * lateral_y)))

    def build_candidates(self, units: list[Any]) -> list[tuple[float, float]]:
        """Return deterministic, separated raw-action coordinates for structures."""
        x, y = self._first_position(units, _UNIT_IDS["COMMANDCENTER"])
        dx, dy = self.width * 0.5 - x, self.height * 0.5 - y
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        forward_x, forward_y = dx / length, dy / length
        lateral_x, lateral_y = -forward_y, forward_x
        candidates = [self._build_point(units)]
        for distance in (16.0, 24.0, 32.0, 40.0, 48.0):
            for lateral in (-24.0, -12.0, 0.0, 12.0, 24.0):
                point = (min(self.width - 2.0, max(2.0, x + distance * forward_x + lateral * lateral_x)),
                         min(self.height - 2.0, max(2.0, y + distance * forward_y + lateral * lateral_y)))
                if all((point[0] - existing[0]) ** 2 + (point[1] - existing[1]) ** 2 > 1e-4 for existing in candidates):
                    candidates.append(point)
        return candidates

    def _toward_map_center(self, point: tuple[float, float], distance: float) -> tuple[float, float]:
        """Place buildings and expansions from a start location toward map center."""
        x, y = point
        dx, dy = self.width * 0.5 - x, self.height * 0.5 - y
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        return (min(self.width - 2.0, max(2.0, x + distance * dx / length)),
                min(self.height - 2.0, max(2.0, y + distance * dy / length)))

    def _opposite(self, point: tuple[float, float]) -> tuple[float, float]:
        return max(2.0, self.width - point[0]), max(2.0, self.height - point[1])

    def _result(self, success: bool, action: MacroAction, failure_reason: str | None = None, **details: Any) -> dict[str, Any]:
        if success and details.get("pending"):
            self.pending_actions[action.name] = True
        return {"success": success, "action": action.name, "issued_commands": int(success and action != MacroAction.NO_OP), "failure_reason": failure_reason, **details}


class PySC2Backend:
    """Own a PySC2 ``SC2Env`` and satisfy :class:`RealSC2Backend`."""

    def __init__(self, config: PySC2BackendConfig) -> None:
        self.config = config
        self._env: Any | None = None
        self._time_step: Any | None = None
        self._adapter = PySC2StateAdapter(config.map_width, config.map_height)
        self._executor = PySC2MacroActionExecutor(config.map_width, config.map_height, config.command_cooldown_game_loops)
        self._last_state: dict[str, Any] | None = None
        self._started_at: float | None = None

    def reset_game(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a fresh non-realtime PySC2 match and return its first state."""
        self.close()
        try:
            from absl import flags
            from pysc2.env import sc2_env
            from pysc2.lib import actions, features
        except ImportError as exc:
            raise RuntimeError("PySC2 is required. Use the project's Python 3.10 PySC2 environment.") from exc
        # PySC2 reads its Abseil flags while constructing SC2Env.  The project
        # uses argparse, so initialise an empty, known-only flag set once.
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["sc2wmrl-pysc2"], known_only=True)
        race = self._enum(sc2_env.Race, self.config.race)
        opponent_race = self._enum(sc2_env.Race, self.config.opponent_race)
        difficulty = self._enum(sc2_env.Difficulty, self.config.opponent_difficulty)
        interface = features.AgentInterfaceFormat(action_space=actions.ActionSpace.RAW, use_raw_units=True, use_raw_actions=True,
                                                  raw_resolution=int(max(self.config.map_width, self.config.map_height)))
        map_setting: Any = self.config.map_name
        if self.config.map_file:
            from pysc2.maps import lib as maps_lib

            # Battle.net installs commonly place ladder maps directly under
            # ``Maps`` while PySC2's historical registry expects a season
            # subdirectory.  A configured Map preserves the installed path.
            map_setting = type("ConfiguredLocalMap", (maps_lib.Map,), {
                "filename": self.config.map_file, "players": 2,
            })()
        self._env = sc2_env.SC2Env(map_name=map_setting, players=[sc2_env.Agent(race), sc2_env.Bot(opponent_race, difficulty)],
                                   agent_interface_format=interface, step_mul=1, realtime=self.config.realtime,
                                   game_steps_per_episode=0, random_seed=seed, ensure_available_actions=False)
        self._adapter.reset(); self._executor.clear_completed()
        self._time_step = self._env.reset()[0]
        self._started_at = time.monotonic()
        return self._publish("RESET")

    def _select_build_point(self, action: MacroAction) -> tuple[float, float] | None:
        """Ask SC2 to validate candidate raw-action tiles for a structure.

        A raw command can be syntactically accepted even when its target tile
        is blocked.  The placement query prevents the collector and PPO from
        recording those silent no-op transitions as successful building skills.
        """
        ability_id = _BUILD_ABILITY_IDS.get(action)
        if ability_id is None:
            return None
        if self._env is None or self._time_step is None:
            return None
        own = [unit for unit in _units(self._time_step.observation) if int(_field(unit, "alliance")) == PySC2StateAdapter.SELF]
        worker = self._executor._first(own, _UNIT_IDS["SCV"])
        if worker is None:
            return None
        try:
            from pysc2.lib.point import Point
            from s2clientprotocol import common_pb2, query_pb2

            transform = self._env._features[0]._world_to_minimap_px
            candidates = self._executor.build_candidates(own)
            placements = []
            for point in candidates:
                world = transform.back_pt(Point(float(point[0]), float(point[1])))
                placements.append(query_pb2.RequestQueryBuildingPlacement(
                    ability_id=ability_id, target_pos=common_pb2.Point2D(x=world.x, y=world.y),
                    placing_unit_tag=int(_field(worker, "tag")),
                ))
            response = self._env._controllers[0].query(query_pb2.RequestQuery(placements=placements))
            for point, result in zip(candidates, response.placements):
                if int(result.result) == 1:  # ActionResult.Success
                    return point
            return None
        except (AttributeError, IndexError, TypeError, ValueError):
            # A backend/API mismatch must not issue an unchecked construction
            # order.  The returned execution metadata remains diagnosable.
            return None

    def execute_macro(self, action: MacroAction, duration: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Issue one raw macro command, advance exactly ``duration`` loops, and publish state."""
        if self._env is None or self._time_step is None:
            raise RuntimeError("real SC2 game is not running; call reset_game first")
        if self._started_at is not None and time.monotonic() - self._started_at >= self.config.timeout_seconds:
            state, info = self._publish({"success": False, "action": action.name, "issued_commands": 0,
                                         "failure_reason": "match_timeout", "submission_status": "not_submitted"})
            info["outcome"] = "draw"
            info["match_timeout"] = True
            return state, 0.0, False, True, info
        issued_loop = self._game_loop()
        build_point = self._select_build_point(action)
        command, execution = self._executor.action(action, self._time_step.observation, self._adapter._enemy_position, issued_loop, duration, build_point)
        if execution.get("success") and execution.get("pending"):
            self._executor.reserve(action, issued_loop, duration)
        self._time_step = self._env.step([command], step_mul=max(1, int(duration)))[0]
        self._executor.advance(self._game_loop())
        execution = self._with_action_feedback(execution)
        state, info = self._publish(execution)
        terminated = bool(self._time_step.last())
        info["outcome"] = "win" if terminated and self._time_step.reward > 0 else "loss" if terminated and self._time_step.reward < 0 else "draw" if terminated else None
        return state, float(self._time_step.reward), terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        """Return macro legality from the current PySC2-derived visible state."""
        if self._last_state is None:
            mask = np.zeros(len(MacroAction), dtype=np.bool_); mask[MacroAction.NO_OP] = True; return mask
        mask = action_mask(self._last_state)
        for action in MacroAction:
            if action != MacroAction.NO_OP and not self._executor.is_ready(action):
                mask[action] = False
        return mask

    def close(self) -> None:
        """Close the PySC2 process and clear all game-local state."""
        if self._env is not None:
            self._env.close()
        self._env = None; self._time_step = None; self._last_state = None; self._started_at = None

    def _publish(self, execution: str | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self._time_step is not None
        state = self._adapter.extract(self._time_step.observation, self._executor.pending_actions)
        self._last_state = state
        return state, {"environment_type": "real_sc2", "map_name": self.config.map_name, "opponent_id": self.config.opponent_type,
                       "opponent_type": self.config.opponent_type, "game_loop": self._game_loop(),
                       "execution": execution}

    def _game_loop(self) -> int:
        assert self._time_step is not None
        value = np.asarray(self._time_step.observation.get("game_loop", [0])).reshape(-1)
        return int(value[0]) if value.size else 0

    def _with_action_feedback(self, execution: dict[str, Any]) -> dict[str, Any]:
        """Attach SC2's post-step action result instead of assuming a submission worked."""
        assert self._time_step is not None
        reported = np.asarray(self._time_step.observation.get("action_result", [])).reshape(-1)
        result_codes = [int(value) for value in reported]
        accepted = bool(execution.get("success")) and not any(result_codes)
        receipt = dict(execution)
        receipt["sc2_action_results"] = result_codes
        receipt["submission_status"] = "accepted" if accepted else ("rejected" if execution.get("success") else "not_submitted")
        if execution.get("success") and not accepted:
            receipt["success"] = False
            receipt["failure_reason"] = "sc2_action_rejected"
        return receipt

    @staticmethod
    def _enum(enum_type: Any, name: str) -> Any:
        normalized = str(name).replace(" ", "_").lower()
        for candidate in (normalized, normalized.replace("_", "")):
            if hasattr(enum_type, candidate):
                return getattr(enum_type, candidate)
        raise ValueError(f"unsupported PySC2 enum value {name!r} for {enum_type.__name__}")
