"""Map low-level replay commands into the project's stable macro vocabulary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import hypot
from typing import Any, Iterable, Mapping

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction


@dataclass(frozen=True)
class MacroLabel:
    """One interval's macro action, confidence, and audit evidence."""

    action: MacroAction
    confidence: float
    evidence: tuple[str, ...]
    ability_ids: tuple[int, ...]
    candidate_actions: tuple[int, ...]
    source: str


class AbilityNameResolver:
    """Resolve replay ability IDs through PySC2's currently installed table."""

    _FALLBACK = {
        318: ("Build_CommandCenter_pt",), 319: ("Build_SupplyDepot_pt",),
        320: ("Build_Refinery_pt",), 321: ("Build_Barracks_pt",),
        328: ("Build_Factory_pt",), 329: ("Build_Starport_pt",),
        524: ("Train_SCV_quick",), 560: ("Train_Marine_quick",),
        561: ("Train_Reaper_quick",), 563: ("Train_Marauder_quick",),
        591: ("Train_SiegeTank_quick",), 595: ("Train_Hellion_quick",),
        620: ("Train_Medivac_quick",), 624: ("Train_VikingFighter_quick",),
    }

    def __init__(self, mapping: Mapping[int, Iterable[str]] | None = None) -> None:
        if mapping is not None:
            self._names = {int(key): tuple(str(name) for name in value) for key, value in mapping.items()}
            return
        names: dict[int, set[str]] = {key: set(value) for key, value in self._FALLBACK.items()}
        try:
            from pysc2.lib import actions
            for table_name in ("RAW_FUNCTIONS", "FUNCTIONS"):
                for function in getattr(actions, table_name, ()):
                    ability_id = int(getattr(function, "ability_id", 0) or 0)
                    if ability_id:
                        names.setdefault(ability_id, set()).add(str(function.name))
        except ImportError:
            # Metadata inspection and offline unit tests do not need PySC2.
            pass
        self._names = {key: tuple(sorted(value)) for key, value in names.items()}

    def names(self, ability_id: int) -> tuple[str, ...]:
        """Return known aliases, preserving an audit-friendly unknown fallback."""
        return self._names.get(int(ability_id), (f"ability_{int(ability_id)}",))


_PRIORITY = {
    MacroAction.EXPAND: 100, MacroAction.BUILD_STARPORT: 95, MacroAction.BUILD_FACTORY: 94,
    MacroAction.BUILD_BARRACKS: 93, MacroAction.BUILD_REFINERY: 92, MacroAction.BUILD_SUPPLY: 91,
    MacroAction.RESEARCH_UPGRADE: 88, MacroAction.TRAIN_ANTI_AIR: 82,
    MacroAction.TRAIN_ANTI_GROUND: 81, MacroAction.TRAIN_BASIC_ARMY: 80,
    MacroAction.TRAIN_WORKERS: 79, MacroAction.ATTACK_ENEMY_MAIN: 75,
    MacroAction.ATTACK_ENEMY_NATURAL: 74, MacroAction.HARASS: 73,
    MacroAction.DEFEND_MAIN: 72, MacroAction.DEFEND_NATURAL: 71,
    MacroAction.RETREAT: 70, MacroAction.SCOUT_ENEMY_MAIN: 65,
    MacroAction.SCOUT_EXPANSION: 64, MacroAction.NO_OP: 0,
}
_WORKER = 45
_ARMY = {33, 35, 48, 49, 51, 53, 54, 57}


def _field(value: Any, name: str, default: Any = 0) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _unit_command(action: Any) -> Any | None:
    for name in ("action_raw", "action_feature_layer", "action_render"):
        container = getattr(action, name, None)
        if container is not None and getattr(container, "HasField", lambda _: False)("unit_command"):
            return container.unit_command
    return None


def _target(command: Any) -> tuple[float, float] | None:
    if getattr(command, "HasField", lambda _: False)("target_world_space_pos"):
        point = command.target_world_space_pos
        return float(point.x), float(point.y)
    return None


def _named_action(name: str) -> MacroAction | None:
    key = name.lower().replace("_", "")
    if "trainscv" in key:
        return MacroAction.TRAIN_WORKERS
    if "buildsupplydepot" in key:
        return MacroAction.BUILD_SUPPLY
    if "buildbarracks" in key:
        return MacroAction.BUILD_BARRACKS
    if "buildrefinery" in key:
        return MacroAction.BUILD_REFINERY
    if "buildfactory" in key:
        return MacroAction.BUILD_FACTORY
    if "buildstarport" in key:
        return MacroAction.BUILD_STARPORT
    if "buildcommandcenter" in key:
        return MacroAction.EXPAND
    if any(token in key for token in ("trainmarine", "trainreaper")):
        return MacroAction.TRAIN_BASIC_ARMY
    if any(token in key for token in ("trainmarauder", "trainhellion", "trainsiegetank", "trainwidowmine", "trainmedivac")):
        return MacroAction.TRAIN_ANTI_GROUND
    if any(token in key for token in ("trainviking", "trainliberator", "trainbattlecruiser")):
        return MacroAction.TRAIN_ANTI_AIR
    if key.startswith("research") or any(token in key for token in ("buildtechlab", "buildreactor", "buildengineeringbay", "buildarmory", "morphorbitalcommand")):
        return MacroAction.RESEARCH_UPGRADE
    if "effectscan" in key:
        return MacroAction.SCOUT_ENEMY_MAIN
    return None


class ReplayMacroLabeler:
    """Compress commands issued inside one macro interval into one label."""

    def __init__(self, resolver: AbilityNameResolver | None = None) -> None:
        self.resolver = resolver or AbilityNameResolver()

    def infer(self, actions: Iterable[Any], previous_state: Mapping[str, Any], next_state: Mapping[str, Any],
              previous_units: Mapping[int, Any], *, own_main: tuple[float, float] | None = None,
              enemy_main: tuple[float, float] | None = None) -> MacroLabel:
        """Infer a label from commands, then a low-confidence visible state delta."""
        candidates: list[MacroAction] = []
        confidence: list[float] = []
        evidence: list[str] = []
        ability_ids: list[int] = []
        for action in actions:
            command = _unit_command(action)
            if command is None:
                continue
            ability_id = int(getattr(command, "ability_id", 0) or 0)
            if ability_id <= 0:
                continue
            ability_ids.append(ability_id)
            names = self.resolver.names(ability_id)
            resolved = next((value for name in names if (value := _named_action(name)) is not None), None)
            if resolved is not None:
                candidates.append(resolved); confidence.append(0.98); evidence.append(next(name for name in names if _named_action(name) == resolved))
                continue
            name_text = " ".join(names).lower()
            unit_types = {int(_field(previous_units[tag], "unit_type")) for tag in getattr(command, "unit_tags", ()) if int(tag) in previous_units}
            target = _target(command)
            if "attack" in name_text:
                candidates.append(self._target_action(target, unit_types, own_main, enemy_main, attack=True)); confidence.append(0.86); evidence.append(f"attack:{target}")
            elif target is not None and any(token in name_text for token in ("move", "patrol")):
                resolved = self._target_action(target, unit_types, own_main, enemy_main, attack=False)
                if resolved is not None:
                    candidates.append(resolved); confidence.append(0.72); evidence.append(f"move:{target}")
        source = "command"
        if not candidates:
            fallback = self._state_delta(previous_state, next_state)
            if fallback is not None:
                candidates, confidence, evidence, source = [fallback], [0.56], ["state_delta"], "state_delta"
        if not candidates:
            return MacroLabel(MacroAction.NO_OP, 1.0, (), tuple(ability_ids), (), "none")
        counts = Counter(candidates)
        selected = max(counts, key=lambda action: (_PRIORITY[action], counts[action]))
        selected_confidence = [value for action, value in zip(candidates, confidence) if action == selected]
        support = 0.7 + 0.3 * counts[selected] / len(candidates)
        return MacroLabel(selected, float(np.clip(np.mean(selected_confidence) * support, 0.0, 1.0)), tuple(evidence),
                          tuple(ability_ids), tuple(int(action) for action in candidates), source)

    @staticmethod
    def _target_action(target: tuple[float, float] | None, unit_types: set[int], own_main: tuple[float, float] | None,
                       enemy_main: tuple[float, float] | None, *, attack: bool) -> MacroAction:
        distance = lambda a, b: hypot(a[0] - b[0], a[1] - b[1])
        if target is not None and own_main is not None and distance(target, own_main) <= 28.0:
            return MacroAction.DEFEND_MAIN
        if target is not None and enemy_main is not None and distance(target, enemy_main) <= 35.0:
            return MacroAction.ATTACK_ENEMY_MAIN if attack else MacroAction.SCOUT_ENEMY_MAIN
        if unit_types and unit_types <= {_WORKER}:
            return MacroAction.SCOUT_EXPANSION
        return MacroAction.HARASS if attack else MacroAction.SCOUT_EXPANSION

    @staticmethod
    def _state_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> MacroAction | None:
        before, after = previous.get("buildings", {}), current.get("buildings", {})
        for name, action in (("command_center", MacroAction.EXPAND), ("starport", MacroAction.BUILD_STARPORT),
                             ("factory", MacroAction.BUILD_FACTORY), ("barracks", MacroAction.BUILD_BARRACKS),
                             ("refinery", MacroAction.BUILD_REFINERY)):
            if int(after.get(name, 0)) > int(before.get(name, 0)):
                return action
        if float(current.get("maximum_supply", 0)) > float(previous.get("maximum_supply", 0)):
            return MacroAction.BUILD_SUPPLY
        if int(current.get("worker_count", 0)) > int(previous.get("worker_count", 0)):
            return MacroAction.TRAIN_WORKERS
        before_units, after_units = previous.get("units", {}), current.get("units", {})
        if any(int(after_units.get(name, 0)) > int(before_units.get(name, 0)) for name in ("viking", "battlecruiser")):
            return MacroAction.TRAIN_ANTI_AIR
        if any(int(after_units.get(name, 0)) > int(before_units.get(name, 0)) for name in ("marauder", "hellion", "tank", "medivac")):
            return MacroAction.TRAIN_ANTI_GROUND
        if any(int(after_units.get(name, 0)) > int(before_units.get(name, 0)) for name in ("marine", "reaper")):
            return MacroAction.TRAIN_BASIC_ARMY
        return None


def replay_event_vector(action: MacroAction, outcome: str | None = None) -> np.ndarray:
    """Construct the existing seven-dimensional event target from a macro label."""
    events = np.zeros(7, dtype=np.float32)
    if action in {MacroAction.TRAIN_WORKERS, MacroAction.EXPAND}: events[0] = 1.0
    if action == MacroAction.BUILD_SUPPLY: events[1] = 1.0
    if action in {MacroAction.BUILD_BARRACKS, MacroAction.BUILD_REFINERY, MacroAction.BUILD_FACTORY, MacroAction.BUILD_STARPORT,
                  MacroAction.TRAIN_BASIC_ARMY, MacroAction.TRAIN_ANTI_GROUND, MacroAction.TRAIN_ANTI_AIR}: events[2] = 1.0
    if action == MacroAction.RESEARCH_UPGRADE: events[3] = 1.0
    if action in {MacroAction.SCOUT_ENEMY_MAIN, MacroAction.SCOUT_EXPANSION}: events[4] = 1.0
    if action in {MacroAction.DEFEND_MAIN, MacroAction.DEFEND_NATURAL, MacroAction.HARASS, MacroAction.ATTACK_ENEMY_NATURAL,
                  MacroAction.ATTACK_ENEMY_MAIN, MacroAction.RETREAT}: events[5] = 1.0
    if outcome is not None: events[6] = 1.0
    return events
