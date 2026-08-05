"""Race-neutral expert labels and race-specific macro labels for replays."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import hypot
from typing import Any, Iterable, Mapping

import numpy as np

from sc2wmrl.replays.macro_labeler import AbilityNameResolver, _target, _unit_command


class UniversalIntent(IntEnum):
    """Shared policy-independent intent vocabulary used by the world model."""

    NO_OP = 0
    GATHER_ECONOMY = 1
    TRAIN_WORKER = 2
    ADD_SUPPLY = 3
    EXPAND = 4
    BUILD_PRODUCTION = 5
    ADVANCE_TECH = 6
    TRAIN_ARMY = 7
    SCOUT = 8
    DEFEND = 9
    HARASS = 10
    ATTACK = 11
    RETREAT = 12
    CONTROL_MAP = 13


class RaceMacroAction(IntEnum):
    """Action class that is meaningful after conditioning on the player's race."""

    NO_OP = 0
    TRAIN_WORKER = 1
    BUILD_SUPPLY = 2
    BUILD_GAS = 3
    BUILD_BASIC_PRODUCTION = 4
    BUILD_TECH_PRODUCTION = 5
    BUILD_TECH = 6
    EXPAND = 7
    TRAIN_BASIC_GROUND = 8
    TRAIN_ARMORED_GROUND = 9
    TRAIN_AIR = 10
    TRAIN_SUPPORT = 11
    RESEARCH = 12
    COMBAT_ADVANCE = 13
    COMBAT_HOLD = 14
    COMBAT_RETREAT = 15
    COMBAT_HOME_DEFENSE = 16
    COMBAT_AWAY_FROM_HOME = 17
    SCOUT = 18


@dataclass(frozen=True)
class MultiRaceLabel:
    """One interval's universal and race-conditioned labels with evidence."""

    universal_intent: UniversalIntent
    race_action: RaceMacroAction
    confidence: float
    valid: bool
    source: str
    evidence: tuple[str, ...]
    ability_ids: tuple[int, ...]


def _field(value: Any, name: str, default: Any = 0) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _normalised_names(names: Iterable[str]) -> str:
    return " ".join(str(name).lower().replace("_", "") for name in names)


def _named_label(names: Iterable[str]) -> tuple[UniversalIntent, RaceMacroAction] | None:
    """Map ability aliases from all three races to stable broad labels."""
    key = _normalised_names(names)
    if any(token in key for token in ("trainscv", "trainprobe", "traindrone")):
        return UniversalIntent.TRAIN_WORKER, RaceMacroAction.TRAIN_WORKER
    if any(token in key for token in ("buildsupplydepot", "buildpylon", "trainoverlord", "morphoverlord")):
        return UniversalIntent.ADD_SUPPLY, RaceMacroAction.BUILD_SUPPLY
    if any(token in key for token in ("buildrefinery", "buildassimilator", "buildextractor")):
        return UniversalIntent.GATHER_ECONOMY, RaceMacroAction.BUILD_GAS
    if any(token in key for token in ("buildcommandcenter", "buildnexus", "buildhatchery")):
        return UniversalIntent.EXPAND, RaceMacroAction.EXPAND
    if any(token in key for token in ("buildbarracks", "buildgateway", "buildspawningpool")):
        return UniversalIntent.BUILD_PRODUCTION, RaceMacroAction.BUILD_BASIC_PRODUCTION
    if any(token in key for token in ("buildfactory", "buildstarport", "buildroboticsfacility", "buildstargate", "buildhydraliskden", "buildspire")):
        return UniversalIntent.BUILD_PRODUCTION, RaceMacroAction.BUILD_TECH_PRODUCTION
    if key.startswith("research"):
        return UniversalIntent.ADVANCE_TECH, RaceMacroAction.RESEARCH
    if any(token in key for token in ("buildengineeringbay", "buildarmory", "buildcyberneticscore", "buildtwilightcouncil", "buildforge", "buildfleetbeacon", "buildtemplararchives", "buildinfestationpit", "buildultraliskcavern", "buildbanelingnest", "buildroachwarren", "morphlair", "morphhive", "morphorbital")):
        return UniversalIntent.ADVANCE_TECH, RaceMacroAction.BUILD_TECH
    if any(token in key for token in ("trainmarine", "trainreaper", "trainzealot", "trainzergling", "trainhellion")):
        return UniversalIntent.TRAIN_ARMY, RaceMacroAction.TRAIN_BASIC_GROUND
    if any(token in key for token in ("trainmarauder", "trainsiegetank", "trainthor", "trainstalker", "trainimmortal", "traincolossus", "trainroach", "trainhydralisk", "trainultralisk")):
        return UniversalIntent.TRAIN_ARMY, RaceMacroAction.TRAIN_ARMORED_GROUND
    if any(token in key for token in ("trainviking", "trainliberator", "trainbattlecruiser", "trainphoenix", "trainvoidray", "traincarrier", "trainmutalisk", "traincorruptor", "trainbroodlord")):
        return UniversalIntent.TRAIN_ARMY, RaceMacroAction.TRAIN_AIR
    if any(token in key for token in ("trainmedivac", "trainraven", "trainsentry", "trainhightemplar", "trainobserver", "trainwarprism", "trainqueen", "traininfestor")):
        return UniversalIntent.TRAIN_ARMY, RaceMacroAction.TRAIN_SUPPORT
    if any(token in key for token in ("scan", "rally")):
        return UniversalIntent.SCOUT, RaceMacroAction.SCOUT
    return None


def _combat_label(target: tuple[float, float] | None, own_main: tuple[float, float] | None,
                  enemy_main: tuple[float, float] | None, *, attack: bool) -> tuple[UniversalIntent, RaceMacroAction]:
    """Classify combat by target geometry, not an unreliable harass default."""
    if target is None:
        return (UniversalIntent.ATTACK, RaceMacroAction.COMBAT_ADVANCE) if attack else (UniversalIntent.CONTROL_MAP, RaceMacroAction.SCOUT)
    distance = lambda first, second: hypot(first[0] - second[0], first[1] - second[1])
    if own_main is not None and distance(target, own_main) <= 28.0:
        return UniversalIntent.DEFEND, RaceMacroAction.COMBAT_HOME_DEFENSE
    if enemy_main is not None and distance(target, enemy_main) <= 35.0:
        return UniversalIntent.ATTACK, RaceMacroAction.COMBAT_ADVANCE
    if own_main is not None and distance(target, own_main) < 60.0:
        return UniversalIntent.DEFEND, RaceMacroAction.COMBAT_HOLD
    return (UniversalIntent.HARASS, RaceMacroAction.COMBAT_AWAY_FROM_HOME) if attack else (UniversalIntent.CONTROL_MAP, RaceMacroAction.SCOUT)


class MultiRaceReplayLabeler:
    """Label commands without forcing Protoss/Zerg actions into Terran classes."""

    def __init__(self, resolver: AbilityNameResolver | None = None) -> None:
        self.resolver = resolver or AbilityNameResolver()

    def infer(self, actions: Iterable[Any], previous_state: Mapping[str, Any], current_state: Mapping[str, Any], *,
              own_main: tuple[float, float] | None, enemy_main: tuple[float, float] | None) -> MultiRaceLabel:
        candidates: list[tuple[UniversalIntent, RaceMacroAction, float, str]] = []
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
            named = _named_label(names)
            if named is not None:
                candidates.append((*named, 0.98, next(iter(names), f"ability_{ability_id}")))
                continue
            name_text = _normalised_names(names)
            target = _target(command)
            if "attack" in name_text:
                candidates.append((*_combat_label(target, own_main, enemy_main, attack=True), 0.86, f"attack:{target}"))
            elif any(token in name_text for token in ("move", "patrol")):
                candidates.append((*_combat_label(target, own_main, enemy_main, attack=False), 0.72, f"move:{target}"))
        if candidates:
            # Prefer semantically informative economic/production commands
            # over simultaneous unit-selection or movement commands.
            priority = {UniversalIntent.EXPAND: 100, UniversalIntent.BUILD_PRODUCTION: 95, UniversalIntent.ADVANCE_TECH: 94,
                        UniversalIntent.ADD_SUPPLY: 93, UniversalIntent.GATHER_ECONOMY: 92, UniversalIntent.TRAIN_ARMY: 85,
                        UniversalIntent.TRAIN_WORKER: 84, UniversalIntent.ATTACK: 80, UniversalIntent.HARASS: 79,
                        UniversalIntent.DEFEND: 78, UniversalIntent.SCOUT: 70, UniversalIntent.CONTROL_MAP: 65}
            selected = max(candidates, key=lambda value: priority.get(value[0], 0))
            return MultiRaceLabel(selected[0], selected[1], selected[2], True, "command", (selected[3],), tuple(ability_ids))
        before, after = previous_state.get("buildings", {}), current_state.get("buildings", {})
        if int(after.get("base", 0)) > int(before.get("base", 0)):
            return MultiRaceLabel(UniversalIntent.EXPAND, RaceMacroAction.EXPAND, 0.58, False, "state_delta", ("base_delta",), tuple(ability_ids))
        if int(after.get("supply", 0)) > int(before.get("supply", 0)):
            return MultiRaceLabel(UniversalIntent.ADD_SUPPLY, RaceMacroAction.BUILD_SUPPLY, 0.58, False, "state_delta", ("supply_delta",), tuple(ability_ids))
        if int(current_state.get("worker_count", 0)) > int(previous_state.get("worker_count", 0)):
            return MultiRaceLabel(UniversalIntent.TRAIN_WORKER, RaceMacroAction.TRAIN_WORKER, 0.56, False, "state_delta", ("worker_delta",), tuple(ability_ids))
        return MultiRaceLabel(UniversalIntent.NO_OP, RaceMacroAction.NO_OP, 1.0, False, "none", (), tuple(ability_ids))


def universal_event_vector(intent: UniversalIntent, outcome: str | None = None) -> np.ndarray:
    """Keep the existing seven event targets meaningful for all three races."""
    events = np.zeros(7, dtype=np.float32)
    if intent in {UniversalIntent.GATHER_ECONOMY, UniversalIntent.TRAIN_WORKER, UniversalIntent.EXPAND}:
        events[0] = 1.0
    if intent == UniversalIntent.ADD_SUPPLY:
        events[1] = 1.0
    if intent in {UniversalIntent.BUILD_PRODUCTION, UniversalIntent.TRAIN_ARMY}:
        events[2] = 1.0
    if intent == UniversalIntent.ADVANCE_TECH:
        events[3] = 1.0
    if intent == UniversalIntent.SCOUT:
        events[4] = 1.0
    if intent in {UniversalIntent.DEFEND, UniversalIntent.HARASS, UniversalIntent.ATTACK, UniversalIntent.RETREAT}:
        events[5] = 1.0
    if outcome is not None:
        events[6] = 1.0
    return events
