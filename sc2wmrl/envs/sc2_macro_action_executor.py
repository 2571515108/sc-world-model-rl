"""python-sc2 implementation of the shared Terran macro-action vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base_macro_env import MacroAction


@dataclass
class MacroExecutionResult:
    """Structured command result; failure is never silently reported as success."""

    success: bool
    action: str
    issued_commands: int = 0
    failure_reason: str | None = None
    pending: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"success": self.success, "action": self.action, "issued_commands": self.issued_commands, "failure_reason": self.failure_reason, "pending": self.pending, **self.details}


class SC2MacroActionExecutor:
    """Issue real BotAI commands once per macro interval.

    It imports python-sc2 only when a live game is requested, leaving synthetic
    tests and model-only installation independent from the SC2 client package.
    """

    def __init__(self) -> None:
        self.pending_actions: dict[str, bool] = {}

    def clear_completed(self, bot: Any) -> None:
        """Clear optimistic pending flags once a subsequent observation arrives."""
        del bot  # Completion is primarily represented by python-sc2 unit state.
        self.pending_actions.clear()

    async def execute(self, action: MacroAction, bot: Any) -> dict[str, Any]:
        """Translate one macro action into legal python-sc2 commands."""
        if self.pending_actions.get(action.name):
            return MacroExecutionResult(False, action.name, failure_reason="temporarily_unavailable", pending=True).as_dict()
        try:
            from sc2.ids.ability_id import AbilityId
            from sc2.ids.unit_typeid import UnitTypeId
        except ImportError as exc:
            raise RuntimeError("Real SC2 execution requires optional dependency burnysc2.") from exc
        try:
            result = await self._execute(action, bot, UnitTypeId, AbilityId)
        except Exception as exc:  # The live client may reject a race/patch-specific ability.
            return MacroExecutionResult(False, action.name, failure_reason="command_failed", details={"exception": type(exc).__name__}).as_dict()
        if result.success and result.pending:
            self.pending_actions[action.name] = True
        return result.as_dict()

    async def _execute(self, action: MacroAction, bot: Any, unit: Any, ability: Any) -> MacroExecutionResult:
        def first_idle(name: Any) -> Any | None:
            candidates = bot.structures(name).idle
            return candidates.first if candidates.exists else None

        async def train(structure_type: Any, trained_type: Any) -> MacroExecutionResult:
            structure = first_idle(structure_type)
            if structure is None:
                return MacroExecutionResult(False, action.name, failure_reason="temporarily_unavailable")
            await bot.do(structure.train(trained_type))
            return MacroExecutionResult(True, action.name, 1, pending=True)

        async def build(structure_type: Any, *, near: Any | None = None, geyser: bool = False) -> MacroExecutionResult:
            worker = bot.select_build_worker(near or bot.start_location)
            if worker is None:
                return MacroExecutionResult(False, action.name, failure_reason="no_builder")
            if geyser:
                geysers = bot.vespene_geyser.closer_than(12, bot.townhalls.random)
                if not geysers.exists:
                    return MacroExecutionResult(False, action.name, failure_reason="no_geyser")
                await bot.do(worker.build_gas(geysers.first))
            else:
                ok = await bot.build(structure_type, near=near or bot.start_location)
                if not ok:
                    return MacroExecutionResult(False, action.name, failure_reason="command_failed")
            return MacroExecutionResult(True, action.name, 1, pending=True)

        if action == MacroAction.NO_OP:
            return MacroExecutionResult(True, action.name)
        if action == MacroAction.TRAIN_WORKERS:
            return await train(unit.COMMANDCENTER, unit.SCV)
        if action == MacroAction.BUILD_SUPPLY:
            return await build(unit.SUPPLYDEPOT, near=bot.start_location)
        if action == MacroAction.BUILD_BARRACKS:
            return await build(unit.BARRACKS, near=bot.start_location)
        if action == MacroAction.BUILD_REFINERY:
            return await build(unit.REFINERY, geyser=True)
        if action == MacroAction.BUILD_FACTORY:
            return await build(unit.FACTORY, near=bot.start_location)
        if action == MacroAction.BUILD_STARPORT:
            return await build(unit.STARPORT, near=bot.start_location)
        if action == MacroAction.EXPAND:
            location = await bot.get_next_expansion()
            return await build(unit.COMMANDCENTER, near=location)
        if action == MacroAction.TRAIN_BASIC_ARMY:
            return await train(unit.BARRACKS, unit.MARINE)
        if action == MacroAction.TRAIN_ANTI_GROUND:
            return await train(unit.BARRACKS, unit.MARAUDER)
        if action == MacroAction.TRAIN_ANTI_AIR:
            return await train(unit.STARPORT, unit.VIKINGFIGHTER)
        if action == MacroAction.RESEARCH_UPGRADE:
            structure = first_idle(unit.ENGINEERINGBAY)
            if structure is None:
                return MacroExecutionResult(False, action.name, failure_reason="temporarily_unavailable")
            await bot.do(structure(ability.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1))
            return MacroExecutionResult(True, action.name, 1, pending=True)
        army = bot.units.filter(lambda item: not item.is_worker and item.can_attack)
        async def command_army(method: str, target: Any) -> None:
            commands = [getattr(item, method)(target) for item in army]
            if not commands:
                raise RuntimeError("no army commands available")
            await bot.do_actions(commands)
        if action in (MacroAction.SCOUT_ENEMY_MAIN, MacroAction.SCOUT_EXPANSION):
            scout = bot.workers.random if bot.workers.exists else (army.random if army.exists else None)
            if scout is None or not bot.enemy_start_locations:
                return MacroExecutionResult(False, action.name, failure_reason="no_scout_or_target")
            await bot.do(scout.move(bot.enemy_start_locations[0]))
            return MacroExecutionResult(True, action.name, 1, pending=True)
        if not army.exists:
            return MacroExecutionResult(False, action.name, failure_reason="no_army")
        if action in (MacroAction.DEFEND_MAIN, MacroAction.RETREAT):
            await command_army("move", bot.start_location)
        elif action == MacroAction.DEFEND_NATURAL:
            target = bot.townhalls.closest_to(bot.start_location).position if bot.townhalls.exists else bot.start_location
            await command_army("move", target)
        elif action in (MacroAction.HARASS, MacroAction.ATTACK_ENEMY_NATURAL, MacroAction.ATTACK_ENEMY_MAIN):
            if not bot.enemy_start_locations:
                return MacroExecutionResult(False, action.name, failure_reason="no_enemy_target")
            await command_army("attack", bot.enemy_start_locations[0])
        else:
            return MacroExecutionResult(False, action.name, failure_reason="unsupported_action")
        return MacroExecutionResult(True, action.name, 1, pending=True)
