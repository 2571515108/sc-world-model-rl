"""Dispatch selected macro actions through a lifecycle-aware skill executor."""

from __future__ import annotations

from sc2wmrl.controllers.macro_skill_executor import MacroSkillExecutor, SkillContext, SkillHandle, SkillStatus
from sc2wmrl.envs.base_macro_env import MacroAction


class ActionDispatcher:
    """Converts one selected legal action into a tracked skill execution."""
    def __init__(self, executor: MacroSkillExecutor) -> None:
        self.executor = executor; self.handle: SkillHandle | None = None

    async def dispatch(self, action: int, context: SkillContext) -> SkillHandle:
        """Reject illegal actions before delegating execution to the skill executor."""
        macro_action = MacroAction(action)
        if not context.action_mask[action]: raise ValueError("dispatcher received illegal action")
        self.handle = await self.executor.start_skill(macro_action, context); return self.handle

    async def status(self, context: SkillContext) -> SkillStatus | None:
        """Return the active skill status when a skill exists."""
        return None if self.handle is None else await self.executor.update_skill(self.handle, context)
