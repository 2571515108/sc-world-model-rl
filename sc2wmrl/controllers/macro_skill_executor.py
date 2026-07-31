"""Backend-neutral macro-skill boundary used by real SC2 integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from sc2wmrl.envs.base_macro_env import MacroAction


@dataclass(frozen=True)
class SkillContext:
    """State supplied by the real bot controller at a macro decision boundary."""

    raw_observation: dict[str, Any]
    action_mask: list[bool]
    game_loop: int


@dataclass(frozen=True)
class SkillResult:
    """Auditable outcome of a macro action submitted to a concrete controller."""

    succeeded: bool
    commands_issued: int
    events: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SkillStatus(str, Enum):
    """Lifecycle states for an asynchronously running macro skill."""
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    TIMED_OUT = "TIMED_OUT"


class SkillFailureReason(str, Enum):
    """Structured failure causes exposed to real-time recovery logic."""
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    MISSING_PREREQUISITE = "MISSING_PREREQUISITE"
    NO_AVAILABLE_UNIT = "NO_AVAILABLE_UNIT"
    INVALID_TARGET = "INVALID_TARGET"
    PATHING_FAILED = "PATHING_FAILED"
    TARGET_DESTROYED = "TARGET_DESTROYED"
    TIMEOUT = "TIMEOUT"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"


@dataclass(frozen=True)
class SkillHandle:
    """Opaque live-skill identifier and initial execution result."""
    identifier: str
    action: MacroAction
    started_loop: int
    result: SkillResult


class MacroCommandSink(Protocol):
    """Minimal command surface an existing python-sc2/Ares controller can expose."""

    async def execute_macro(self, action: MacroAction, duration: int, context: SkillContext) -> SkillResult:
        """Issue valid unit-level commands for the requested macro skill."""


class MacroSkillExecutor:
    """Validates macro requests before delegating micro execution to a controller."""

    def __init__(self, command_sink: MacroCommandSink) -> None:
        self.command_sink = command_sink
        self._handles: dict[str, SkillHandle] = {}

    async def execute(self, action: MacroAction, duration: int, context: SkillContext) -> SkillResult:
        """Execute an allowed macro action for a positive number of game loops."""
        if duration <= 0:
            raise ValueError("macro duration must be positive")
        if len(context.action_mask) != len(MacroAction) or not context.action_mask[int(action)]:
            raise ValueError(f"skill executor received illegal action {action.name}")
        result = await self.command_sink.execute_macro(action, duration, context)
        if result.commands_issued < 0:
            raise ValueError("controller returned a negative command count")
        return result

    async def start_skill(self, action: MacroAction, context: SkillContext) -> SkillHandle:
        """Start a macro skill and return a handle for status polling or cancellation."""
        result = await self.execute(action, 1, context); identifier = f"{context.game_loop}:{int(action)}:{len(self._handles)}"
        handle = SkillHandle(identifier, action, context.game_loop, result); self._handles[identifier] = handle; return handle

    async def update_skill(self, handle: SkillHandle, context: SkillContext) -> SkillStatus:
        """Return current status; sinks may expose richer polling via diagnostics."""
        if handle.identifier not in self._handles: return SkillStatus.INTERRUPTED
        if context.game_loop <= handle.started_loop: return SkillStatus.RUNNING
        return SkillStatus.SUCCEEDED if handle.result.succeeded else SkillStatus.FAILED

    async def cancel_skill(self, handle: SkillHandle, reason: str) -> None:
        """Cancel a tracked skill and retain the explicit cancellation reason."""
        if handle.identifier in self._handles:
            result = self._handles[handle.identifier].result
            self._handles[handle.identifier] = SkillHandle(handle.identifier, handle.action, handle.started_loop,
                                                            SkillResult(False, result.commands_issued, result.events, result.diagnostics | {"cancel_reason": reason}))
