"""Safe macro-level real-time agent usable with mock or real SC2 backends."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from sc2wmrl.controllers.macro_skill_executor import MacroSkillExecutor, SkillContext, SkillStatus
from sc2wmrl.deployment.action_dispatcher import ActionDispatcher
from sc2wmrl.deployment.checkpoint_loader import CheckpointLoader
from sc2wmrl.deployment.decision_recorder import DecisionRecord, DecisionRecorder
from sc2wmrl.deployment.game_overlay import GameOverlay
from sc2wmrl.deployment.inference_scheduler import InferenceScheduler
from sc2wmrl.deployment.replay_manager import ReplayManager
from sc2wmrl.deployment.safety_fallback import SafetyFallbackPolicy
from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.envs.feature_extractor import FeatureExtractor


class RealtimeBackend(Protocol):
    """Adapter contract implemented by a BotAI wrapper or deterministic mock."""
    def structured_state(self) -> dict[str, Any]: ...
    def action_mask(self) -> np.ndarray: ...
    def game_loop(self) -> int: ...
    def urgent_event(self) -> bool: ...
    def skill_finished(self) -> bool: ...


@dataclass(frozen=True)
class RealtimeAgentConfig:
    checkpoint_path: Path
    deterministic: bool = True
    macro_interval_game_loops: int = 32
    device: str = "auto"
    enable_world_model: bool = False
    enable_overlay: bool = False
    record_decisions: bool = True
    fallback_on_error: bool = True
    maximum_inference_latency_ms: float = 100.0


@dataclass
class AgentRuntimeState:
    game_loop: int = 0
    episode_step: int = 0
    current_macro_action: int = int(MacroAction.NO_OP)
    current_skill_start_loop: int = 0
    recurrent_state: object | None = None
    world_model_state: object | None = None
    opponent_context: object | None = None
    previous_observation: object | None = None
    previous_action: int | None = None
    last_inference_duration_ms: float = 0.0
    fallback_active: bool = False


class RealtimeRLAgent:
    """Loads a real checkpoint, selects masked actions periodically, and survives isolated failures."""
    def __init__(self, config: RealtimeAgentConfig, backend: RealtimeBackend, executor: MacroSkillExecutor,
                 output_dir: str | Path, overlay: GameOverlay | None = None) -> None:
        self.config, self.backend, self.extractor = config, backend, FeatureExtractor()
        self.dispatcher, self.scheduler, self.fallback, self.overlay = ActionDispatcher(executor), InferenceScheduler(config.macro_interval_game_loops), SafetyFallbackPolicy(), overlay or GameOverlay(None, False)
        self.output_dir = Path(output_dir); self.state = AgentRuntimeState(); self.loaded = None; self.recorder: DecisionRecorder | None = None; self.game_id = ""

    async def on_start(self) -> None:
        """Strictly load checkpoint and initialize per-game latent/recurrent state."""
        self.loaded = CheckpointLoader().load_ppo(self.config.checkpoint_path, expected_observation_dim=self.extractor.dimension, device=self.config.device)
        self.loaded.actor.network.eval(); self.loaded.actor.policy_head.eval(); self.loaded.actor.value_head.eval()
        self.state = AgentRuntimeState(); self.scheduler.reset(); self.game_id = uuid.uuid4().hex
        if self.config.record_decisions: self.recorder = DecisionRecorder(self.output_dir / self.game_id, self.game_id)

    async def on_step(self, iteration: int) -> None:
        """Run an early urgent or periodic macro decision while safely containing failures."""
        if self.loaded is None: raise RuntimeError("on_start must complete before on_step")
        game_loop = self.backend.game_loop(); self.state.game_loop = game_loop
        skill_status = await self.dispatcher.status(self._context())
        if not self.scheduler.should_infer(game_loop, urgent_event=self.backend.urgent_event(), skill_finished=skill_status in {SkillStatus.SUCCEEDED, SkillStatus.FAILED, SkillStatus.TIMED_OUT}): return
        observation, mask = self.extractor.extract(self.backend.structured_state()), np.asarray(self.backend.action_mask(), dtype=np.bool_)
        if mask.shape != (len(MacroAction),) or not mask.any():
            mask = np.zeros(len(MacroAction), dtype=np.bool_); mask[int(MacroAction.NO_OP)] = True
        started = time.perf_counter(); fallback_used = False; failure: str | None = None
        try:
            if not np.isfinite(observation).all() or not mask.any(): raise ValueError("invalid observation or empty action mask")
            action, details = select_macro_action(self.loaded.actor, observation, mask, deterministic=self.config.deterministic)
            elapsed = (time.perf_counter() - started) * 1000
            if elapsed > self.config.maximum_inference_latency_ms: raise TimeoutError("inference latency exceeded configured limit")
        except Exception as exc:
            if not self.config.fallback_on_error: raise
            action = self.fallback.select_action(observation, mask, {"error": str(exc)}); details = {"probability": 0.0, "top_actions": [], "value": 0.0, "entropy": 0.0}; elapsed = (time.perf_counter() - started) * 1000; fallback_used = True; failure = f"{type(exc).__name__}: {exc}"
        context = self._context(mask=mask); handle = await self.dispatcher.dispatch(action, context)
        self.scheduler.mark_inference(game_loop); self.state.episode_step += 1; self.state.current_macro_action = action; self.state.current_skill_start_loop = game_loop; self.state.previous_observation = observation; self.state.previous_action = action; self.state.last_inference_duration_ms = elapsed; self.state.fallback_active = fallback_used
        if self.recorder is not None:
            self.recorder.record(DecisionRecord(self.game_id, game_loop, observation.tolist(), mask.astype(int).tolist(), action, MacroAction(action).name,
                details["probability"], details["top_actions"], details["value"], details["entropy"], [], None, elapsed, failure or ("SUCCEEDED" if handle.result.succeeded else "FAILED"), fallback_used))
        self.overlay.render({"Action": MacroAction(action).name, "Value": f"{details['value']:.3f}", "Latency ms": f"{elapsed:.1f}", "Fallback": fallback_used})

    async def on_end(self, result: str) -> None:
        """Finalize decision artifacts and attempt replay saving without false success claims."""
        if self.recorder is not None:
            game_dir = self.output_dir / self.game_id; replay_path, replay_error = ReplayManager().save(self.backend if hasattr(self.backend, "save_replay") else None, game_dir)
            self.recorder.close({"game_id": self.game_id, "result": result, "decision_count": self.state.episode_step, "replay_path": str(replay_path) if replay_path else None, "replay_error": replay_error})

    def _context(self, mask: np.ndarray | None = None) -> SkillContext:
        """Build validated executor context from current backend state."""
        current_mask = np.asarray(self.backend.action_mask() if mask is None else mask, dtype=np.bool_)
        return SkillContext(self.backend.structured_state(), current_mask.tolist(), self.backend.game_loop())


def select_macro_action(actor: object, observation: np.ndarray, action_mask: np.ndarray, recurrent_state: object | None = None, *, deterministic: bool) -> tuple[int, dict[str, Any]]:
    """Return a legal PPO action with probability, top-k, value, and entropy diagnostics."""
    del recurrent_state
    torch_actor = actor; device = torch_actor.device
    with torch.no_grad():
        distribution, value = torch_actor._distribution(torch.as_tensor(observation[None], device=device), torch.as_tensor(action_mask[None], device=device))
        probabilities = distribution.probs[0]; action_tensor = torch.argmax(probabilities) if deterministic else distribution.sample()
        action = int(action_tensor.item()); top_values, top_indices = torch.topk(probabilities, min(3, len(MacroAction)))
        return action, {"probability": float(probabilities[action].item()), "top_actions": [{"action": int(index.item()), "name": MacroAction(int(index.item())).name, "probability": float(prob.item())} for prob, index in zip(top_values, top_indices)], "value": float(value[0].item()), "entropy": float(distribution.entropy()[0].item())}
