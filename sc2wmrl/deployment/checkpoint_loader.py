"""Strict loading and compatibility validation for deployable PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.utils.device import resolve_device


@dataclass
class LoadedAgent:
    """Validated actor/critic bundle and immutable deployment metadata."""
    actor: PPOAgent
    critic: object | None
    world_model: object | None
    opponent_encoder: object | None
    observation_normalizer: object | None
    action_names: list[str]
    metadata: dict[str, Any]


def training_config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for checkpoint/config compatibility checks."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CheckpointLoader:
    """Rejects malformed or incompatible checkpoints rather than silently adapting them."""
    def load_ppo(self, path: str | Path, *, expected_observation_dim: int, device: str = "auto",
                 expected_config_hash: str | None = None) -> LoadedAgent:
        """Load a PPO checkpoint and validate version, dimensions, action schema, and hash."""
        resolved = resolve_device(device); payload = torch.load(Path(path), map_location=resolved, weights_only=False)
        if payload.get("format_version") != 1: raise ValueError("unsupported checkpoint version")
        if int(payload.get("observation_dim", -1)) != expected_observation_dim: raise ValueError("checkpoint observation dimension mismatch")
        names = [action.name for action in MacroAction]
        metadata = dict(payload.get("training_state", {})); saved_names = metadata.get("macro_action_names", names)
        if saved_names != names: raise ValueError("checkpoint macro action definitions mismatch")
        if expected_config_hash is not None and metadata.get("training_config_hash") != expected_config_hash:
            raise ValueError("checkpoint training configuration hash mismatch")
        agent, state = PPOAgent.load(path, device=str(resolved)); metadata.update(state)
        return LoadedAgent(agent, agent.value_head, None, None, metadata.get("normalization_statistics"), names, metadata)
