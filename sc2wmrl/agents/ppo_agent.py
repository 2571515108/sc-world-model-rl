"""Mask-aware discrete PPO actor-critic for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sc2wmrl.envs.base_macro_env import ACTION_COUNT
from sc2wmrl.utils.device import resolve_device


def _torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PPO requires PyTorch. Install dependencies with: pip install -r requirements.txt") from exc


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters; all are serializable in the YAML config/checkpoint."""

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64


class PPOAgent:
    """Neural actor-critic that applies action masks before categorical sampling."""

    def __init__(self, observation_dim: int, config: PPOConfig | None = None, *, device: str = "auto") -> None:
        torch = _torch()
        if observation_dim <= 0:
            raise ValueError("observation dimension must be positive")
        self.observation_dim, self.config, self.device = observation_dim, config or PPOConfig(), resolve_device(device)
        nn = torch.nn
        self.network = nn.Sequential(nn.Linear(observation_dim, self.config.hidden_dim), nn.Tanh(),
                                     nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.Tanh()).to(self.device)
        self.policy_head = nn.Linear(self.config.hidden_dim, ACTION_COUNT).to(self.device)
        self.value_head = nn.Linear(self.config.hidden_dim, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def parameters(self) -> list[Any]:
        """Return all trainable modules' parameters for optimizer/checkpoint use."""
        return list(self.network.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters())

    def _distribution(self, observations: Any, masks: Any) -> tuple[Any, Any]:
        torch = _torch()
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("invalid observation batch shape")
        if masks.shape != (observations.shape[0], ACTION_COUNT) or not torch.all(masks.any(dim=-1)):
            raise ValueError("invalid action-mask batch")
        hidden = self.network(observations)
        logits = self.policy_head(hidden).masked_fill(~masks.bool(), float("-inf"))
        if not torch.isfinite(logits.masked_select(masks.bool())).all():
            raise ValueError("policy produced non-finite legal logits")
        return torch.distributions.Categorical(logits=logits), self.value_head(hidden).squeeze(-1)

    def act(self, observation: np.ndarray, action_mask: np.ndarray, *, deterministic: bool = False) -> tuple[int, float, float]:
        """Select a legal action and return action, log-probability, and value."""
        torch = _torch()
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=np.bool_)
        if observation.shape != (self.observation_dim,) or mask.shape != (ACTION_COUNT,) or not mask.any() or not np.isfinite(observation).all():
            raise ValueError("invalid PPO action input")
        actions, log_probs, values = self.act_batch(observation[None], mask[None], deterministic=deterministic)
        return int(actions[0]), float(log_probs[0]), float(values[0])

    def act_batch(self, observations: np.ndarray, action_masks: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer many mask-respecting actions in one device call.

        Vector collectors use this method to avoid thousands of batch-one CUDA
        launches; the scalar :meth:`act` API remains fully compatible.
        """
        torch = _torch(); observations = np.asarray(observations, dtype=np.float32); masks = np.asarray(action_masks, dtype=np.bool_)
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim or masks.shape != (len(observations), ACTION_COUNT) or not masks.any(-1).all() or not np.isfinite(observations).all():
            raise ValueError("invalid PPO batch action input")
        with torch.inference_mode():
            distribution, values = self._distribution(torch.as_tensor(observations, device=self.device), torch.as_tensor(masks, device=self.device))
            actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(actions)
        return actions.cpu().numpy().astype(np.int64), log_probs.cpu().numpy().astype(np.float32), values.cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Run clipped PPO updates on validated rollout data."""
        torch = _torch()
        required = {"observations", "action_masks", "actions", "old_log_probs", "advantages", "returns"}
        if set(batch) != required:
            raise ValueError(f"PPO batch keys must be exactly {sorted(required)}")
        size = len(batch["actions"])
        if size == 0:
            raise ValueError("cannot update PPO with an empty batch")
        tensors = {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}
        tensors["observations"] = tensors["observations"].float(); tensors["action_masks"] = tensors["action_masks"].bool()
        tensors["actions"] = tensors["actions"].long(); tensors["old_log_probs"] = tensors["old_log_probs"].float()
        tensors["advantages"] = tensors["advantages"].float(); tensors["returns"] = tensors["returns"].float()
        if tensors["observations"].shape != (size, self.observation_dim) or tensors["action_masks"].shape != (size, ACTION_COUNT):
            raise ValueError("PPO batch contains incompatible shapes")
        if not all(torch.isfinite(tensors[key]).all() for key in ("observations", "old_log_probs", "advantages", "returns")):
            raise ValueError("PPO batch contains non-finite values")
        advantages = (tensors["advantages"] - tensors["advantages"].mean()) / (tensors["advantages"].std(unbiased=False) + 1e-8)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        updates = 0
        for _ in range(self.config.epochs):
            for indices in torch.randperm(size, device=self.device).split(min(self.config.minibatch_size, size)):
                distribution, values = self._distribution(tensors["observations"][indices], tensors["action_masks"][indices])
                new_log_probs = distribution.log_prob(tensors["actions"][indices])
                ratio = (new_log_probs - tensors["old_log_probs"][indices]).exp()
                surrogate_a = ratio * advantages[indices]
                surrogate_b = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages[indices]
                policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
                value_loss = torch.nn.functional.mse_loss(values, tensors["returns"][indices])
                entropy = distribution.entropy().mean()
                loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss is non-finite")
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm); self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item()); stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item()); stats["approx_kl"] += float((tensors["old_log_probs"][indices] - new_log_probs).mean().item()); updates += 1
        return {key: value / updates for key, value in stats.items()}

    def save(self, path: str | Path, *, training_state: dict[str, Any] | None = None) -> None:
        """Save model, optimizer, architecture, and caller-owned resume metadata."""
        torch = _torch(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "observation_dim": self.observation_dim, "config": asdict(self.config),
                    "network": self.network.state_dict(), "policy_head": self.policy_head.state_dict(), "value_head": self.value_head.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "training_state": training_state or {}}, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> tuple["PPOAgent", dict[str, Any]]:
        """Load a checkpoint and return agent plus exact training-resume metadata."""
        torch = _torch(); resolved_device = resolve_device(device); payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported PPO checkpoint version")
        agent = cls(int(payload["observation_dim"]), PPOConfig(**payload["config"]), device=str(resolved_device))
        agent.network.load_state_dict(payload["network"]); agent.policy_head.load_state_dict(payload["policy_head"]); agent.value_head.load_state_dict(payload["value_head"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent, dict(payload.get("training_state", {}))
