"""Actor-only behavior cloning from high-confidence expert replay labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent, _torch
from sc2wmrl.envs.base_macro_env import ACTION_COUNT, MacroAction
from sc2wmrl.replay.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class BehaviorCloningConfig:
    """Supervised actor-pretraining hyperparameters."""

    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 30
    validation_fraction: float = 0.1
    minimum_confidence: float = 0.70
    include_no_op: bool = False
    class_balance_power: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 1.0
    seed: int = 7


@dataclass(frozen=True)
class ExpertBatch:
    """Numpy tensors selected from conversion-approved expert labels."""

    observations: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    confidences: np.ndarray
    episode_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, indexes: np.ndarray) -> "ExpertBatch":
        return ExpertBatch(*(value[indexes] for value in (
            self.observations, self.action_masks, self.actions, self.confidences, self.episode_ids
        )))


def expert_batch_from_replay(replay: ReplayBuffer, config: BehaviorCloningConfig) -> ExpertBatch:
    """Select high-confidence macro labels while keeping all data for the world model."""
    selected = []
    for item in replay.transitions():
        confidence = float(item.info.get("label_confidence", 0.0))
        if not bool(item.info.get("expert_label", False)) or confidence < config.minimum_confidence:
            continue
        if not config.include_no_op and int(item.action) == int(MacroAction.NO_OP):
            continue
        selected.append((item.observation, item.action_mask, int(item.action), confidence, int(item.episode_id)))
    if not selected:
        raise ValueError("no conversion-approved expert labels satisfy the behavior-cloning filters")
    observations, masks, actions, confidences, episodes = zip(*selected)
    return ExpertBatch(
        np.stack(observations).astype(np.float32), np.stack(masks).astype(np.bool_), np.asarray(actions, dtype=np.int64),
        np.asarray(confidences, dtype=np.float32), np.asarray(episodes, dtype=np.int64),
    )


def split_expert_batch(batch: ExpertBatch, validation_fraction: float) -> tuple[ExpertBatch, ExpertBatch | None]:
    """Split by whole replay episode, never by adjacent correlated intervals."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    episode_ids = np.unique(batch.episode_ids)
    if validation_fraction == 0.0 or len(episode_ids) < 2:
        return batch, None
    validation_count = max(1, int(round(len(episode_ids) * validation_fraction)))
    validation_ids = set(episode_ids[-validation_count:].tolist())
    validation = np.asarray([episode in validation_ids for episode in batch.episode_ids], dtype=np.bool_)
    if validation.all() or not validation.any():
        return batch, None
    return batch.subset(np.flatnonzero(~validation)), batch.subset(np.flatnonzero(validation))


class BehaviorCloningTrainer:
    """Update only PPO's shared actor trunk and policy head, never its value head."""

    def __init__(self, agent: PPOAgent, config: BehaviorCloningConfig) -> None:
        torch = _torch()
        self.agent, self.config = agent, config
        self.rng = np.random.default_rng(config.seed)
        self.optimizer = torch.optim.Adam(
            list(agent.network.parameters()) + list(agent.policy_head.parameters()), lr=config.learning_rate
        )

    def _class_weights(self, actions: np.ndarray) -> np.ndarray:
        counts = np.bincount(actions, minlength=ACTION_COUNT).astype(np.float64)
        weights = np.zeros(ACTION_COUNT, dtype=np.float32)
        present = counts > 0
        weights[present] = np.power(counts[present], -self.config.class_balance_power)
        if present.any():
            weights[present] /= weights[present].mean()
        return weights

    def train_epoch(self, batch: ExpertBatch) -> dict[str, float]:
        """Run one class-balanced, confidence-weighted actor update pass."""
        torch = _torch()
        self.agent.network.train(); self.agent.policy_head.train()
        weights = torch.as_tensor(self._class_weights(batch.actions), device=self.agent.device)
        order = self.rng.permutation(len(batch))
        totals = {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_entropy": 0.0}
        seen = 0
        for start in range(0, len(order), self.config.batch_size):
            indexes = order[start:start + self.config.batch_size]
            observations = torch.as_tensor(batch.observations[indexes], device=self.agent.device)
            masks = torch.as_tensor(batch.action_masks[indexes], device=self.agent.device)
            actions = torch.as_tensor(batch.actions[indexes], device=self.agent.device)
            confidence = torch.as_tensor(batch.confidences[indexes], device=self.agent.device)
            distribution, _ = self.agent._distribution(observations, masks)
            per_item = -distribution.log_prob(actions)
            sample_weights = confidence * weights[actions]
            policy_loss = (per_item * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            entropy = distribution.entropy().mean()
            loss = policy_loss - self.config.entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("behavior-cloning loss is non-finite")
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.agent.network.parameters()) + list(self.agent.policy_head.parameters()), self.config.max_grad_norm)
            self.optimizer.step()
            size = len(indexes); seen += size
            totals["bc_loss"] += float(policy_loss.item()) * size
            totals["bc_accuracy"] += float((distribution.probs.argmax(-1) == actions).float().mean().item()) * size
            totals["bc_entropy"] += float(entropy.item()) * size
        return {name: value / seen for name, value in totals.items()}

    def evaluate(self, batch: ExpertBatch | None) -> dict[str, float]:
        """Evaluate masked top-1/top-3 accuracy without mutating the actor."""
        if batch is None or not len(batch):
            return {}
        torch = _torch(); self.agent.network.eval(); self.agent.policy_head.eval()
        loss_sum = 0.0; correct = 0; top3 = 0; count = 0
        with torch.inference_mode():
            for start in range(0, len(batch), self.config.batch_size):
                indexes = slice(start, min(len(batch), start + self.config.batch_size))
                observations = torch.as_tensor(batch.observations[indexes], device=self.agent.device)
                masks = torch.as_tensor(batch.action_masks[indexes], device=self.agent.device)
                actions = torch.as_tensor(batch.actions[indexes], device=self.agent.device)
                distribution, _ = self.agent._distribution(observations, masks)
                probabilities = distribution.probs
                loss_sum += float((-distribution.log_prob(actions)).sum().item())
                correct += int((probabilities.argmax(-1) == actions).sum().item())
                top3 += int((probabilities.topk(min(3, ACTION_COUNT), dim=-1).indices == actions[:, None]).any(-1).sum().item())
                count += len(actions)
        return {"validation_bc_loss": loss_sum / count, "validation_bc_accuracy": correct / count,
                "validation_bc_top3_accuracy": top3 / count}

    def fit(self, train: ExpertBatch, validation: ExpertBatch | None = None) -> list[dict[str, float]]:
        """Train for the configured number of epochs and return epoch metrics."""
        history = []
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(train); metrics.update(self.evaluate(validation)); metrics["epoch"] = float(epoch + 1)
            history.append(metrics)
        return history
