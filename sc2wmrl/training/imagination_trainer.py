"""Uncertainty-gated imagined actor-critic rollout and optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.critic import LatentCritic
from sc2wmrl.models.rssm import RSSMState
from sc2wmrl.models.world_model import WorldModel


@dataclass(frozen=True)
class ImaginationConfig:
    horizon: int = 15
    gamma: float = 0.99
    lambda_: float = 0.95
    uncertainty_threshold: float = 1.0
    pessimism_scale: float = 0.1
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 100.0


class ImaginationTrainer:
    """Generates only bounded, finite, uncertainty-gated imagined trajectories."""

    def __init__(self, world_model: WorldModel, actor: LatentActor, critic: LatentCritic, config: ImaginationConfig | None = None) -> None:
        device = next(world_model.parameters()).device
        self.world_model, self.actor, self.critic, self.config = world_model, actor.to(device), critic.to(device), config or ImaginationConfig()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    def rollout(self, observations: Tensor, actions: Tensor, action_masks: Tensor, opponent_ids: Tensor | None = None) -> dict[str, Tensor]:
        """Start at posterior states and stop each trajectory on uncertainty/terminal anomalies."""
        self.world_model.eval()
        with torch.no_grad():
            _, posts, context, _ = self.world_model.observe(observations, actions, opponent_ids, sample=False)
        state: RSSMState = posts[-1]; mask = action_masks[:, -1].bool()
        if not torch.all(mask.any(-1)): raise ValueError("imagination start contains empty masks")
        features: list[Tensor] = []; sampled_actions: list[Tensor] = []; log_probs: list[Tensor] = []; rewards: list[Tensor] = []; continues: list[Tensor] = []; alive: list[Tensor] = []; entropies: list[Tensor] = []
        active = torch.ones(observations.shape[0], dtype=torch.bool, device=observations.device)
        for _ in range(self.config.horizon):
            feature = self.world_model.rssm.feature(state).detach(); distribution = self.actor.distribution(feature, mask)
            action = distribution.sample(); next_state, reward, continuation, uncertainty, _ = self.world_model.imagine_step(state, action, context)
            valid = torch.isfinite(reward) & torch.isfinite(uncertainty) & torch.isfinite(continuation) & (uncertainty <= self.config.uncertainty_threshold) & (continuation > 0.5) & mask.any(-1)
            active = active & valid
            features.append(feature); sampled_actions.append(action); log_probs.append(distribution.log_prob(action)); entropies.append(distribution.entropy())
            rewards.append((reward - self.config.pessimism_scale * uncertainty) * active.float()); continues.append(continuation * active.float()); alive.append(active.float())
            state = next_state
            if not active.any(): break
        if not rewards: raise RuntimeError("imagination produced no rollout steps")
        return {"features": torch.stack(features, 1), "actions": torch.stack(sampled_actions, 1), "log_probs": torch.stack(log_probs, 1),
                "entropies": torch.stack(entropies, 1), "rewards": torch.stack(rewards, 1), "continues": torch.stack(continues, 1), "alive": torch.stack(alive, 1)}

    def update(self, rollout: dict[str, Tensor]) -> dict[str, float]:
        """Optimize policy-gradient actor and lambda-return critic without world-model gradients."""
        features, rewards, continues = rollout["features"], rollout["rewards"], rollout["continues"]
        values = self.critic(features); bootstrap = torch.zeros_like(values[:, 0]); returns: list[Tensor] = []
        target = bootstrap
        for time in range(rewards.shape[1] - 1, -1, -1):
            next_value = bootstrap if time == rewards.shape[1] - 1 else values[:, time + 1].detach()
            target = rewards[:, time] + self.config.gamma * continues[:, time] * ((1 - self.config.lambda_) * next_value + self.config.lambda_ * target)
            returns.append(target)
        returns_tensor = torch.stack(list(reversed(returns)), 1).detach(); advantages = returns_tensor - values.detach()
        actor_loss = -(rollout["log_probs"] * advantages * rollout["alive"]).sum() / rollout["alive"].sum().clamp_min(1)
        actor_loss -= self.config.entropy_coefficient * (rollout["entropies"] * rollout["alive"]).sum() / rollout["alive"].sum().clamp_min(1)
        critic_loss = F.mse_loss(values, returns_tensor)
        if not torch.isfinite(actor_loss + critic_loss): raise FloatingPointError("imagined actor-critic loss is non-finite")
        self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm); self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm); self.critic_optimizer.step()
        return {"imagined_actor_loss": float(actor_loss.item()), "imagined_critic_loss": float(critic_loss.item()), "imagined_length": float(rollout["alive"].sum(1).mean().item()), "imagined_return": float(returns_tensor[:, 0].mean().item())}
