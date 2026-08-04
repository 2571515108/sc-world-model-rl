"""Schedules real PPO and imagined updates without replacing real experience."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sc2wmrl.training.imagination_trainer import ImaginationTrainer
from sc2wmrl.training.ppo_trainer import PPOTrainer, Rollout
from sc2wmrl.training.world_model_trainer import sequence_batch
from sc2wmrl.replay.sequence_sampler import SequenceSampler


@dataclass(frozen=True)
class HybridConfig:
    """Controlled ramp from model-free PPO to a bounded imagination contribution."""
    imagination_weight_max: float = 0.5
    imagination_warmup_updates: int = 100
    imagined_batch_size: int = 8
    imagined_sequence_length: int = 16


class HybridTrainer:
    """Runs always-on real PPO and optional post-warmup imagined actor-critic updates."""

    def __init__(self, ppo: PPOTrainer, imagination: ImaginationTrainer, sampler: SequenceSampler, config: HybridConfig | None = None) -> None:
        self.ppo, self.imagination, self.sampler, self.config, self.updates = ppo, imagination, sampler, config or HybridConfig(), 0

    def imagination_weight(self) -> float:
        """Return a deterministic linear warmup weight capped by configuration."""
        return self.config.imagination_weight_max * min(1.0, self.updates / max(1, self.config.imagination_warmup_updates))

    def update(self, rollout: Rollout | list[tuple[str, PPOTrainer, Rollout]]) -> dict[str, float]:
        """Update a shared PPO policy from one or more on-policy environments.

        A list lets the curriculum alternate real and synthetic rollouts while
        preserving PPO's on-policy update rule for every source independently.
        """
        entries = [("real", self.ppo, rollout)] if isinstance(rollout, Rollout) else rollout
        if not entries:
            raise ValueError("hybrid update requires at least one rollout")
        result: dict[str, float] = {}
        for source, trainer, source_rollout in entries:
            result.update({f"{source}_{key}": value for key, value in trainer.update(source_rollout).items()})
        weight = self.imagination_weight()
        result["imagination_weight"] = weight
        if weight > 0:
            sequences = self.sampler.sample(self.config.imagined_batch_size, self.config.imagined_sequence_length)
            batch = sequence_batch(sequences, str(next(self.imagination.actor.parameters()).device))
            masks = torch.as_tensor(np.stack([[item.action_mask for item in seq] for seq in sequences]), device=batch["observations"].device)
            imagined = self.imagination.rollout(batch["observations"], batch["actions"], masks, batch["opponent_ids"], batch["next_observations"])
            result.update({key: weight * value for key, value in self.imagination.update(imagined).items()})
            # Distill the imagined latent actor's first decision into the deployed
            # observation policy, so imagination changes the PPO checkpoint rather
            # than training an unreachable side policy.
            agent = self.ppo.agent
            distribution, _ = agent._distribution(batch["observations"][:, -1], masks[:, -1])
            distillation_loss = -distribution.log_prob(imagined["actions"][:, 0].detach()).mean()
            agent.optimizer.zero_grad(set_to_none=True); (weight * distillation_loss).backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.config.max_grad_norm); agent.optimizer.step()
            result["imagination_distillation_loss"] = float(distillation_loss.item())
        self.updates += 1
        return result
