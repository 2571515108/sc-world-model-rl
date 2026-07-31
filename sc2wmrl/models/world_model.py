"""Opponent-conditioned RSSM world model with all required prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sc2wmrl.envs.base_macro_env import ACTION_COUNT
from .ensemble import DynamicsEnsemble
from .opponent_encoder import OpponentContextEncoder
from .rssm import RSSM, RSSMState


@dataclass(frozen=True)
class WorldModelConfig:
    observation_dim: int
    deterministic_dim: int = 256
    stochastic_dim: int = 64
    action_embedding_dim: int = 32
    opponent_embedding_dim: int = 32
    hidden_dim: int = 256
    ensemble_size: int = 5
    event_dim: int = 7
    opponent_count: int = 128
    kl_free_nats: float = 1.0
    kl_balance: float = 0.8


@dataclass
class WorldModelLoss:
    """Loss decomposition logged by the pretraining loop."""
    total: Tensor
    observation: Tensor
    reward: Tensor
    continuation: Tensor
    event: Tensor
    opponent: Tensor
    kl: Tensor


class WorldModel(nn.Module):
    """Trains from real sequences and generates uncertainty-aware imagined steps."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.observation_encoder = nn.Sequential(nn.Linear(config.observation_dim, h), nn.ELU(), nn.Linear(h, h), nn.ELU())
        self.opponent_id_embedding = nn.Embedding(config.opponent_count, config.opponent_embedding_dim)
        self.opponent_encoder = OpponentContextEncoder(config.observation_dim, ACTION_COUNT, config.opponent_embedding_dim)
        self.rssm = RSSM(h, ACTION_COUNT, config.opponent_embedding_dim, config.deterministic_dim, config.stochastic_dim, config.action_embedding_dim, h)
        f = config.deterministic_dim + config.stochastic_dim
        self.observation_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, config.observation_dim))
        self.reward_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, 1))
        self.continue_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, 1))
        self.event_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, config.event_dim))
        self.opponent_action_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, ACTION_COUNT))
        self.value_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, 1))
        self.ensemble = DynamicsEnsemble(f, ACTION_COUNT, config.opponent_embedding_dim, config.ensemble_size, h)

    def _context(self, observations: Tensor, actions: Tensor, opponent_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        online, logits = self.opponent_encoder(observations, actions)
        if opponent_ids is None:
            return online, logits
        if opponent_ids.shape != (observations.shape[0],): raise ValueError("opponent IDs must have shape [B]")
        return online + self.opponent_id_embedding(opponent_ids.long()), logits

    def observe(self, observations: Tensor, actions: Tensor, opponent_ids: Tensor | None = None, sample: bool = True) -> tuple[list[RSSMState], list[RSSMState], Tensor, Tensor]:
        """Infer posterior states for an observed ``[B,T,O]`` trajectory."""
        if observations.ndim != 3 or observations.shape[-1] != self.config.observation_dim or actions.shape != observations.shape[:2]:
            raise ValueError("world model sequences must be [B,T,observation_dim] and [B,T]")
        context, strategy_logits = self._context(observations, actions, opponent_ids)
        state = self.rssm.initial(observations.shape[0], observations.device); priors: list[RSSMState] = []; posteriors: list[RSSMState] = []
        for step in range(observations.shape[1]):
            prior, state = self.rssm.posterior(state, actions[:, step], context, self.observation_encoder(observations[:, step]), sample)
            priors.append(prior); posteriors.append(state)
        return priors, posteriors, context, strategy_logits

    def loss(self, observations: Tensor, actions: Tensor, rewards: Tensor, continues: Tensor, events: Tensor,
             opponent_actions: Tensor | None = None, opponent_ids: Tensor | None = None, strategy_targets: Tensor | None = None, mask: Tensor | None = None) -> WorldModelLoss:
        """Compute masked reconstruction, prediction, KL-balanced losses with free bits."""
        priors, posteriors, _, strategy_logits = self.observe(observations, actions, opponent_ids)
        features = torch.stack([self.rssm.feature(state) for state in posteriors], dim=1)
        if mask is None: mask = torch.ones(observations.shape[:2], device=observations.device)
        mask = mask.float()
        if mask.shape != observations.shape[:2] or mask.sum() <= 0: raise ValueError("invalid sequence mask")
        def avg(value: Tensor) -> Tensor: return (value * mask).sum() / mask.sum()
        observation_loss = avg(F.mse_loss(self.observation_head(features), observations, reduction="none").mean(-1))
        reward_loss = avg(F.mse_loss(self.reward_head(features).squeeze(-1), rewards, reduction="none"))
        continuation_loss = avg(F.binary_cross_entropy_with_logits(self.continue_head(features).squeeze(-1), continues, reduction="none"))
        event_loss = avg(F.binary_cross_entropy_with_logits(self.event_head(features), events, reduction="none").mean(-1))
        target_actions = actions if opponent_actions is None else opponent_actions
        action_logits = self.opponent_action_head(features)
        opponent_loss = avg(F.cross_entropy(action_logits.transpose(1, 2), target_actions.long(), reduction="none"))
        if strategy_targets is not None:
            if strategy_targets.shape != (observations.shape[0],): raise ValueError("strategy targets must have shape [B]")
            opponent_loss = opponent_loss + F.cross_entropy(strategy_logits, strategy_targets.long())
        balanced_kl: list[Tensor] = []
        for prior, post in zip(priors, posteriors):
            # KL balancing keeps gradients flowing to both inference and prior dynamics.
            post_detached = RSSMState(post.deter, post.stoch, post.mean.detach(), post.std.detach())
            prior_detached = RSSMState(prior.deter, prior.stoch, prior.mean.detach(), prior.std.detach())
            balanced_kl.append(self.config.kl_balance * self.rssm.kl(post_detached, prior) + (1 - self.config.kl_balance) * self.rssm.kl(post, prior_detached))
        raw_kl = torch.stack(balanced_kl, dim=1)
        kl = avg(torch.maximum(raw_kl, torch.full_like(raw_kl, self.config.kl_free_nats)))
        total = observation_loss + reward_loss + continuation_loss + event_loss + opponent_loss + kl
        if not torch.isfinite(total): raise FloatingPointError("world model loss is non-finite")
        return WorldModelLoss(total, observation_loss, reward_loss, continuation_loss, event_loss, opponent_loss, kl)

    def imagine_step(self, state: RSSMState, action: Tensor, opponent_context: Tensor) -> tuple[RSSMState, Tensor, Tensor, Tensor, Tensor]:
        """Produce latent transition, pessimism-ready reward, continuation, uncertainty, and value."""
        next_state = self.rssm.prior(state, action, opponent_context, sample=True); feature = self.rssm.feature(next_state)
        uncertainty = self.ensemble.disagreement(feature, action, opponent_context)
        return next_state, self.reward_head(feature).squeeze(-1), torch.sigmoid(self.continue_head(feature).squeeze(-1)), uncertainty, self.value_head(feature).squeeze(-1)
