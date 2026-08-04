"""Aligned RSSM world model with trained uncertainty and imagined legality heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sc2wmrl.envs.base_macro_env import ACTION_COUNT, MacroAction
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
    ensemble_coefficient: float = 1.0
    action_mask_coefficient: float = 0.25


@dataclass
class WorldModelLoss:
    """Every trainable world-model component has an explicit loss source."""

    total: Tensor
    observation: Tensor
    reward: Tensor
    continuation: Tensor
    event: Tensor
    opponent: Tensor
    kl: Tensor
    ensemble: Tensor
    action_mask: Tensor


class WorldModel(nn.Module):
    """Predict ``(next observation, reward, continuation)`` for each action.

    Checkpoint format version 2 uses action-to-next-observation alignment.  It
    intentionally does not load version-1 weights, whose recurrent semantics
    treated ``action_t`` as conditioning ``observation_t``.
    """

    FORMAT_VERSION = 2

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__(); self.config = config; h = config.hidden_dim
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
        self.action_mask_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, ACTION_COUNT))
        self.ensemble = DynamicsEnsemble(f, ACTION_COUNT, config.opponent_embedding_dim, config.ensemble_size, h)

    def _context(self, observations: Tensor, actions: Tensor, opponent_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        online, logits = self.opponent_encoder(observations, actions)
        if opponent_ids is None: return online, logits
        if opponent_ids.shape != (observations.shape[0],): raise ValueError("opponent IDs must have shape [B]")
        return online + self.opponent_id_embedding(opponent_ids.long()), logits

    def observe(self, observations: Tensor, actions: Tensor, opponent_ids: Tensor | None = None, *, next_observations: Tensor | None = None, sample: bool = True, burn_in_length: int = 0) -> tuple[list[RSSMState], list[RSSMState], Tensor, Tensor]:
        """Infer posterior next states aligned to ``action_t -> next_observation_t``."""
        if observations.ndim != 3 or observations.shape[-1] != self.config.observation_dim or actions.shape != observations.shape[:2]:
            raise ValueError("world model sequences must be [B,T,observation_dim] and [B,T]")
        target_observations = observations if next_observations is None else next_observations
        if target_observations.shape != observations.shape or not 0 <= burn_in_length < observations.shape[1] + 1:
            raise ValueError("invalid next-observation shape or burn-in length")
        context, strategy_logits = self._context(observations, actions, opponent_ids)
        # Encode all time steps at once; only the RSSM recurrence remains sequential.
        initial_embed = self.observation_encoder(observations[:, 0])
        target_embeds = self.observation_encoder(target_observations.reshape(-1, observations.shape[-1])).reshape(*observations.shape[:2], -1)
        state = self.rssm.initial_posterior(initial_embed, sample=sample)
        priors: list[RSSMState] = []; posteriors: list[RSSMState] = []
        for step in range(observations.shape[1]):
            if step < burn_in_length:
                with torch.no_grad():
                    prior = self.rssm.prior(state, actions[:, step], context, sample=sample)
                    state = self.rssm.posterior_from_prior(prior, target_embeds[:, step], sample=sample)
                if step + 1 == burn_in_length: state = RSSM.detach(state)
            else:
                prior = self.rssm.prior(state, actions[:, step], context, sample=sample)
                state = self.rssm.posterior_from_prior(prior, target_embeds[:, step], sample=sample)
            priors.append(prior); posteriors.append(state)
        return priors, posteriors, context, strategy_logits

    def loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor, rewards: Tensor, continues: Tensor, events: Tensor,
             next_action_masks: Tensor | None = None, opponent_actions: Tensor | None = None, opponent_ids: Tensor | None = None,
             strategy_targets: Tensor | None = None, opponent_action_valid: Tensor | None = None,
             mask: Tensor | None = None, burn_in_length: int = 0) -> WorldModelLoss:
        """Compute masked losses; burn-in updates state but has no gradient/loss."""
        priors, posteriors, context, strategy_logits = self.observe(observations, actions, opponent_ids, next_observations=next_observations, burn_in_length=burn_in_length)
        features = torch.stack([self.rssm.feature(state) for state in posteriors], dim=1)
        initial = self.rssm.initial_posterior(self.observation_encoder(observations[:, 0]), sample=False)
        source_features = torch.cat([self.rssm.feature(initial).unsqueeze(1), features[:, :-1].detach()], dim=1)
        if mask is None: mask = torch.ones(observations.shape[:2], device=observations.device)
        mask = mask.float()
        if burn_in_length: mask = mask.clone(); mask[:, :burn_in_length] = 0
        if mask.shape != observations.shape[:2] or mask.sum() <= 0: raise ValueError("invalid sequence mask")
        def avg(value: Tensor) -> Tensor: return (value * mask).sum() / mask.sum()
        observation_loss = avg(F.mse_loss(self.observation_head(features), next_observations, reduction="none").mean(-1))
        reward_loss = avg(F.mse_loss(self.reward_head(features).squeeze(-1), rewards, reduction="none"))
        continuation_loss = avg(F.binary_cross_entropy_with_logits(self.continue_head(features).squeeze(-1), continues, reduction="none"))
        event_loss = avg(F.binary_cross_entropy_with_logits(self.event_head(features), events, reduction="none").mean(-1))
        if opponent_actions is None:
            opponent_loss = features.sum() * 0.0
        else:
            if opponent_actions.shape != observations.shape[:2]:
                raise ValueError("opponent actions must have shape [B,T]")
            valid = torch.ones_like(mask, dtype=torch.bool) if opponent_action_valid is None else opponent_action_valid.bool()
            if valid.shape != observations.shape[:2]:
                raise ValueError("opponent action validity must have shape [B,T]")
            opponent_mask = mask * valid.float()
            denominator = opponent_mask.sum()
            if denominator <= 0:
                opponent_loss = features.sum() * 0.0
            else:
                values = F.cross_entropy(self.opponent_action_head(features).transpose(1, 2), opponent_actions.long(), reduction="none")
                opponent_loss = (values * opponent_mask).sum() / denominator
        if strategy_targets is not None:
            if strategy_targets.shape != (observations.shape[0],): raise ValueError("strategy targets must have shape [B]")
            opponent_loss = opponent_loss + F.cross_entropy(strategy_logits, strategy_targets.long())
        balanced_kl = []
        for prior, post in zip(priors, posteriors):
            post_detached = RSSMState(post.deter, post.stoch, post.mean.detach(), post.std.detach())
            prior_detached = RSSMState(prior.deter, prior.stoch, prior.mean.detach(), prior.std.detach())
            balanced_kl.append(self.config.kl_balance * self.rssm.kl(post_detached, prior) + (1 - self.config.kl_balance) * self.rssm.kl(post, prior_detached))
        kl = avg(torch.maximum(torch.stack(balanced_kl, dim=1), torch.full_like(mask, self.config.kl_free_nats)))
        predicted_features, predicted_rewards, predicted_continue_logits = self.ensemble(source_features, actions, context)
        bootstrap = torch.rand(predicted_features.shape[:3], device=features.device) < 0.5
        target_feature = features.detach().unsqueeze(0).expand_as(predicted_features)
        ensemble_mask = bootstrap.float() * mask.unsqueeze(0)
        denom = ensemble_mask.sum().clamp_min(1.0)
        ensemble = (((predicted_features - target_feature).pow(2).mean(-1) + (predicted_rewards - rewards.unsqueeze(0)).pow(2) + F.binary_cross_entropy_with_logits(predicted_continue_logits, continues.unsqueeze(0).expand_as(predicted_continue_logits), reduction="none")) * ensemble_mask).sum() / denom
        if next_action_masks is None:
            action_mask_loss = torch.zeros((), device=features.device)
        else:
            if next_action_masks.shape != (*observations.shape[:2], ACTION_COUNT): raise ValueError("next action masks must be [B,T,A]")
            action_mask_loss = avg(F.binary_cross_entropy_with_logits(self.action_mask_head(features), next_action_masks.float(), reduction="none").mean(-1))
        total = observation_loss + reward_loss + continuation_loss + event_loss + opponent_loss + kl + self.config.ensemble_coefficient * ensemble + self.config.action_mask_coefficient * action_mask_loss
        if not torch.isfinite(total): raise FloatingPointError("world model loss is non-finite")
        return WorldModelLoss(total, observation_loss, reward_loss, continuation_loss, event_loss, opponent_loss, kl, ensemble, action_mask_loss)

    def imagined_action_mask(self, feature: Tensor, threshold: float = 0.5) -> Tensor:
        """Predict next legal actions and enforce the invariant that NO_OP is legal."""
        mask = torch.sigmoid(self.action_mask_head(feature)) >= threshold
        mask[..., MacroAction.NO_OP] = True
        return mask

    def imagine_step(self, state: RSSMState, action: Tensor, opponent_context: Tensor) -> tuple[RSSMState, Tensor, Tensor, Tensor]:
        """Produce one prior latent transition, reward, continuation, and uncertainty."""
        next_state = self.rssm.prior(state, action, opponent_context, sample=True); feature = self.rssm.feature(next_state)
        uncertainty = self.ensemble.disagreement(self.rssm.feature(state), action, opponent_context)
        return next_state, self.reward_head(feature).squeeze(-1), torch.sigmoid(self.continue_head(feature).squeeze(-1)), uncertainty
