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
    universal_intent_count: int = 14
    universal_intent_coefficient: float = 0.25
    race_embedding_dim: int = 8


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
    universal_intent: Tensor


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
        self.race_embedding = nn.Embedding(4, config.race_embedding_dim)
        self.race_adapter = nn.Sequential(nn.Linear(config.race_embedding_dim * 2, h), nn.ELU(), nn.Linear(h, h))
        self.race_observation_bias = nn.Embedding(4, config.observation_dim)
        with torch.no_grad():
            self.race_embedding.weight[0].zero_()
            self.race_observation_bias.weight[0].zero_()
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
        self.universal_intent_head = nn.Sequential(nn.Linear(f, h), nn.ELU(), nn.Linear(h, config.universal_intent_count))
        self.ensemble = DynamicsEnsemble(f, ACTION_COUNT, config.opponent_embedding_dim, config.ensemble_size, h)

    @staticmethod
    def _race_ids(value: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if value.shape != (batch_size,):
            raise ValueError("player and opponent race IDs must have shape [B]")
        if torch.any((value < 0) | (value > 3)):
            raise ValueError("race IDs must be in [0, 3]")
        return value.long()

    def _race_context(self, player_races: Tensor | None, opponent_races: Tensor | None, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        player = self._race_ids(player_races, batch_size, device); opponent = self._race_ids(opponent_races, batch_size, device)
        if not bool(player.any() or opponent.any()):
            return torch.zeros(batch_size, self.config.hidden_dim, device=device), player
        context = self.race_adapter(torch.cat([self.race_embedding(player), self.race_embedding(opponent)], dim=-1))
        return context, player

    def _encode_observations(self, observations: Tensor, player_races: Tensor | None, opponent_races: Tensor | None) -> Tensor:
        if observations.ndim not in {2, 3}:
            raise ValueError("observations must be [B,D] or [B,T,D]")
        batch_size = observations.shape[0]; context, _ = self._race_context(player_races, opponent_races, batch_size, observations.device)
        encoded = self.observation_encoder(observations)
        return encoded + (context if observations.ndim == 2 else context.unsqueeze(1))

    def _decode_observations(self, features: Tensor, player_races: Tensor | None, opponent_races: Tensor | None) -> Tensor:
        batch_size = features.shape[0]; _, player = self._race_context(player_races, opponent_races, batch_size, features.device)
        bias = self.race_observation_bias(player)
        return self.observation_head(features) + (bias if features.ndim == 2 else bias.unsqueeze(1))

    def _context(self, observations: Tensor, actions: Tensor, opponent_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        online, logits = self.opponent_encoder(observations, actions)
        if opponent_ids is None: return online, logits
        if opponent_ids.shape != (observations.shape[0],): raise ValueError("opponent IDs must have shape [B]")
        return online + self.opponent_id_embedding(opponent_ids.long()), logits

    def observe(self, observations: Tensor, actions: Tensor, opponent_ids: Tensor | None = None, player_races: Tensor | None = None,
                opponent_races: Tensor | None = None, *, next_observations: Tensor | None = None, sample: bool = True,
                burn_in_length: int = 0) -> tuple[list[RSSMState], list[RSSMState], Tensor, Tensor]:
        """Infer posterior next states aligned to ``action_t -> next_observation_t``."""
        if observations.ndim != 3 or observations.shape[-1] != self.config.observation_dim or actions.shape != observations.shape[:2]:
            raise ValueError("world model sequences must be [B,T,observation_dim] and [B,T]")
        target_observations = observations if next_observations is None else next_observations
        if target_observations.shape != observations.shape or not 0 <= burn_in_length < observations.shape[1] + 1:
            raise ValueError("invalid next-observation shape or burn-in length")
        context, strategy_logits = self._context(observations, actions, opponent_ids)
        # Encode all time steps at once; only the RSSM recurrence remains sequential.
        initial_embed = self._encode_observations(observations[:, 0], player_races, opponent_races)
        target_embeds = self._encode_observations(target_observations, player_races, opponent_races)
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
             mask: Tensor | None = None, burn_in_length: int = 0, player_races: Tensor | None = None,
             opponent_races: Tensor | None = None, feature_valid_masks: Tensor | None = None,
             next_feature_valid_masks: Tensor | None = None, next_action_mask_valid: Tensor | None = None,
             universal_intents: Tensor | None = None, universal_intent_valid: Tensor | None = None) -> WorldModelLoss:
        """Compute masked losses; burn-in updates state but has no gradient/loss."""
        priors, posteriors, context, strategy_logits = self.observe(observations, actions, opponent_ids, player_races, opponent_races, next_observations=next_observations, burn_in_length=burn_in_length)
        features = torch.stack([self.rssm.feature(state) for state in posteriors], dim=1)
        initial = self.rssm.initial_posterior(self._encode_observations(observations[:, 0], player_races, opponent_races), sample=False)
        source_features = torch.cat([self.rssm.feature(initial).unsqueeze(1), features[:, :-1].detach()], dim=1)
        if mask is None: mask = torch.ones(observations.shape[:2], device=observations.device)
        mask = mask.float()
        if burn_in_length: mask = mask.clone(); mask[:, :burn_in_length] = 0
        if mask.shape != observations.shape[:2] or mask.sum() <= 0: raise ValueError("invalid sequence mask")
        def avg(value: Tensor) -> Tensor: return (value * mask).sum() / mask.sum()
        feature_mask = torch.ones_like(next_observations) if next_feature_valid_masks is None else next_feature_valid_masks.float()
        if feature_mask.shape != next_observations.shape:
            raise ValueError("next feature-validity masks must be [B,T,D]")
        observation_error = F.mse_loss(self._decode_observations(features, player_races, opponent_races), next_observations, reduction="none")
        feature_denominator = (feature_mask * mask.unsqueeze(-1)).sum().clamp_min(1.0)
        observation_loss = (observation_error * feature_mask * mask.unsqueeze(-1)).sum() / feature_denominator
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
        if universal_intents is None:
            universal_intent_loss = torch.zeros((), device=features.device)
        else:
            if universal_intents.shape != observations.shape[:2]:
                raise ValueError("universal intents must be [B,T]")
            intent_valid = torch.ones_like(mask, dtype=torch.bool) if universal_intent_valid is None else universal_intent_valid.bool()
            if intent_valid.shape != observations.shape[:2]:
                raise ValueError("universal intent validity must be [B,T]")
            intent_mask = mask * intent_valid.float(); denominator = intent_mask.sum()
            if denominator <= 0:
                universal_intent_loss = features.sum() * 0.0
            else:
                values = F.cross_entropy(self.universal_intent_head(features).transpose(1, 2), universal_intents.long(), reduction="none")
                universal_intent_loss = (values * intent_mask).sum() / denominator
        if next_action_masks is None:
            action_mask_loss = torch.zeros((), device=features.device)
        else:
            if next_action_masks.shape != (*observations.shape[:2], ACTION_COUNT): raise ValueError("next action masks must be [B,T,A]")
            legal_valid = torch.ones_like(mask, dtype=torch.bool) if next_action_mask_valid is None else next_action_mask_valid.bool()
            if legal_valid.shape != observations.shape[:2]:
                raise ValueError("next action-mask validity must be [B,T]")
            legal_mask = mask * legal_valid.float(); denominator = legal_mask.sum()
            action_mask_loss = features.sum() * 0.0 if denominator <= 0 else (F.binary_cross_entropy_with_logits(self.action_mask_head(features), next_action_masks.float(), reduction="none").mean(-1) * legal_mask).sum() / denominator
        total = observation_loss + reward_loss + continuation_loss + event_loss + opponent_loss + kl + self.config.ensemble_coefficient * ensemble + self.config.action_mask_coefficient * action_mask_loss + self.config.universal_intent_coefficient * universal_intent_loss
        if not torch.isfinite(total): raise FloatingPointError("world model loss is non-finite")
        return WorldModelLoss(total, observation_loss, reward_loss, continuation_loss, event_loss, opponent_loss, kl, ensemble, action_mask_loss, universal_intent_loss)

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
