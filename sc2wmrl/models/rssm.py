"""Stochastic recurrent state-space model used by the Phase 2 world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass
class RSSMState:
    """Deterministic and stochastic latent state tensors."""
    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor


class RSSM(nn.Module):
    """RSSM with action/opponent-conditioned prior and observation posterior."""

    def __init__(self, embed_dim: int, action_dim: int, opponent_dim: int, deterministic_dim: int = 256,
                 stochastic_dim: int = 64, action_embedding_dim: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.deterministic_dim, self.stochastic_dim = deterministic_dim, stochastic_dim
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        self.gru = nn.GRUCell(stochastic_dim + action_embedding_dim + opponent_dim, deterministic_dim)
        self.prior_net = nn.Sequential(nn.Linear(deterministic_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stochastic_dim))
        self.posterior_net = nn.Sequential(nn.Linear(deterministic_dim + embed_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stochastic_dim))

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        """Create the zero latent state for a batch."""
        zeros_d = torch.zeros(batch_size, self.deterministic_dim, device=device)
        zeros_s = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return RSSMState(zeros_d, zeros_s, zeros_s, torch.ones_like(zeros_s))

    @staticmethod
    def _stats(parameters: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_std = parameters.chunk(2, dim=-1)
        return mean, torch.nn.functional.softplus(raw_std) + 0.1

    @staticmethod
    def _sample(mean: Tensor, std: Tensor, sample: bool) -> Tensor:
        return mean + std * torch.randn_like(std) if sample else mean

    def prior(self, previous: RSSMState, action: Tensor, opponent_context: Tensor, sample: bool = True) -> RSSMState:
        """Advance prior dynamics for a batch of discrete macro actions."""
        if action.ndim != 1 or opponent_context.ndim != 2:
            raise ValueError("RSSM prior expects action [B] and context [B,C]")
        deter = self.gru(torch.cat([previous.stoch, self.action_embedding(action.long()), opponent_context], dim=-1), previous.deter)
        mean, std = self._stats(self.prior_net(deter)); stoch = self._sample(mean, std, sample)
        return RSSMState(deter, stoch, mean, std)

    def posterior(self, previous: RSSMState, action: Tensor, opponent_context: Tensor, observation_embed: Tensor, sample: bool = True) -> tuple[RSSMState, RSSMState]:
        """Return prior and posterior state for one observed transition."""
        prior = self.prior(previous, action, opponent_context, sample=False)
        mean, std = self._stats(self.posterior_net(torch.cat([prior.deter, observation_embed], dim=-1)))
        posterior = RSSMState(prior.deter, self._sample(mean, std, sample), mean, std)
        return prior, posterior

    def feature(self, state: RSSMState) -> Tensor:
        """Concatenate latent components for decoder/prediction heads."""
        return torch.cat([state.deter, state.stoch], dim=-1)

    @staticmethod
    def kl(posterior: RSSMState, prior: RSSMState) -> Tensor:
        """Analytic diagonal-Gaussian KL for each batch item."""
        return torch.distributions.kl_divergence(Normal(posterior.mean, posterior.std), Normal(prior.mean, prior.std)).sum(-1)
