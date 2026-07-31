"""Latent macro-action policy used during imagined actor-critic updates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentActor(nn.Module):
    """Categorical actor that always applies an explicit boolean action mask."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(); self.action_dim = action_dim
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, action_dim))

    def distribution(self, feature: Tensor, action_mask: Tensor) -> torch.distributions.Categorical:
        """Create a legal-only categorical distribution."""
        if action_mask.shape != feature.shape[:-1] + (self.action_dim,) or not torch.all(action_mask.any(-1)):
            raise ValueError("invalid imagined action mask")
        return torch.distributions.Categorical(logits=self.network(feature).masked_fill(~action_mask.bool(), float("-inf")))
