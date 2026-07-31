"""Ensemble dynamics heads and disagreement-based model uncertainty."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicsEnsemble(nn.Module):
    """Independent prediction heads used to detect out-of-distribution rollouts."""

    def __init__(self, feature_dim: int, action_dim: int, context_dim: int, ensemble_size: int = 5, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, min(32, hidden_dim))
        input_dim = feature_dim + min(32, hidden_dim) + context_dim
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, feature_dim + 2)) for _ in range(ensemble_size)])

    def forward(self, feature: Tensor, action: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return head features, rewards, continuations, each with leading ensemble axis."""
        inputs = torch.cat([feature, self.action_embedding(action.long()), context], dim=-1)
        values = torch.stack([head(inputs) for head in self.heads], dim=0)
        return values[..., :-2], values[..., -2], torch.sigmoid(values[..., -1])

    def disagreement(self, feature: Tensor, action: Tensor, context: Tensor) -> Tensor:
        """Scalar predictive disagreement per batch element."""
        next_features, rewards, _ = self(feature, action, context)
        return next_features.var(dim=0, unbiased=False).mean(-1).sqrt() + rewards.var(dim=0, unbiased=False).sqrt()
