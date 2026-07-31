"""Latent state-value critic for imagined trajectories."""

from __future__ import annotations

from torch import Tensor, nn


class LatentCritic(nn.Module):
    """Predicts a scalar discounted return from an RSSM feature."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(); self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
    def forward(self, feature: Tensor) -> Tensor:
        """Return values with the final singleton dimension removed."""
        return self.network(feature).squeeze(-1)
