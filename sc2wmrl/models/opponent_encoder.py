"""Opponent identity and online behavioral-context encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class OpponentContextEncoder(nn.Module):
    """Infer a context vector and strategy distribution from observed history."""

    def __init__(self, observation_dim: int, action_dim: int, context_dim: int = 32, strategy_dim: int = 6) -> None:
        super().__init__()
        self.observation_dim, self.action_dim, self.context_dim = observation_dim, action_dim, context_dim
        self.action_embedding = nn.Embedding(action_dim, context_dim)
        self.gru = nn.GRU(observation_dim + context_dim, context_dim, batch_first=True)
        self.strategy_head = nn.Linear(context_dim, strategy_dim)

    def forward(self, enemy_observation_sequence: Tensor, action_sequence: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return final opponent context and strategy logits for ``[B,T,*]`` input."""
        if enemy_observation_sequence.ndim != 3 or enemy_observation_sequence.shape[-1] != self.observation_dim:
            raise ValueError("invalid opponent observation sequence shape")
        if action_sequence.shape != enemy_observation_sequence.shape[:2]:
            raise ValueError("opponent actions must align with observation sequence")
        embedded = self.action_embedding(action_sequence.long())
        _, hidden = self.gru(torch.cat([enemy_observation_sequence, embedded], dim=-1), hidden_state)
        context = hidden[-1]
        return context, self.strategy_head(context)
