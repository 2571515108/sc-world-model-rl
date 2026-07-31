"""Stateful opponent-conditioned policy for deployment with an RSSM world model."""

from __future__ import annotations

import zlib

import numpy as np
import torch

from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.rssm import RSSMState
from sc2wmrl.models.world_model import WorldModel
from sc2wmrl.utils.device import resolve_device


class WorldModelAgent:
    """Maintains a posterior latent state and samples only allowed macro actions."""

    def __init__(self, world_model: WorldModel, actor: LatentActor, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.world_model, self.actor = world_model.to(self.device).eval(), actor.to(self.device).eval()
        self.state: RSSMState | None = None; self.context: torch.Tensor | None = None; self.previous_action = 0

    def reset(self, opponent_id: str = "unknown") -> None:
        """Clear latent history and choose the stable identity embedding for a match."""
        self.state = self.world_model.rssm.initial(1, self.device)
        index = zlib.crc32(opponent_id.encode()) % self.world_model.config.opponent_count
        self.context = self.world_model.opponent_id_embedding(torch.tensor([index], device=self.device)); self.previous_action = 0

    @torch.no_grad()
    def act(self, observation: np.ndarray, action_mask: np.ndarray, deterministic: bool = False) -> int:
        """Update posterior from one observation then produce a legal macro action."""
        if self.state is None or self.context is None: self.reset()
        observation_tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None], device=self.device)
        mask = torch.as_tensor(np.asarray(action_mask, dtype=np.bool_)[None], device=self.device)
        if observation_tensor.shape[-1] != self.world_model.config.observation_dim or not mask.any(): raise ValueError("invalid world-model action input")
        _, self.state = self.world_model.rssm.posterior(self.state, torch.tensor([self.previous_action], device=self.device), self.context,
                                                        self.world_model.observation_encoder(observation_tensor), sample=not deterministic)
        distribution = self.actor.distribution(self.world_model.rssm.feature(self.state), mask); action = torch.argmax(distribution.logits, -1) if deterministic else distribution.sample()
        self.previous_action = int(action.item()); return self.previous_action
