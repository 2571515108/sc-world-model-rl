"""Sequence replay pretraining and open-loop evaluation for the world model."""

from __future__ import annotations

import zlib
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from sc2wmrl.models.world_model import WorldModel, WorldModelLoss
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.utils.device import resolve_device


def sequence_batch(sequences: list[list[object]], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Convert validated replay sequences into fixed tensors for RSSM training."""
    if not sequences or not sequences[0]: raise ValueError("sequences cannot be empty")
    observations = np.stack([[item.observation for item in sequence] for sequence in sequences])
    actions = np.asarray([[item.action for item in sequence] for sequence in sequences], dtype=np.int64)
    rewards = np.asarray([[item.reward for item in sequence] for sequence in sequences], dtype=np.float32)
    continues = np.asarray([[not (item.terminated or item.truncated) for item in sequence] for sequence in sequences], dtype=np.float32)
    events = np.stack([[item.events for item in sequence] for sequence in sequences])
    opponent_actions = np.asarray([[int(item.info.get("opponent_action", 0)) for item in sequence] for sequence in sequences], dtype=np.int64)
    strategy_ids = {"rush": 0, "economy": 1, "defensive": 2, "ground_tech": 3, "air_tech": 4, "unknown": 5}
    strategies = np.asarray([strategy_ids.get(str(sequence[0].info.get("enemy_strategy", "unknown")).removesuffix("_bot"), 5) for sequence in sequences], dtype=np.int64)
    opponent_ids = np.asarray([zlib.crc32(sequence[0].opponent_id.encode()) % 128 for sequence in sequences], dtype=np.int64)
    return {"observations": torch.as_tensor(observations, device=device), "actions": torch.as_tensor(actions, device=device),
            "rewards": torch.as_tensor(rewards, device=device), "continues": torch.as_tensor(continues, device=device),
            "events": torch.as_tensor(events, device=device), "opponent_actions": torch.as_tensor(opponent_actions, device=device), "opponent_ids": torch.as_tensor(opponent_ids, device=device), "strategy_targets": torch.as_tensor(strategies, device=device)}


class WorldModelTrainer:
    """Optimizes model sequences with clipping and checkpoint-safe state."""

    def __init__(self, model: WorldModel, learning_rate: float = 3e-4, max_grad_norm: float = 100.0, device: str = "auto") -> None:
        resolved_device = resolve_device(device)
        self.model, self.device, self.max_grad_norm = model.to(resolved_device), str(resolved_device), max_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_step(self, sequences: list[list[object]]) -> dict[str, float]:
        """Run one finite, clipped optimization update."""
        batch = sequence_batch(sequences, self.device); self.model.train()
        loss: WorldModelLoss = self.model.loss(**batch)
        self.optimizer.zero_grad(set_to_none=True); loss.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm); self.optimizer.step()
        return {item.name: float(getattr(loss, item.name).detach().item()) for item in fields(loss)}

    @torch.no_grad()
    def open_loop_evaluate(self, sequences: list[list[object]]) -> dict[str, float]:
        """Measure posterior reconstruction and prior multi-step observation drift."""
        batch = sequence_batch(sequences, self.device); self.model.eval()
        priors, posts, context, _ = self.model.observe(batch["observations"], batch["actions"], batch["opponent_ids"], sample=False)
        posterior_features = torch.stack([self.model.rssm.feature(state) for state in posts], 1)
        reconstruction_mse = torch.mean((self.model.observation_head(posterior_features) - batch["observations"]) ** 2).item()
        state = posts[0]; predictions = []
        for time in range(batch["observations"].shape[1]):
            state = self.model.rssm.prior(state, batch["actions"][:, time], context, sample=False)
            predictions.append(self.model.observation_head(self.model.rssm.feature(state)))
        open_loop = torch.stack(predictions, 1)
        horizon_errors = ((open_loop - batch["observations"]) ** 2).mean(dim=(0, 2))
        return {"reconstruction_mse": float(reconstruction_mse), **{f"open_loop_mse_h{index + 1}": float(error) for index, error in enumerate(horizon_errors)}}

    def fit(self, sampler: SequenceSampler, updates: int, batch_size: int, sequence_length: int, burn_in_length: int = 0) -> list[dict[str, float]]:
        """Train a requested number of replay batches."""
        if updates <= 0: raise ValueError("updates must be positive")
        return [self.train_step(sampler.sample(batch_size, sequence_length, burn_in_length)) for _ in range(updates)]

    def save(self, path: str | Path) -> None:
        """Save model config, weights, and optimizer state."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": asdict(self.model.config), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "WorldModelTrainer":
        """Restore a complete world-model training state."""
        from sc2wmrl.models.world_model import WorldModelConfig
        resolved_device = resolve_device(device); payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
        if payload.get("format_version") != 1: raise ValueError("unsupported world model checkpoint")
        trainer = cls(WorldModel(WorldModelConfig(**payload["config"])), device=str(resolved_device))
        trainer.model.load_state_dict(payload["model"]); trainer.optimizer.load_state_dict(payload["optimizer"])
        return trainer
