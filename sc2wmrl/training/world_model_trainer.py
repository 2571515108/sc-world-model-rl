"""Sequence replay pretraining and open-loop evaluation for the world model."""

from __future__ import annotations

import zlib
from contextlib import nullcontext
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
    next_observations = np.stack([[item.next_observation for item in sequence] for sequence in sequences])
    next_action_masks = np.stack([[item.next_action_mask for item in sequence] for sequence in sequences])
    actions = np.asarray([[item.action for item in sequence] for sequence in sequences], dtype=np.int64)
    rewards = np.asarray([[item.reward for item in sequence] for sequence in sequences], dtype=np.float32)
    continues = np.asarray([[not (item.terminated or item.truncated) for item in sequence] for sequence in sequences], dtype=np.float32)
    events = np.stack([[item.events for item in sequence] for sequence in sequences])
    opponent_actions = np.asarray([[int(item.opponent_action) for item in sequence] for sequence in sequences], dtype=np.int64)
    strategy_ids = {"rush": 0, "economy": 1, "defensive": 2, "ground_tech": 3, "air_tech": 4, "unknown": 5}
    strategies = np.asarray([strategy_ids.get(str(sequence[0].info.get("enemy_strategy", "unknown")).removesuffix("_bot"), 5) for sequence in sequences], dtype=np.int64)
    opponent_ids = np.asarray([zlib.crc32(sequence[0].opponent_id.encode()) % 128 for sequence in sequences], dtype=np.int64)
    return {"observations": torch.as_tensor(observations, device=device), "next_observations": torch.as_tensor(next_observations, device=device), "next_action_masks": torch.as_tensor(next_action_masks, device=device), "actions": torch.as_tensor(actions, device=device),
            "rewards": torch.as_tensor(rewards, device=device), "continues": torch.as_tensor(continues, device=device),
            "events": torch.as_tensor(events, device=device), "opponent_actions": torch.as_tensor(opponent_actions, device=device), "opponent_ids": torch.as_tensor(opponent_ids, device=device), "strategy_targets": torch.as_tensor(strategies, device=device)}


def array_batch(batch: dict[str, np.ndarray], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Move a vectorized replay batch to the training device once per update."""
    required = {"observations", "next_observations", "next_action_masks", "actions", "rewards", "continues", "events", "opponent_actions", "opponent_ids"}
    if not required.issubset(batch): raise ValueError("array replay batch is missing model fields")
    return {"observations": torch.as_tensor(batch["observations"], device=device, dtype=torch.float32), "next_observations": torch.as_tensor(batch["next_observations"], device=device, dtype=torch.float32),
            "next_action_masks": torch.as_tensor(batch["next_action_masks"], device=device, dtype=torch.bool), "actions": torch.as_tensor(batch["actions"], device=device, dtype=torch.long),
            "rewards": torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32), "continues": torch.as_tensor(batch["continues"], device=device, dtype=torch.float32),
            "events": torch.as_tensor(batch["events"], device=device, dtype=torch.float32), "opponent_actions": torch.as_tensor(batch["opponent_actions"], device=device, dtype=torch.long),
            "opponent_ids": torch.as_tensor(batch["opponent_ids"], device=device, dtype=torch.long), "strategy_targets": None}


class WorldModelTrainer:
    """Optimizes model sequences with clipping and checkpoint-safe state."""

    def __init__(self, model: WorldModel, learning_rate: float = 3e-4, max_grad_norm: float = 100.0, device: str = "auto", precision: str = "fp32") -> None:
        resolved_device = resolve_device(device)
        if precision not in {"fp32", "bf16-mixed", "fp16-mixed"}: raise ValueError("precision must be fp32, bf16-mixed, or fp16-mixed")
        self.model, self.device, self.max_grad_norm = model.to(resolved_device), str(resolved_device), max_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.precision = precision
        self._amp_enabled = precision != "fp32" and resolved_device.type == "cuda"
        self._amp_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled and precision == "fp16-mixed")

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._amp_enabled) if self._amp_enabled else nullcontext()

    def _step_optimizer(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm); self.scaler.step(self.optimizer); self.scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm); self.optimizer.step()

    def train_step(self, sequences: list[list[object]], *, burn_in_length: int = 0) -> dict[str, float]:
        """Run one finite, clipped optimization update."""
        batch = sequence_batch(sequences, self.device); self.model.train()
        if not torch.isfinite(batch["observations"]).all() or not torch.isfinite(batch["rewards"]).all():
            raise FloatingPointError("world-model batch contains NaN or Inf")
        with self._autocast(): loss: WorldModelLoss = self.model.loss(**batch, burn_in_length=burn_in_length)
        self._step_optimizer(loss.total)
        metrics = {item.name: float(getattr(loss, item.name).detach().item()) for item in fields(loss)}
        metrics["real_data_ratio"] = float(sum(sequence[0].environment_type == "real_sc2" for sequence in sequences) / len(sequences))
        metrics["synthetic_data_ratio"] = 1.0 - metrics["real_data_ratio"]
        return metrics

    @torch.no_grad()
    def validation_step(self, sequences: list[list[object]], *, burn_in_length: int = 0) -> dict[str, float]:
        """Evaluate held-out sequences without optimizer updates."""
        batch = sequence_batch(sequences, self.device); self.model.eval()
        with self._autocast(): loss: WorldModelLoss = self.model.loss(**batch, burn_in_length=burn_in_length)
        return {f"validation_{item.name}": float(getattr(loss, item.name).detach().item()) for item in fields(loss)}

    def train_array_batch(self, raw_batch: dict[str, np.ndarray], *, burn_in_length: int = 0) -> dict[str, float]:
        """Train directly from vectorized offline replay arrays."""
        batch = array_batch(raw_batch, self.device); self.model.train()
        with self._autocast(): loss: WorldModelLoss = self.model.loss(**batch, burn_in_length=burn_in_length)
        self._step_optimizer(loss.total)
        metrics = {item.name: float(getattr(loss, item.name).detach().item()) for item in fields(loss)}
        metrics["real_data_ratio"] = float(raw_batch.get("environment_is_real", np.zeros(1)).mean()); metrics["synthetic_data_ratio"] = 1.0 - metrics["real_data_ratio"]
        return metrics

    @torch.no_grad()
    def open_loop_evaluate(self, sequences: list[list[object]]) -> dict[str, float]:
        """Measure posterior reconstruction and prior multi-step observation drift."""
        batch = sequence_batch(sequences, self.device); self.model.eval()
        priors, posts, context, _ = self.model.observe(batch["observations"], batch["actions"], batch["opponent_ids"], next_observations=batch["next_observations"], sample=False)
        posterior_features = torch.stack([self.model.rssm.feature(state) for state in posts], 1)
        reconstruction_mse = torch.mean((self.model.observation_head(posterior_features) - batch["next_observations"]) ** 2).item()
        state = self.model.rssm.initial_posterior(self.model.observation_encoder(batch["observations"][:, 0]), sample=False); predictions = []
        for time in range(batch["observations"].shape[1]):
            state = self.model.rssm.prior(state, batch["actions"][:, time], context, sample=False)
            predictions.append(self.model.observation_head(self.model.rssm.feature(state)))
        open_loop = torch.stack(predictions, 1)
        horizon_errors = ((open_loop - batch["next_observations"]) ** 2).mean(dim=(0, 2))
        return {"reconstruction_mse": float(reconstruction_mse), **{f"open_loop_mse_h{index + 1}": float(error) for index, error in enumerate(horizon_errors)}}

    def fit(self, sampler: SequenceSampler, updates: int, batch_size: int, sequence_length: int, burn_in_length: int = 0) -> list[dict[str, float]]:
        """Train a requested number of replay batches."""
        if updates <= 0: raise ValueError("updates must be positive")
        return [self.train_step(sampler.sample(batch_size, sequence_length, burn_in_length), burn_in_length=burn_in_length) for _ in range(updates)]

    def save(self, path: str | Path) -> None:
        """Save model config, weights, and optimizer state."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 2, "config": asdict(self.model.config), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "precision": self.precision}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "WorldModelTrainer":
        """Restore a complete world-model training state."""
        from sc2wmrl.models.world_model import WorldModelConfig
        resolved_device = resolve_device(device); payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
        if payload.get("format_version") != 2: raise ValueError("world-model checkpoint format 1 is not transition-aligned; retrain with format 2")
        trainer = cls(WorldModel(WorldModelConfig(**payload["config"])), device=str(resolved_device), precision=str(payload.get("precision", "fp32")))
        trainer.model.load_state_dict(payload["model"]); trainer.optimizer.load_state_dict(payload["optimizer"])
        return trainer
