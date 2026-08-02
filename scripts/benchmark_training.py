"""Measure replay indexing, sampling, PPO batch inference, and world-model updates."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.replay.array_replay import ArrayReplay
from sc2wmrl.replay.batch_sampler import ArrayBatchSampler
from sc2wmrl.training.world_model_trainer import WorldModelTrainer


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def main() -> None:
    """Print reproducible timing fields without making performance claims."""
    parser = argparse.ArgumentParser(); parser.add_argument("--replay", required=True); parser.add_argument("--world-model-checkpoint"); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--sequence-length", type=int, default=32); parser.add_argument("--burn-in-length", type=int, default=8); parser.add_argument("--iterations", type=int, default=20); parser.add_argument("--device", default="auto"); args = parser.parse_args()
    replay = ReplayBuffer.load(args.replay); sampler = SequenceSampler(replay, seed=7); total = args.sequence_length + args.burn_in_length
    begin = time.perf_counter(); sampler.sample(args.batch_size, args.sequence_length, args.burn_in_length); index_build_ms = (time.perf_counter() - begin) * 1000
    sample_times = []
    for _ in range(args.iterations):
        begin = time.perf_counter(); sampler.sample(args.batch_size, args.sequence_length, args.burn_in_length); sample_times.append(time.perf_counter() - begin)
    offline = ArrayReplay.load(args.replay); array_sampler = ArrayBatchSampler(offline, seed=7)
    begin = time.perf_counter(); array_sampler.sample(args.batch_size, args.sequence_length, args.burn_in_length); array_index_build_ms = (time.perf_counter() - begin) * 1000
    array_times = []
    for _ in range(args.iterations):
        begin = time.perf_counter(); array_sampler.sample(args.batch_size, args.sequence_length, args.burn_in_length); array_times.append(time.perf_counter() - begin)
    agent = PPOAgent(replay.observation_shape[0], device=args.device); observations = np.stack([replay[index].observation for index in range(min(len(replay), 4096))]); masks = np.stack([replay[index].action_mask for index in range(len(observations))])
    begin = time.perf_counter(); agent.act_batch(observations, masks); ppo_batch_ms = (time.perf_counter() - begin) * 1000
    result = {"replay_index_build_ms": index_build_ms, "replay_sample_median_ms": _median_ms(sample_times), "array_index_build_ms": array_index_build_ms, "array_sample_median_ms": _median_ms(array_times), "ppo_batch_inference_ms": ppo_batch_ms, "sampled_total_length": total, "transitions": len(replay)}
    if args.world_model_checkpoint:
        trainer = WorldModelTrainer.load(args.world_model_checkpoint, device=args.device); sequences = sampler.sample(args.batch_size, args.sequence_length, args.burn_in_length)
        begin = time.perf_counter(); trainer.train_step(sequences, burn_in_length=args.burn_in_length); result["world_model_train_step_ms"] = (time.perf_counter() - begin) * 1000
    print(result)


if __name__ == "__main__":
    main()
