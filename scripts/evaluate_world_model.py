"""Evaluate open-loop reconstruction of a world-model checkpoint."""

from __future__ import annotations

import argparse

from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.training.world_model_trainer import WorldModelTrainer


def main() -> None:
    """Print horizon-wise open-loop errors for one stored model."""
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--replay", default="outputs/synthetic_replay.npz")
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--sequence-length", type=int, default=16); parser.add_argument("--device", default="auto"); args = parser.parse_args()
    replay = ReplayBuffer.load(args.replay); trainer = WorldModelTrainer.load(args.checkpoint, device=args.device)
    print(trainer.open_loop_evaluate(SequenceSampler(replay).sample(args.batch_size, args.sequence_length)))


if __name__ == "__main__": main()
