"""Deterministically evaluate a saved PPO policy in the synthetic environment."""

from __future__ import annotations

import argparse

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.training.ppo_trainer import PPOTrainer
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from scripts.collect_synthetic import make_env


def main() -> None:
    """Load a checkpoint and print reproducible fixed-opponent policy metrics."""
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", default="configs/env/synthetic.yaml"); parser.add_argument("--episodes", type=int, default=32); parser.add_argument("--device", default="auto")
    args = parser.parse_args(); env = make_env(args.env_config); agent, _ = PPOAgent.load(args.checkpoint, device=args.device)
    print(PPOTrainer(env, agent, ReplayBuffer(1)).evaluate(args.episodes))


if __name__ == "__main__":
    main()
