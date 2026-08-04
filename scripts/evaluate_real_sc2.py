"""Evaluate a saved PPO policy through RealSC2MacroEnv."""

from __future__ import annotations

import argparse

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.envs.factory import build_macro_env
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.training.ppo_trainer import PPOTrainer
from sc2wmrl.utils.config import load_yaml


def main() -> None:
    """Run explicit real-game evaluation; never substitutes SyntheticMacroEnv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/real_sc2.yaml")
    parser.add_argument("--checkpoint", help="PPO checkpoint path overriding checkpoint_path in the YAML config")
    parser.add_argument("--episodes", type=int, help="Number of real SC2 episodes overriding the YAML config")
    parser.add_argument("--device", help="Torch device overriding device in the YAML config")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.checkpoint:
        config["checkpoint_path"] = args.checkpoint
    if args.episodes is not None:
        if args.episodes <= 0:
            raise ValueError("episodes must be positive")
        config["episodes"] = args.episodes
    if args.device:
        config["device"] = args.device
    env = build_macro_env(str(config["env_config"]))
    if env.__class__.__name__ != "RealSC2MacroEnv": raise ValueError("real evaluation config must select real_sc2")
    agent, _ = PPOAgent.load(config["checkpoint_path"], device=str(config.get("device", "auto")))
    try:
        print(PPOTrainer(env, agent, ReplayBuffer(1)).evaluate(int(config.get("episodes", 20)), seed=int(config.get("seed", 7))))
    finally:
        env.close()


if __name__ == "__main__":
    main()
