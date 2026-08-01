"""Collect legal, reloadable synthetic macro trajectories."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from sc2wmrl.agents.random_agent import RandomAgent
from sc2wmrl.envs.reward import RewardConfig
from sc2wmrl.envs.synthetic_macro_env import SyntheticEnvConfig, SyntheticMacroEnv
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.transition import MacroTransition
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed


def make_env(config_path: str) -> SyntheticMacroEnv:
    """Build the synthetic environment from its YAML configuration."""
    raw = load_yaml(config_path); reward = RewardConfig(**raw.pop("reward", {}))
    allowed = {field.name for field in fields(SyntheticEnvConfig)} - {"reward"}
    return SyntheticMacroEnv(SyntheticEnvConfig(**{key: value for key, value in raw.items() if key in allowed}, reward=reward))


def main() -> None:
    """Run collection and prove persistence via a load-after-save check."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/env/synthetic.yaml")
    parser.add_argument("--episodes", type=int, default=100); parser.add_argument("--output", default="outputs/synthetic_replay.npz")
    args = parser.parse_args()
    if args.episodes <= 0: raise ValueError("episodes must be positive")
    env = make_env(args.config); set_global_seed(env.config.seed); agent = RandomAgent(env.config.seed)
    replay = ReplayBuffer(args.episodes * env.config.max_macro_steps, seed=env.config.seed)
    for episode in range(args.episodes):
        observation, info = env.reset(seed=env.config.seed + episode); done = False
        while not done:
            mask = info["action_mask"]; action = agent.act(observation, mask)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            replay.append(MacroTransition(observation=observation, entity_observation=None, action=action, action_mask=mask, reward=reward,
                terminated=terminated, truncated=truncated, next_observation=next_observation, opponent_id=str(info["opponent_id"]),
                opponent_type=str(info["opponent_type"]), policy_version="random-control", map_name="synthetic", game_loop=int(info["game_loop"]),
                events=np.zeros(7, dtype=np.float32), info={"reward_components": next_info.get("reward_components", {}), "opponent_action": next_info.get("opponent_action", 0), "enemy_strategy": next_info.get("enemy_strategy", "unknown"), "environment_type": "synthetic"}, episode_id=episode,
                next_action_mask=np.asarray(next_info["action_mask"], dtype=np.bool_), opponent_action=int(next_info.get("opponent_action", 0)), environment_type="synthetic"))
            observation, info, done = next_observation, next_info, terminated or truncated
    output = Path(args.output); replay.save(output); loaded = ReplayBuffer.load(output, seed=env.config.seed)
    if len(loaded) != len(replay): raise RuntimeError("replay reload integrity check failed")
    print({"episodes": args.episodes, "transitions": len(replay), "observation_dim": env.observation_dim, "output": str(output)})


if __name__ == "__main__":
    main()
