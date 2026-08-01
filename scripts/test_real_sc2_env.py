"""Manual opt-in smoke test for a local StarCraft II installation."""

from __future__ import annotations

import argparse

import numpy as np

from sc2wmrl.agents.rule_based_agent import RuleBasedAgent
from sc2wmrl.envs.factory import build_macro_env
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.transition import MacroTransition


def main() -> None:
    """Start SC2, run legal macro actions, persist a small real replay, and close."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/env/sc2.yaml")
    parser.add_argument("--steps", type=int, default=20); parser.add_argument("--output", default="outputs/real_sc2_smoke_replay.npz"); args = parser.parse_args()
    env = build_macro_env(args.config)
    if env.__class__.__name__ != "RealSC2MacroEnv": raise ValueError("smoke test requires environment.type: real_sc2")
    agent, replay = RuleBasedAgent(), ReplayBuffer(args.steps)
    observation, info = env.reset(); total_reward = 0.0
    try:
        for step in range(args.steps):
            mask = np.asarray(info["action_mask"], dtype=np.bool_); action = agent.act(observation, mask, info.get("raw_state"))
            next_observation, reward, terminated, truncated, next_info = env.step(action); total_reward += reward
            replay.append(MacroTransition(observation=observation, entity_observation=None, action=action, action_mask=mask, reward=reward, terminated=terminated, truncated=truncated, next_observation=next_observation, opponent_id=str(info.get("opponent_id", "builtin")), opponent_type=str(info.get("opponent_type", "builtin")), policy_version="rule_based-smoke", map_name=str(info.get("map_name", "unknown")), game_loop=int(next_info.get("game_loop", 0)), events=np.zeros(7, dtype=np.float32), info={"execution": next_info.get("execution", {}), "reward_components": next_info.get("reward_components", {})}, episode_id=0, next_action_mask=np.asarray(next_info["action_mask"], dtype=np.bool_), environment_type="real_sc2"))
            print({"step": step + 1, "game_loop": next_info.get("game_loop"), "workers": (next_info.get("raw_state") or {}).get("worker_count"), "army_value": (next_info.get("raw_state") or {}).get("army_value"), "reward": reward, "execution": next_info.get("execution")})
            observation, info = next_observation, next_info
            if terminated or truncated: break
        if len(replay): replay.save(args.output)
        print({"transitions": len(replay), "total_reward": total_reward, "output": args.output})
    finally:
        env.close()


if __name__ == "__main__":
    main()
