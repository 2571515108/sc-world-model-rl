"""Collect schema-compatible synthetic or real-SC2 macro trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sc2wmrl.agents.random_agent import RandomAgent
from sc2wmrl.agents.rule_based_agent import RuleBasedAgent
from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.envs.factory import build_macro_env
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.transition import MacroTransition
from sc2wmrl.utils.config import load_yaml


def _select_action(policy: str, random_agent: RandomAgent, rule_agent: RuleBasedAgent, checkpoint: PPOAgent | None,
                   observation: np.ndarray, mask: np.ndarray, raw_state: dict[str, object], rng: np.random.Generator,
                   probabilities: dict[str, float]) -> tuple[int, str]:
    """Select a policy while retaining its source label in replay metadata."""
    selected = policy
    if policy == "mixed":
        names = ("random", "rule_based", "checkpoint")
        weights = np.asarray([probabilities.get(name, 0.0) for name in names], dtype=np.float64)
        if weights.sum() <= 0: raise ValueError("mixed collector probabilities must sum to a positive value")
        selected = str(rng.choice(names, p=weights / weights.sum()))
    if selected == "random": return random_agent.act(observation, mask), "random"
    if selected == "rule_based": return rule_agent.act(observation, mask, raw_state), "rule_based"
    if selected == "checkpoint":
        if checkpoint is None: raise ValueError("checkpoint collector requires checkpoint_path")
        return checkpoint.act(observation, mask, deterministic=False), "checkpoint"
    raise ValueError(f"unknown collector policy {selected!r}")


def main() -> None:
    """Collect real or synthetic episodes based only on YAML configuration."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_yaml(args.config); collector = dict(config.get("collector", {})); env = build_macro_env(str(config["env_config"]))
    episodes = int(collector.get("episodes", 100)); seed = int(config.get("seed", 7)); output = Path(str(collector["replay_path"]))
    policy_name = str(collector.get("policy", "rule_based")); checkpoint_path = collector.get("checkpoint_path")
    checkpoint = PPOAgent.load(str(checkpoint_path), device=str(config.get("device", "auto")))[0] if checkpoint_path else None
    random_agent, rule_agent, rng = RandomAgent(seed), RuleBasedAgent(), np.random.default_rng(seed)
    replay = ReplayBuffer(episodes * int(getattr(getattr(env, "config", None), "max_macro_steps", getattr(env, "max_macro_steps", 256))), seed=seed)
    environment_type = "real_sc2" if env.__class__.__name__ == "RealSC2MacroEnv" else "synthetic"
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode); done = False
            while not done:
                mask = np.asarray(info["action_mask"], dtype=np.bool_); raw_state = dict(info.get("raw_state") or {})
                action, selected_policy = _select_action(policy_name, random_agent, rule_agent, checkpoint, observation, mask, raw_state, rng, {"random": float(collector.get("random_probability", 0.0)), "rule_based": float(collector.get("rule_based_probability", 1.0)), "checkpoint": float(collector.get("policy_probability", 0.0))})
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                replay.append(MacroTransition(observation=observation, entity_observation=None, action=action, action_mask=mask, reward=reward, terminated=terminated, truncated=truncated, next_observation=next_observation, opponent_id=str(info.get("opponent_id", "unknown")), opponent_type=str(info.get("opponent_type", "unknown")), policy_version=selected_policy, map_name=str(info.get("map_name", "synthetic")), game_loop=int(next_info.get("game_loop", info.get("game_loop", 0))), events=np.asarray(next_info.get("events", np.zeros(7)), dtype=np.float32), info={"reward_components": next_info.get("reward_components", {}), "execution": next_info.get("execution", {}), "environment_type": environment_type}, episode_id=episode, next_action_mask=np.asarray(next_info["action_mask"], dtype=np.bool_), opponent_action=int(next_info.get("opponent_action", 0)), environment_type=environment_type))
                observation, info, done = next_observation, next_info, bool(terminated or truncated)
    finally:
        env.close()
    replay.save(output); print({"episodes": episodes, "transitions": len(replay), "environment_type": environment_type, "output": str(output)})


if __name__ == "__main__":
    main()
