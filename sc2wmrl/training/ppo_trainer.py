"""On-policy data collection, GAE, deterministic evaluation, and PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.envs.base_macro_env import MacroSC2Env
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.transition import MacroTransition


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite generalized advantages and bootstrapped returns."""
    if rewards.ndim != values.ndim != dones.ndim != 1 or not (len(rewards) == len(values) == len(dones)):
        raise ValueError("GAE inputs must be aligned one-dimensional arrays")
    advantages = np.zeros_like(rewards, dtype=np.float32); running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = last_value if index == len(rewards) - 1 else float(values[index + 1])
        running = float(rewards[index]) + gamma * (1.0 - float(dones[index])) * next_value - float(values[index]) + gamma * gae_lambda * (1.0 - float(dones[index])) * running
        advantages[index] = running
    returns = advantages + values
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise FloatingPointError("GAE produced non-finite values")
    return advantages, returns


@dataclass
class Rollout:
    """Typed on-policy rollout used for one PPO update."""

    observations: list[np.ndarray]; masks: list[np.ndarray]; actions: list[int]; log_probs: list[float]
    rewards: list[float]; values: list[float]; dones: list[bool]; last_observation: np.ndarray; last_mask: np.ndarray


class PPOTrainer:
    """Phase 1 trainer that keeps real trajectory storage and PPO data distinct."""

    def __init__(self, env: MacroSC2Env, agent: PPOAgent, replay: ReplayBuffer | None = None, *, policy_version: str = "ppo-v1") -> None:
        if env.observation_dim != agent.observation_dim:
            raise ValueError("environment and PPO observation dimensions differ")
        self.env, self.agent, self.replay, self.policy_version = env, agent, replay, policy_version
        self._episode_id = 0
        self._current_observation: np.ndarray | None = None
        self._current_info: dict[str, Any] | None = None

    def collect(self, rollout_steps: int, *, seed: int | None = None) -> Rollout:
        """Collect mask-respecting real transitions until an exact step budget is met."""
        if rollout_steps <= 0: raise ValueError("rollout steps must be positive")
        if self._current_observation is None or self._current_info is None:
            self._current_observation, self._current_info = self.env.reset(seed=seed)
        observation, info, episode_id = self._current_observation, self._current_info, self._episode_id
        records: dict[str, list[Any]] = {key: [] for key in ("observations", "masks", "actions", "log_probs", "rewards", "values", "dones")}
        for _ in range(rollout_steps):
            mask = np.asarray(info["action_mask"], dtype=np.bool_); action, log_prob, value = self.agent.act(observation, mask)
            next_observation, reward, terminated, truncated, next_info = self.env.step(action)
            if self.replay is not None: self.replay.append(MacroTransition(observation=observation, entity_observation=None, action=action, action_mask=mask, reward=reward,
                                               terminated=terminated, truncated=truncated, next_observation=next_observation,
                                               opponent_id=str(info.get("opponent_id", "unknown")), opponent_type=str(info.get("opponent_type", "unknown")),
                                               policy_version=self.policy_version, map_name=str(info.get("map_name", "synthetic")), game_loop=int(info.get("game_loop", 0)),
                                               events=np.zeros(7, dtype=np.float32), info={"reward_components": next_info.get("reward_components", {}), "opponent_action": next_info.get("opponent_action", 0), "enemy_strategy": next_info.get("enemy_strategy", "unknown")}, episode_id=episode_id,
                                               next_action_mask=np.asarray(next_info["action_mask"], dtype=np.bool_), opponent_action=int(next_info.get("opponent_action", 0)), environment_type=str(next_info.get("environment_type", "synthetic"))))
            for key, value_item in (("observations", observation), ("masks", mask), ("actions", action), ("log_probs", log_prob), ("rewards", reward), ("values", value), ("dones", terminated or truncated)):
                records[key].append(value_item)
            observation, info = next_observation, next_info
            if terminated or truncated:
                self._episode_id += 1; episode_id = self._episode_id; observation, info = self.env.reset()
        self._current_observation, self._current_info = observation, info
        return Rollout(**records, last_observation=observation, last_mask=np.asarray(info["action_mask"], dtype=np.bool_))

    def update(self, rollout: Rollout) -> dict[str, float]:
        """Compute GAE from rollout and update the agent."""
        _, _, last_value = self.agent.act(rollout.last_observation, rollout.last_mask, deterministic=True)
        advantages, returns = compute_gae(np.asarray(rollout.rewards, dtype=np.float32), np.asarray(rollout.values, dtype=np.float32), np.asarray(rollout.dones, dtype=np.bool_), last_value, self.agent.config.gamma, self.agent.config.gae_lambda)
        return self.agent.update({"observations": np.stack(rollout.observations), "action_masks": np.stack(rollout.masks), "actions": np.asarray(rollout.actions),
                                  "old_log_probs": np.asarray(rollout.log_probs), "advantages": advantages, "returns": returns})

    def evaluate(self, episodes: int, *, seed: int = 0) -> dict[str, float]:
        """Run deterministic policy evaluation with reproducible reset seeds."""
        if episodes <= 0: raise ValueError("episodes must be positive")
        returns, wins = [], 0
        for index in range(episodes):
            observation, info = self.env.reset(seed=seed + index); total, done = 0.0, False
            while not done:
                action, _, _ = self.agent.act(observation, info["action_mask"], deterministic=True)
                observation, reward, terminated, truncated, info = self.env.step(action); total += reward; done = terminated or truncated
            returns.append(total); wins += int(info.get("outcome") == "win")
        self._current_observation, self._current_info = None, None
        return {"mean_return": float(np.mean(returns)), "win_rate": wins / episodes, "episodes": float(episodes)}
