"""CPU-only Phase 0 smoke test independent of PyTorch."""

import unittest

from sc2wmrl.agents.random_agent import RandomAgent
from sc2wmrl.envs.synthetic_macro_env import SyntheticMacroEnv


class SyntheticSmokeTests(unittest.TestCase):
    """Exercise a complete synthetic episode through legal actions."""

    def test_complete_episode_without_invalid_observations(self) -> None:
        env = SyntheticMacroEnv(); agent = RandomAgent(8); observation, info = env.reset(seed=8); total = 0.0
        while True:
            action = agent.act(observation, info["action_mask"]); observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated: break
        self.assertLessEqual(info["step"], env.config.max_macro_steps); self.assertTrue(-2.0 <= total <= 2.0)
