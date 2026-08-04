"""Inspect raw PySC2 state around rule-driven construction commands."""

from __future__ import annotations

import argparse

from sc2wmrl.agents.rule_based_agent import RuleBasedAgent
from sc2wmrl.envs.factory import build_macro_env


def main() -> None:
    """Print compact raw-unit evidence for each construction request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env/sc2.yaml")
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    env = build_macro_env(args.config)
    agent = RuleBasedAgent()
    try:
        observation, info = env.reset(seed=7)
        for step in range(args.steps):
            action = agent.act(observation, info["action_mask"], info.get("raw_state"))
            observation, _, terminated, truncated, info = env.step(action)
            execution = info.get("execution", {})
            if execution.get("issued_commands"):
                raw = env.backend._time_step.observation["raw_units"]
                units = [(int(unit["unit_type"]), round(float(unit["x"]), 1), round(float(unit["y"]), 1),
                          round(float(unit["build_progress"]), 2), int(unit["order_length"]))
                         for unit in raw if int(unit["alliance"]) == 1 and int(unit["unit_type"]) in (18, 19, 21, 45, 48)]
                print({"step": step + 1, "execution": execution, "minerals": info["raw_state"]["minerals"],
                       "buildings": info["raw_state"]["buildings"], "units": units})
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
