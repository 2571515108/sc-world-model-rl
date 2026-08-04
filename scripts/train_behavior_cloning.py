"""Pretrain a PPO actor on high-confidence expert replay macro labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.training.behavior_cloning_trainer import (
    BehaviorCloningConfig, BehaviorCloningTrainer, expert_batch_from_replay, split_expert_batch,
)
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed


def main() -> None:
    """Fit actor-only behavior cloning and persist a PPO-compatible checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/behavior_cloning_expert.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config); seed = int(config.get("seed", 7)); set_global_seed(seed)
    replay = ReplayBuffer.load(config["replay_path"], seed=seed)
    settings = {name: config[name] for name in BehaviorCloningConfig.__dataclass_fields__ if name in config}
    bc_config = BehaviorCloningConfig(**settings)
    resume = config.get("resume_checkpoint")
    if resume:
        agent, state = PPOAgent.load(resume, device=str(config.get("device", "auto")))
    else:
        if replay.observation_shape is None or len(replay.observation_shape) != 1:
            raise ValueError("expert replay must contain fixed one-dimensional observations")
        agent = PPOAgent(replay.observation_shape[0], PPOConfig(**load_yaml(config["ppo_config"])),
                         device=str(config.get("device", "auto")))
        state = {}
    batch = expert_batch_from_replay(replay, bc_config)
    train, validation = split_expert_batch(batch, bc_config.validation_fraction)
    history = BehaviorCloningTrainer(agent, bc_config).fit(train, validation)
    output = Path(config["checkpoint_path"])
    state = dict(state)
    state.update({"pretraining": "expert_behavior_cloning", "bc_epochs": bc_config.epochs, "bc_samples": len(train),
                  "bc_validation_samples": 0 if validation is None else len(validation), "source_replay": str(config["replay_path"])})
    agent.save(output, training_state=state)
    report = {"checkpoint": str(output), "train_samples": len(train), "validation_samples": 0 if validation is None else len(validation),
              "last_metrics": history[-1], "history": history}
    output.with_suffix(output.suffix + ".bc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({key: report[key] for key in ("checkpoint", "train_samples", "validation_samples", "last_metrics")})


if __name__ == "__main__":
    main()
