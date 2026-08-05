"""Pretrain only the Terran PPO actor from a shared P/T/Z replay dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig
from sc2wmrl.training.behavior_cloning_trainer import BehaviorCloningConfig, BehaviorCloningTrainer, ExpertBatch, split_expert_batch
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed


def main() -> None:
    """Route high-confidence Terran labels to the Terran actor and nothing else."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train/behavior_cloning_multirace_terran.yaml"); args = parser.parse_args()
    config = load_yaml(args.config); set_global_seed(int(config.get("seed", 7)))
    path = Path(config["replay_path"]); metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 3:
        raise ValueError("Terran multi-race pretraining requires a v3 multi-race dataset")
    with np.load(path, allow_pickle=False) as arrays:
        minimum = float(config.get("minimum_confidence", 0.70))
        selected = (arrays["player_races"] == 1) & arrays["terran_macro_action_valid"].astype(bool) & (arrays["label_confidences"] >= minimum)
        if not bool(config.get("include_no_op", False)):
            selected &= arrays["terran_macro_actions"] != 0
        if not selected.any():
            raise ValueError("the multi-race dataset has no selected Terran actor labels")
        batch = ExpertBatch(arrays["observations"][selected].astype(np.float32), arrays["terran_action_masks"][selected].astype(np.bool_),
                           arrays["terran_macro_actions"][selected].astype(np.int64), arrays["label_confidences"][selected].astype(np.float32), arrays["episode_ids"][selected].astype(np.int64))
    bc_config = BehaviorCloningConfig(learning_rate=float(config.get("learning_rate", 3e-4)), batch_size=int(config.get("batch_size", 256)),
        epochs=int(config.get("epochs", 30)), validation_fraction=float(config.get("validation_fraction", 0.1)), minimum_confidence=minimum,
        include_no_op=bool(config.get("include_no_op", False)), class_balance_power=float(config.get("class_balance_power", 0.5)),
        entropy_coefficient=float(config.get("entropy_coefficient", 0.001)), max_grad_norm=float(config.get("max_grad_norm", 1.0)), seed=int(config.get("seed", 7)))
    train, validation = split_expert_batch(batch, bc_config.validation_fraction)
    agent = PPOAgent(batch.observations.shape[1], PPOConfig(**load_yaml(config["ppo_config"])), device=str(config.get("device", "auto")))
    history = BehaviorCloningTrainer(agent, bc_config).fit(train, validation)
    output = Path(config["checkpoint_path"]); agent.save(output, training_state={"dataset": str(path), "selected_terran_labels": len(batch), "epochs": bc_config.epochs, "dataset_schema": metadata.get("schema")})
    print({"checkpoint": str(output), "selected_terran_labels": len(batch), "last": history[-1]})


if __name__ == "__main__":
    main()
