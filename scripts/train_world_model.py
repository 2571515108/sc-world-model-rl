"""Pretrain the RSSM world model on a persisted real/synthetic replay buffer."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.envs.synthetic_macro_env import SyntheticMacroEnv
from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.training.world_model_trainer import WorldModelTrainer
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed
from sc2wmrl.reporting.experiment_logger import ExperimentLogger, create_run_directory


def main() -> None:
    """Fit, evaluate, and persist an RSSM model without claiming experiment results."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train/world_model.yaml"); parser.add_argument("--run-dir"); args = parser.parse_args()
    config = load_yaml(args.config); set_global_seed(int(config.get("seed", 7)))
    replay = ReplayBuffer.load(config["replay_path"], seed=int(config.get("seed", 7))); model_data = load_yaml(config["model_config"])
    model = WorldModel(WorldModelConfig(observation_dim=replay.observation_shape[0], **model_data))
    run_dir = Path(args.run_dir) if args.run_dir else create_run_directory("outputs/runs", "world_model", int(config.get("seed", 7)))
    trainer = WorldModelTrainer(model, learning_rate=float(config["learning_rate"]), device=str(config.get("device", "auto"))); sampler = SequenceSampler(replay, seed=int(config.get("seed", 7)))
    with ExperimentLogger(run_dir, config=config, metadata={"random_seed": int(config.get("seed", 7)), "algorithm": "world_model"}) as logger:
        metrics = []
        for update in range(int(config["updates"])):
            item = trainer.train_step(sampler.sample(int(config["batch_size"]), int(config["sequence_length"]), int(config.get("burn_in_length", 0)))); metrics.append(item)
            for name, value in item.items(): logger.log_scalar(name if name != "total" else "total_world_model_loss", value, step=update + 1, episode=0, phase="world_model")
    evaluation = trainer.open_loop_evaluate(sampler.sample(int(config["batch_size"]), int(config["sequence_length"])))
    trainer.save(config["checkpoint_path"]); trainer.save(run_dir / "checkpoints" / "final.pt"); print({"last_loss": metrics[-1], "open_loop": evaluation, "checkpoint": config["checkpoint_path"], "run_dir": str(run_dir)})


if __name__ == "__main__": main()
