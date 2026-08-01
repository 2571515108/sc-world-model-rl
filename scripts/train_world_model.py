"""Pretrain the RSSM world model on a persisted real/synthetic replay buffer."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.envs.synthetic_macro_env import SyntheticMacroEnv
from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import MixedSequenceSampler, SequenceSampler, split_replay_by_episode
from sc2wmrl.training.world_model_trainer import WorldModelTrainer
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed
from sc2wmrl.reporting.experiment_logger import ExperimentLogger, create_run_directory


def main() -> None:
    """Fit, evaluate, and persist an RSSM model without claiming experiment results."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train/world_model.yaml"); parser.add_argument("--run-dir"); args = parser.parse_args()
    config = load_yaml(args.config); set_global_seed(int(config.get("seed", 7)))
    mode = str(config.get("dataset_mode", "synthetic_only")); seed = int(config.get("seed", 7))
    validation_fraction = float(config.get("validation_fraction", 0.0)); validation_sampler = None
    if mode == "mixed":
        synthetic = ReplayBuffer.load(config["synthetic_replay_path"], seed=seed)
        real = ReplayBuffer.load(config["real_replay_path"], seed=seed + 1)
        if synthetic.observation_shape != real.observation_shape: raise ValueError("mixed replay observation shapes must match")
        replay = real; ratio = float(config.get("synthetic_ratio", 0.2))
        train_synthetic, validation_synthetic = split_replay_by_episode(synthetic, validation_fraction)
        train_real, validation_real = split_replay_by_episode(real, validation_fraction)
        sampler = MixedSequenceSampler(SequenceSampler(train_synthetic, seed=seed), SequenceSampler(train_real, seed=seed + 1), synthetic_ratio=ratio)
        if validation_synthetic is not None and validation_real is not None:
            validation_sampler = MixedSequenceSampler(SequenceSampler(validation_synthetic, seed=seed + 2), SequenceSampler(validation_real, seed=seed + 3), synthetic_ratio=ratio)
    else:
        replay = ReplayBuffer.load(config["replay_path"], seed=seed)
        expected = "real_sc2" if mode == "real_only" else "synthetic"
        if any(item.environment_type != expected for item in replay.transitions()): raise ValueError(f"{mode} configuration received non-{expected} transitions")
        train_replay, validation_replay = split_replay_by_episode(replay, validation_fraction)
        sampler = SequenceSampler(train_replay, seed=seed)
        if validation_replay is not None: validation_sampler = SequenceSampler(validation_replay, seed=seed + 1)
    model_data = load_yaml(config["model_config"])
    model = WorldModel(WorldModelConfig(observation_dim=replay.observation_shape[0], **model_data))
    run_dir = Path(args.run_dir) if args.run_dir else create_run_directory("outputs/runs", "world_model", int(config.get("seed", 7)))
    resume = config.get("resume_checkpoint")
    trainer = WorldModelTrainer.load(resume, device=str(config.get("device", "auto"))) if resume else WorldModelTrainer(model, learning_rate=float(config["learning_rate"]), max_grad_norm=float(config.get("max_grad_norm", 100.0)), device=str(config.get("device", "auto")))
    with ExperimentLogger(run_dir, config=config, metadata={"random_seed": int(config.get("seed", 7)), "algorithm": "world_model"}) as logger:
        metrics = []
        validation_interval = int(config.get("validation_interval", 100))
        for update in range(int(config["updates"])):
            item = trainer.train_step(sampler.sample(int(config["batch_size"]), int(config["sequence_length"]), int(config.get("burn_in_length", 0)))); metrics.append(item)
            for name, value in item.items(): logger.log_scalar(name if name != "total" else "total_world_model_loss", value, step=update + 1, episode=0, phase="world_model")
            if validation_sampler is not None and (update + 1) % validation_interval == 0:
                validation = trainer.validation_step(validation_sampler.sample(int(config["batch_size"]), int(config["sequence_length"]), int(config.get("burn_in_length", 0))))
                for name, value in validation.items(): logger.log_scalar(name, value, step=update + 1, episode=0, phase="validation")
    evaluation = trainer.open_loop_evaluate(sampler.sample(int(config["batch_size"]), int(config["sequence_length"])))
    trainer.save(config["checkpoint_path"]); trainer.save(run_dir / "checkpoints" / "final.pt"); print({"last_loss": metrics[-1], "open_loop": evaluation, "checkpoint": config["checkpoint_path"], "run_dir": str(run_dir)})


if __name__ == "__main__": main()
