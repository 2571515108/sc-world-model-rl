"""Run real PPO plus uncertainty-gated imagined actor-critic updates."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.critic import LatentCritic
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.training.hybrid_trainer import HybridConfig, HybridTrainer
from sc2wmrl.training.imagination_trainer import ImaginationConfig, ImaginationTrainer
from sc2wmrl.training.ppo_trainer import PPOTrainer
from sc2wmrl.training.world_model_trainer import WorldModelTrainer
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed
from sc2wmrl.reporting.experiment_logger import ExperimentLogger, create_run_directory
from scripts.collect_synthetic import make_env


def main() -> None:
    """Load validated model components and train a hybrid policy on synthetic data."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train/hybrid.yaml"); parser.add_argument("--run-dir"); args = parser.parse_args(); config = load_yaml(args.config)
    device = str(config.get("device", "auto"))
    set_global_seed(int(config["seed"])); env = make_env(config["env_config"]); policy, _ = PPOAgent.load(config["policy_checkpoint"], device=device)
    replay = ReplayBuffer.load(config["replay_path"], seed=int(config["seed"])); ppo = PPOTrainer(env, policy, replay, policy_version="hybrid-v1")
    world_model = WorldModelTrainer.load(config["world_model_checkpoint"], device=device).model; feature_dim = world_model.config.deterministic_dim + world_model.config.stochastic_dim
    imagination = ImaginationTrainer(world_model, LatentActor(feature_dim, env.action_dim, world_model.config.hidden_dim), LatentCritic(feature_dim, world_model.config.hidden_dim),
        ImaginationConfig(horizon=int(config["imagination_horizon"]), uncertainty_threshold=float(config["uncertainty_threshold"])))
    hybrid = HybridTrainer(ppo, imagination, SequenceSampler(replay, seed=int(config["seed"])), HybridConfig(
        imagination_weight_max=float(config["imagination_weight_max"]), imagination_warmup_updates=int(config["imagination_warmup_updates"]),
        imagined_batch_size=int(config["imagined_batch_size"]), imagined_sequence_length=int(config["imagined_sequence_length"])))
    run_dir = Path(args.run_dir) if args.run_dir else create_run_directory("outputs/runs", "hybrid", int(config["seed"]))
    with ExperimentLogger(run_dir, config=config, metadata={"random_seed": int(config["seed"]), "algorithm": "hybrid"}) as logger:
        for update in range(int(config["updates"])):
            metrics = hybrid.update(ppo.collect(int(config["rollout_steps"]), seed=int(config["seed"]) + update))
            for name, value in metrics.items(): logger.log_scalar(name, value, step=(update + 1) * int(config["rollout_steps"]), episode=update + 1, phase="hybrid")
            print({"update": update + 1, **metrics})
    policy.save(config["checkpoint_path"], training_state={"hybrid_updates": hybrid.updates}); policy.save(run_dir / "checkpoints" / "final.pt", training_state={"hybrid_updates": hybrid.updates}); print({"checkpoint": config["checkpoint_path"], "run_dir": str(run_dir)})


if __name__ == "__main__": main()
