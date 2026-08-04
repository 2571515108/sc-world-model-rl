"""Train one PPO policy with curriculum real/synthetic rollouts and imagination."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.envs.factory import build_macro_env
from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.critic import LatentCritic
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import MixedSequenceSampler, SequenceSampler
from sc2wmrl.training.hybrid_trainer import HybridConfig, HybridTrainer
from sc2wmrl.training.imagination_trainer import ImaginationConfig, ImaginationTrainer
from sc2wmrl.training.ppo_trainer import PPOTrainer
from sc2wmrl.training.world_model_trainer import WorldModelTrainer
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed
from sc2wmrl.reporting.experiment_logger import ExperimentLogger, create_run_directory


def _curriculum_real_ratio(config: dict[str, object], update: int) -> float:
    """Linearly move from cheap synthetic practice to real SC2 experience."""
    start = float(config.get("real_ratio_start", 1.0))
    end = float(config.get("real_ratio_end", start))
    warmup = max(1, int(config.get("real_ratio_warmup_updates", 1)))
    return max(0.0, min(1.0, start + (end - start) * min(1.0, update / warmup)))


def _mixed_sampler(config: dict[str, object], seed: int):
    """Build the imagination source without merging incompatible episode IDs."""
    synthetic_path, real_path = config.get("synthetic_replay_path"), config.get("real_replay_path")
    if synthetic_path and real_path:
        synthetic = ReplayBuffer.load(str(synthetic_path), seed=seed)
        real = ReplayBuffer.load(str(real_path), seed=seed + 1)
        if synthetic.observation_shape != real.observation_shape:
            raise ValueError("synthetic and real replay observation shapes must match")
        return MixedSequenceSampler(SequenceSampler(synthetic, seed=seed), SequenceSampler(real, seed=seed + 1),
                                    synthetic_ratio=float(config.get("imagination_synthetic_ratio", 0.3)))
    if config.get("replay_path"):
        return SequenceSampler(ReplayBuffer.load(str(config["replay_path"]), seed=seed), seed=seed)
    raise ValueError("hybrid training requires synthetic_replay_path and real_replay_path, or a legacy replay_path")


def main() -> None:
    """Run curriculum PPO and uncertainty-gated mixed-world-model imagination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/hybrid.yaml")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed, device = int(config["seed"]), str(config.get("device", "auto"))
    set_global_seed(seed)

    policy_path = str(config.get("resume_checkpoint") or config["policy_checkpoint"])
    policy, resume_state = PPOAgent.load(policy_path, device=device)
    # A model-free PPO checkpoint has no hybrid curriculum history.  Starting
    # its hybrid warmup at the PPO update count would immediately enable the
    # maximum imagination weight with a randomly initialized latent actor.
    completed_updates = int(resume_state.get("hybrid_updates", 0))

    synthetic_path = config.get("synthetic_env_config")
    real_path = config.get("real_env_config")
    if not synthetic_path and not real_path:
        # Preserve the original single-environment configuration contract.
        selected = str(config["env_config"])
        if "sc2" in Path(selected).stem.lower():
            real_path = selected
        else:
            synthetic_path = selected
    environments = {}
    if synthetic_path:
        environments["synthetic"] = build_macro_env(str(synthetic_path))
    if real_path:
        environments["real"] = build_macro_env(str(real_path))
    if not environments:
        raise ValueError("hybrid training requires at least one environment")
    if any(env.observation_dim != policy.observation_dim for env in environments.values()):
        raise ValueError("policy checkpoint observation dimension does not match an environment")

    trainers = {name: PPOTrainer(env, policy, policy_version=f"hybrid-{name}") for name, env in environments.items()}
    primary = trainers.get("real") or trainers["synthetic"]
    world_model = WorldModelTrainer.load(str(config["world_model_checkpoint"]), device=device).model
    feature_dim = world_model.config.deterministic_dim + world_model.config.stochastic_dim
    imagination = ImaginationTrainer(world_model, LatentActor(feature_dim, primary.env.action_dim, world_model.config.hidden_dim),
                                     LatentCritic(feature_dim, world_model.config.hidden_dim),
                                     ImaginationConfig(horizon=int(config["imagination_horizon"]),
                                                       uncertainty_threshold=float(config["uncertainty_threshold"])))
    sampler = _mixed_sampler(config, seed)
    hybrid = HybridTrainer(primary, imagination, sampler, HybridConfig(
        imagination_weight_max=float(config["imagination_weight_max"]),
        imagination_warmup_updates=int(config["imagination_warmup_updates"]),
        imagined_batch_size=int(config["imagined_batch_size"]), imagined_sequence_length=int(config["imagined_sequence_length"])))
    hybrid.updates = completed_updates
    rollout_steps, updates = int(config["rollout_steps"]), int(config["updates"])
    if rollout_steps <= 0 or updates <= 0:
        raise ValueError("rollout_steps and updates must be positive")
    run_dir = Path(args.run_dir) if args.run_dir else create_run_directory("outputs/runs", "hybrid", seed)

    try:
        with ExperimentLogger(run_dir, config=config, metadata={"random_seed": seed, "algorithm": "hybrid_curriculum"}) as logger:
            for local_update in range(updates):
                global_update = completed_updates + local_update
                ratio = _curriculum_real_ratio(config, global_update)
                real_steps = int(round(rollout_steps * ratio)) if "real" in trainers else 0
                synthetic_steps = rollout_steps - real_steps if "synthetic" in trainers else 0
                if "real" in trainers and "synthetic" not in trainers:
                    real_steps = rollout_steps
                if "synthetic" in trainers and "real" not in trainers:
                    synthetic_steps = rollout_steps
                entries = []
                if synthetic_steps:
                    entries.append(("synthetic", trainers["synthetic"], trainers["synthetic"].collect(synthetic_steps, seed=seed + global_update * 10_000)))
                if real_steps:
                    entries.append(("real", trainers["real"], trainers["real"].collect(real_steps, seed=seed + global_update * 10_000 + 1)))
                metrics = hybrid.update(entries)
                metrics["curriculum_real_ratio"] = ratio
                total_steps = (global_update + 1) * rollout_steps
                for name, value in metrics.items():
                    logger.log_scalar(name, value, step=total_steps, episode=global_update + 1, phase="hybrid")
                print({"update": global_update + 1, **metrics})
        final_updates = completed_updates + updates
        state = {"hybrid_updates": final_updates, "world_model_checkpoint": str(config["world_model_checkpoint"])}
        policy.save(config["checkpoint_path"], training_state=state)
        policy.save(run_dir / "checkpoints" / "final.pt", training_state=state)
        print({"checkpoint": config["checkpoint_path"], "run_dir": str(run_dir), "updates": final_updates})
    finally:
        for env in environments.values():
            env.close()


if __name__ == "__main__":
    main()
