"""Train and checkpoint the Phase 1 model-free PPO baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig
from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.training.ppo_trainer import PPOTrainer
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed
from sc2wmrl.reporting.experiment_logger import ExperimentLogger, create_run_directory
from scripts.collect_synthetic import make_env


def main() -> None:
    """Train on the configured synthetic environment and save a resumable checkpoint."""
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train/model_free.yaml"); parser.add_argument("--run-dir")
    args = parser.parse_args(); train_config = load_yaml(args.config); seed = int(train_config.get("seed", 7)); set_global_seed(seed)
    run_dir = Path(args.run_dir) if args.run_dir else create_run_directory("outputs/runs", "ppo", seed)
    env = make_env(str(train_config["env_config"])); ppo_config = PPOConfig(**load_yaml(str(train_config["ppo_config"])))
    device = str(train_config.get("device", "auto"))
    agent = PPOAgent(env.observation_dim, ppo_config, device=device); replay = ReplayBuffer(max(2048, int(train_config["rollout_steps"]) * 2), seed=seed)
    trainer = PPOTrainer(env, agent, replay); episodes = int(train_config["episodes"]); rollout_steps = int(train_config["rollout_steps"])
    updates = max(1, episodes * env.config.max_macro_steps // rollout_steps)
    with ExperimentLogger(run_dir, config=train_config, metadata={"random_seed": seed, "algorithm": "ppo"}) as logger:
        for update in range(updates):
            rollout = trainer.collect(rollout_steps, seed=seed + update * 1000); metrics = trainer.update(rollout)
            for name, value in metrics.items(): logger.log_scalar(name, value, step=(update + 1) * rollout_steps, episode=update + 1, phase="ppo")
            logger.log_scalar("environment_steps", (update + 1) * rollout_steps, step=(update + 1) * rollout_steps, episode=update + 1, phase="ppo")
            print({"update": update + 1, **metrics})
        evaluation = trainer.evaluate(int(train_config["evaluation_episodes"]), seed=seed + 50_000)
        for name, value in evaluation.items(): logger.log_scalar(name, value, step=updates * rollout_steps, episode=updates, phase="evaluation")
    checkpoint = Path(str(train_config["checkpoint_path"])); agent.save(checkpoint, training_state={"updates": updates, "evaluation": evaluation, "macro_action_names": [action.name for action in MacroAction]})
    agent.save(run_dir / "checkpoints" / "final.pt", training_state={"updates": updates, "evaluation": evaluation})
    restored, metadata = PPOAgent.load(checkpoint, device=device)
    if restored.observation_dim != env.observation_dim or metadata.get("updates") != updates: raise RuntimeError("checkpoint restore validation failed")
    print({"checkpoint": str(checkpoint), "run_dir": str(run_dir), "replay_size": len(replay), "evaluation": evaluation})


if __name__ == "__main__":
    main()
