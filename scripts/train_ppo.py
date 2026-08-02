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
    record_replay = bool(train_config.get("record_training_replay", False))
    replay = ReplayBuffer(max(2048, int(train_config["rollout_steps"]) * 2), seed=seed) if record_replay else None
    agent = PPOAgent(env.observation_dim, ppo_config, device=device)
    trainer = PPOTrainer(env, agent, replay); episodes = int(train_config["episodes"]); rollout_steps = int(train_config["rollout_steps"])
    updates = max(1, episodes * env.config.max_macro_steps // rollout_steps)
    best_return, best_win_rate = float("-inf"), float("-inf")
    evaluation_interval = int(train_config.get("evaluation_interval_steps", episodes * env.config.max_macro_steps))
    checkpoint_interval = int(train_config.get("checkpoint_interval_steps", evaluation_interval))
    with ExperimentLogger(run_dir, config=train_config, metadata={"random_seed": seed, "algorithm": "ppo"}) as logger:
        for update in range(updates):
            rollout = trainer.collect(rollout_steps, seed=seed + update * 1000); metrics = trainer.update(rollout)
            for name, value in metrics.items(): logger.log_scalar(name, value, step=(update + 1) * rollout_steps, episode=update + 1, phase="ppo")
            logger.log_scalar("environment_steps", (update + 1) * rollout_steps, step=(update + 1) * rollout_steps, episode=update + 1, phase="ppo")
            step = (update + 1) * rollout_steps
            if step % evaluation_interval < rollout_steps:
                current = trainer.evaluate(int(train_config["evaluation_episodes"]), seed=seed + 50_000 + update)
                for name, value in current.items(): logger.log_scalar(name, value, step=step, episode=update + 1, phase="evaluation")
                if current["mean_return"] >= best_return:
                    best_return = current["mean_return"]; agent.save(run_dir / "checkpoints" / "best_return.pt", training_state={"updates": update + 1, "evaluation": current})
                if current["win_rate"] >= best_win_rate:
                    best_win_rate = current["win_rate"]; agent.save(run_dir / "checkpoints" / "best_win_rate.pt", training_state={"updates": update + 1, "evaluation": current})
            if step % checkpoint_interval < rollout_steps:
                agent.save(run_dir / "checkpoints" / "latest.pt", training_state={"updates": update + 1})
            print({"update": update + 1, **metrics})
        evaluation = trainer.evaluate(int(train_config["evaluation_episodes"]), seed=seed + 50_000)
        for name, value in evaluation.items(): logger.log_scalar(name, value, step=updates * rollout_steps, episode=updates, phase="evaluation")
    checkpoint = Path(str(train_config["checkpoint_path"])); agent.save(checkpoint, training_state={"updates": updates, "evaluation": evaluation, "macro_action_names": [action.name for action in MacroAction]})
    agent.save(run_dir / "checkpoints" / "final.pt", training_state={"updates": updates, "evaluation": evaluation})
    restored, metadata = PPOAgent.load(checkpoint, device=device)
    if restored.observation_dim != env.observation_dim or metadata.get("updates") != updates: raise RuntimeError("checkpoint restore validation failed")
    print({"checkpoint": str(checkpoint), "run_dir": str(run_dir), "replay_size": len(replay) if replay is not None else 0, "evaluation": evaluation})


if __name__ == "__main__":
    main()
