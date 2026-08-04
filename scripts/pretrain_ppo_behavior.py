"""Bootstrap the existing PPO policy by behavior cloning validated trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig, _torch
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.utils.config import load_yaml
from sc2wmrl.utils.seed import set_global_seed


def main() -> None:
    """Imitate balanced rule trajectories before sparse-reward online PPO."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/ppo_behavior_bootstrap.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed, device = int(config.get("seed", 7)), str(config.get("device", "auto"))
    set_global_seed(seed)
    paths = [str(path) for path in config["replay_paths"]]
    transitions = [item for path in paths for item in ReplayBuffer.load(path, seed=seed).transitions()]
    if not transitions:
        raise ValueError("behavior cloning requires at least one replay transition")
    observation_dim = transitions[0].observation.shape[0]
    if any(item.observation.shape != (observation_dim,) for item in transitions):
        raise ValueError("replay observations must share one fixed shape")
    agent = PPOAgent(observation_dim, PPOConfig(**load_yaml(config["ppo_config"])), device=device)
    observations = np.stack([item.observation for item in transitions]).astype(np.float32)
    masks = np.stack([item.action_mask for item in transitions]).astype(np.bool_)
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    counts = np.bincount(actions, minlength=masks.shape[1]).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights /= class_weights[actions].mean()
    torch = _torch()
    generator = np.random.default_rng(seed)
    batch_size, epochs = int(config.get("batch_size", 256)), int(config.get("epochs", 20))
    if batch_size <= 0 or epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    for epoch in range(epochs):
        order = generator.permutation(len(actions)); losses = []; correct = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            action_tensor = torch.as_tensor(actions[indices], device=agent.device)
            distribution, _ = agent._distribution(torch.as_tensor(observations[indices], device=agent.device),
                                                  torch.as_tensor(masks[indices], device=agent.device))
            per_item = -distribution.log_prob(action_tensor)
            weights = torch.as_tensor(class_weights[actions[indices]], device=agent.device)
            loss = (per_item * weights).mean()
            agent.optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.config.max_grad_norm); agent.optimizer.step()
            losses.append(float(loss.item())); correct += int((distribution.probs.argmax(-1) == action_tensor).sum().item())
        print({"epoch": epoch + 1, "behavior_loss": float(np.mean(losses)), "balanced_action_accuracy": correct / len(actions)})
    output = Path(str(config["checkpoint_path"]))
    agent.save(output, training_state={"behavior_cloning_epochs": epochs, "source_replays": paths})
    print({"checkpoint": str(output), "transitions": len(transitions)})


if __name__ == "__main__":
    main()
