"""Regression coverage for replay indexing, aligned world-model losses, and batching."""

import unittest

import numpy as np
import torch

from sc2wmrl.agents.ppo_agent import PPOAgent
from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.replay.transition import MacroTransition
from sc2wmrl.training.imagination_trainer import ImaginationConfig, ImaginationTrainer
from sc2wmrl.models.actor import LatentActor
from sc2wmrl.models.critic import LatentCritic


def transition(index: int, episode: int = 0) -> MacroTransition:
    """Create one contiguous transition with every action-mask field present."""
    mask = np.zeros(len(MacroAction), dtype=np.bool_); mask[MacroAction.NO_OP] = True
    return MacroTransition(np.array([index, 0], np.float32), None, 0, mask, 0.0, False, False, np.array([index + 1, 0], np.float32), "bot", "scripted", "test", "map", index + 1, np.zeros(7, np.float32), episode_id=episode, next_action_mask=mask)


class TrainingEfficiencyTests(unittest.TestCase):
    """Verify optimization changes preserve legality and training gradients."""

    def test_sampler_builds_index_once_until_replay_mutates(self) -> None:
        replay = ReplayBuffer(16); replay.extend(transition(index) for index in range(8)); sampler = SequenceSampler(replay, seed=1)
        sampler.sample(2, 3); sampler.sample(2, 3); self.assertEqual(sampler.index_builds, 1)
        replay.append(transition(8)); sampler.sample(2, 3); self.assertEqual(sampler.index_builds, 2)

    def test_burn_in_masks_prefix_and_ensemble_gets_gradients(self) -> None:
        model = WorldModel(WorldModelConfig(observation_dim=2, hidden_dim=16, deterministic_dim=8, stochastic_dim=4, opponent_embedding_dim=4, ensemble_size=3))
        observations = torch.rand(3, 5, 2); next_observations = torch.rand(3, 5, 2); actions = torch.zeros(3, 5, dtype=torch.long)
        masks = torch.zeros(3, 5, len(MacroAction)); masks[..., 0] = 1
        loss = model.loss(observations, next_observations, actions, torch.zeros(3, 5), torch.ones(3, 5), torch.zeros(3, 5, 7), next_action_masks=masks, burn_in_length=2)
        loss.total.backward(); self.assertTrue(torch.isfinite(loss.total)); self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.ensemble.parameters()))

    def test_act_batch_returns_only_legal_actions(self) -> None:
        agent = PPOAgent(3, device="cpu"); observations = np.zeros((6, 3), np.float32); masks = np.zeros((6, len(MacroAction)), np.bool_); masks[:, MacroAction.NO_OP] = True
        actions, log_probs, values = agent.act_batch(observations, masks)
        self.assertEqual(actions.shape, (6,)); self.assertTrue(np.all(actions == MacroAction.NO_OP)); self.assertEqual(log_probs.shape, values.shape)

    def test_imagination_freezes_world_model_and_updates_masks(self) -> None:
        model = WorldModel(WorldModelConfig(observation_dim=2, hidden_dim=16, deterministic_dim=8, stochastic_dim=4, opponent_embedding_dim=4, ensemble_size=2))
        feature = model.config.deterministic_dim + model.config.stochastic_dim
        trainer = ImaginationTrainer(model, LatentActor(feature, len(MacroAction), 16), LatentCritic(feature, 16), ImaginationConfig(horizon=2))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        observations = torch.rand(2, 3, 2); actions = torch.zeros(2, 3, dtype=torch.long); masks = torch.zeros(2, 3, len(MacroAction), dtype=torch.bool); masks[..., 0] = True
        rollout = trainer.rollout(observations, actions, masks, next_observations=torch.rand(2, 3, 2)); self.assertGreaterEqual(rollout["actions"].shape[1], 1)

