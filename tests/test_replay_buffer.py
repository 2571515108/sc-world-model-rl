"""Replay persistence and sequence-contiguity tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.sequence_sampler import SequenceSampler
from sc2wmrl.replay.transition import MacroTransition


def make_transition(index: int, episode: int = 0) -> MacroTransition:
    """Create a validated minimal transition for buffer tests."""
    mask = np.zeros(len(MacroAction), dtype=np.bool_); mask[MacroAction.NO_OP] = True
    return MacroTransition(observation=np.array([index, 1], dtype=np.float32), entity_observation=None, action=0, action_mask=mask,
        reward=0.1, terminated=False, truncated=False, next_observation=np.array([index + 1, 1], dtype=np.float32),
        opponent_id="bot", opponent_type="scripted", policy_version="v1", map_name="synthetic", game_loop=index * 32,
        events=np.zeros(7, dtype=np.float32), info={"index": index}, episode_id=episode)


class ReplayBufferTests(unittest.TestCase):
    """Validate storage, reload, capacity, and continuous sequence sampling."""

    def test_save_and_load_preserves_records(self) -> None:
        replay = ReplayBuffer(10, seed=3); replay.extend(make_transition(index) for index in range(4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"; replay.save(path); loaded = ReplayBuffer.load(path, seed=3)
        self.assertEqual(len(loaded), 4); self.assertTrue(np.array_equal(loaded.transitions()[2].observation, np.array([2, 1], dtype=np.float32)))

    def test_sequence_sampler_does_not_cross_episodes(self) -> None:
        replay = ReplayBuffer(10); replay.extend([make_transition(0, 0), make_transition(1, 0), make_transition(2, 0), make_transition(3, 1), make_transition(4, 1)])
        sequences = SequenceSampler(replay, seed=1).sample(batch_size=1, sequence_length=2)
        self.assertEqual(sequences[0][0].episode_id, sequences[0][1].episode_id)

    def test_rejects_action_illegal_under_its_mask(self) -> None:
        transition = make_transition(0); transition.action_mask[0] = False
        with self.assertRaises(ValueError): transition.__post_init__()
