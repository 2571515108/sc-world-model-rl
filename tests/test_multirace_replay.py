"""Regression tests for masked multi-race replay data routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sc2wmrl.envs.base_macro_env import ACTION_COUNT
from sc2wmrl.replay.array_replay import ArrayReplay
from sc2wmrl.replay.batch_sampler import RaceBalancedArrayBatchSampler
from sc2wmrl.replays.multirace import MultiRaceFeatureExtractor, MultiRaceTransition, save_multirace_dataset
from sc2wmrl.replays.multirace_labels import RaceMacroAction, UniversalIntent
from sc2wmrl.training.world_model_trainer import array_batch


def _item(index: int, race: int, player_id: int) -> MultiRaceTransition:
    dimension = MultiRaceFeatureExtractor(200, 200).dimension
    observation = np.full(dimension, index / 10.0, dtype=np.float32)
    return MultiRaceTransition(observation, np.full(dimension, (index + 1) / 10.0, dtype=np.float32), np.ones(dimension, dtype=np.bool_), np.ones(dimension, dtype=np.bool_),
        int(UniversalIntent.TRAIN_ARMY), True, int(RaceMacroAction.TRAIN_BASIC_GROUND), True, 0.0, index == 2,
        np.zeros(7, dtype=np.float32), index * 32, race * 10, 0, player_id, race, 3 if race != 3 else 1, 0.98,
        8 if race == 1 else 0, race == 1, np.ones(ACTION_COUNT, dtype=np.bool_))


def test_multirace_features_mask_unobserved_enemy_and_keep_race_context() -> None:
    extractor = MultiRaceFeatureExtractor(100, 100)
    state = {"buildings": {}, "upgrades": [], "queue": [], "own_categories": {}, "enemy_categories": {}, "enemy_buildings": {},
             "army_center": (0.0, 0.0), "enemy_position": None, "enemy_seen": False}
    observation, valid = extractor.extract(state, "Terran", "Zerg")
    assert observation.shape == (112,) and valid.shape == (112,)
    enemy_index = extractor.names.index("enemy_observed_light_ground")
    assert observation[enemy_index] == 0 and not valid[enemy_index]
    assert observation[extractor.names.index("player_race_terran")] == 1
    assert observation[extractor.names.index("opponent_race_zerg")] == 1


def test_multirace_npz_keeps_training_labels_and_terran_route(tmp_path: Path) -> None:
    items = [_item(index, race, player_id) for race, player_id in ((1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (2, 2), (3, 1), (3, 1), (3, 1)) for index in range(3)]
    path = tmp_path / "multi.npz"; save_multirace_dataset(path, items, [])
    replay = ArrayReplay.load(path)
    assert replay.observation_shape == (112,)
    sampler = RaceBalancedArrayBatchSampler(replay, race_sampling={"Terran": 1, "Protoss": 1, "Zerg": 1}, seed=2)
    batch = sampler.sample(3, 2)
    assert set(batch["player_races"].tolist()) == {1, 2, 3}
    converted = array_batch(batch)
    assert converted["next_feature_valid_masks"].shape == (3, 2, 112)
    with np.load(path, allow_pickle=False) as arrays:
        valid = arrays["terran_macro_action_valid"].astype(bool)
        assert np.all(arrays["player_races"][valid] == 1)
