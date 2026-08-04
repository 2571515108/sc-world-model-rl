"""Regression tests for expert replay conversion and actor-only pretraining."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from collections import namedtuple

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig, _torch
from sc2wmrl.envs.base_macro_env import ACTION_COUNT, MacroAction
from sc2wmrl.models.world_model import WorldModel, WorldModelConfig
from sc2wmrl.replay.replay_buffer import ReplayBuffer
from sc2wmrl.replay.transition import MacroTransition
from sc2wmrl.replays.macro_labeler import AbilityNameResolver, ReplayMacroLabeler
from sc2wmrl.replays.metadata import inspect_replay_metadata, select_expert_players
from sc2wmrl.replays.extractor import _align_next_action_masks
from sc2wmrl.replays.metadata import ReplayMetadata
from sc2wmrl.replays.versioning import exact_replay_version, prepare_replay_version, replay_executable_path
from sc2wmrl.training.behavior_cloning_trainer import (
    BehaviorCloningConfig, BehaviorCloningTrainer, expert_batch_from_replay,
)


def _transition(index: int, action: MacroAction) -> MacroTransition:
    observation = np.zeros(4, dtype=np.float32); observation[index % 4] = 1.0
    return MacroTransition(
        observation=observation, entity_observation=None, action=int(action), action_mask=np.ones(ACTION_COUNT, dtype=np.bool_),
        reward=0.0, terminated=False, truncated=False, next_observation=observation.copy(), opponent_id="human",
        opponent_type="human_replay", policy_version="expert", map_name="map", game_loop=index * 32,
        events=np.zeros(7, dtype=np.float32), info={"expert_label": True, "label_confidence": 1.0},
        episode_id=index // 4, opponent_action_valid=False, environment_type="real_sc2",
    )


def test_metadata_inspection_selects_terran_winner(tmp_path: Path, monkeypatch) -> None:
    replay = tmp_path / "sample.SC2Replay"; replay.write_bytes(b"fixture")
    payload = {"Title": "Map", "GameVersion": "5.0.16.97563", "BaseBuild": "Base97563", "Duration": 100,
               "Players": [{"PlayerID": 1, "APM": 200, "Result": "Loss", "SelectedRace": "Zerg", "AssignedRace": "Zerg"},
                           {"PlayerID": 2, "APM": 450, "Result": "Win", "SelectedRace": "Terr", "AssignedRace": "Terr"}]}
    class Archive:
        def __init__(self, _: str) -> None: pass
        def read_file(self, _: object) -> bytes: return json.dumps(payload).encode()
        def close(self) -> None: pass
    monkeypatch.setitem(sys.modules, "mpyq", types.SimpleNamespace(MPQArchive=Archive))
    metadata = inspect_replay_metadata(replay)
    selected = select_expert_players(metadata, target_race="Terran", winners_only=True, min_apm=300)
    assert metadata.base_build == 97563
    assert [player.player_id for player in selected] == [2]


def test_labeler_maps_barracks_ability() -> None:
    class Command:
        ability_id = 321
        unit_tags: list[int] = []
    class Raw:
        unit_command = Command()
        def HasField(self, field: str) -> bool: return field == "unit_command"
    action = types.SimpleNamespace(action_raw=Raw(), action_feature_layer=None, action_render=None)
    label = ReplayMacroLabeler(AbilityNameResolver({321: ["Build_Barracks_pt"]})).infer([action], {}, {}, {})
    assert label.action == MacroAction.BUILD_BARRACKS
    assert label.confidence > 0.9


def test_behavior_cloning_never_updates_value_head() -> None:
    replay = ReplayBuffer(16)
    replay.extend(_transition(index, MacroAction.TRAIN_WORKERS if index % 2 == 0 else MacroAction.BUILD_SUPPLY) for index in range(12))
    config = BehaviorCloningConfig(epochs=2, batch_size=4, validation_fraction=0.0, entropy_coefficient=0.0)
    agent = PPOAgent(4, PPOConfig(hidden_dim=16, epochs=1, minibatch_size=4), device="cpu")
    before = {name: value.detach().clone() for name, value in agent.value_head.state_dict().items()}
    history = BehaviorCloningTrainer(agent, config).fit(expert_batch_from_replay(replay, config))
    assert len(history) == 2 and np.isfinite(history[-1]["bc_loss"])
    for name, value in agent.value_head.state_dict().items():
        assert _torch().equal(value, before[name])


def test_world_model_ignores_unknown_opponent_actions() -> None:
    model = WorldModel(WorldModelConfig(observation_dim=3, deterministic_dim=8, stochastic_dim=4, hidden_dim=16, ensemble_size=2))
    torch = _torch()
    loss = model.loss(
        observations=torch.zeros(2, 3, 3), next_observations=torch.zeros(2, 3, 3),
        actions=torch.zeros(2, 3, dtype=torch.long), rewards=torch.zeros(2, 3), continues=torch.ones(2, 3),
        events=torch.zeros(2, 3, 7), opponent_actions=torch.zeros(2, 3, dtype=torch.long),
        opponent_action_valid=torch.zeros(2, 3, dtype=torch.bool),
    )
    assert float(loss.opponent.detach()) == 0.0


def test_replay_persists_unknown_opponent_action_mask(tmp_path: Path) -> None:
    replay = ReplayBuffer(2); replay.append(_transition(0, MacroAction.BUILD_BARRACKS))
    path = tmp_path / "expert.npz"; replay.save(path)
    restored = ReplayBuffer.load(path)
    assert restored[0].opponent_action_valid is False


def test_corrected_expert_mask_becomes_previous_next_mask() -> None:
    transitions = [_transition(0, MacroAction.BUILD_BARRACKS), _transition(1, MacroAction.TRAIN_BASIC_ARMY)]
    corrected = np.zeros(ACTION_COUNT, dtype=np.bool_)
    corrected[[int(MacroAction.NO_OP), int(MacroAction.TRAIN_BASIC_ARMY)]] = True
    transitions[1].action_mask = corrected
    _align_next_action_masks(transitions)
    assert np.array_equal(transitions[0].next_action_mask, corrected)


def test_exact_replay_version_bypasses_static_version_lookup() -> None:
    Version = namedtuple("Version", "game_version build_version data_version binary")
    metadata = ReplayMetadata("replay.SC2Replay", "Map", "5.0.16.97563", 97563, 97563, "HASH", 1, False, ())
    version = exact_replay_version(types.SimpleNamespace(get_replay_version=lambda _: Version("5.0.16", 97563, "HASH", None)), b"replay", metadata)
    assert version == Version("5.0.16", 97563, "HASH", "replay")
    assert replay_executable_path("D:/SC2", 97563).as_posix().endswith("Versions/Base97563/SC2_x64.exe")


def test_prepare_replay_version_sends_download_data_request(tmp_path: Path) -> None:
    replay = tmp_path / "sample.SC2Replay"; replay.write_bytes(b"replay")
    metadata = ReplayMetadata(str(replay), "Map", "5.0.16.97563", 97563, 97563, "HASH", 1, False, ())
    sent: list[object] = []
    class Controller:
        def __init__(self) -> None:
            self._client = types.SimpleNamespace(send=lambda **kwargs: sent.append(kwargs["replay_info"]) or types.SimpleNamespace(error=0))
        def __enter__(self): return self
        def __exit__(self, *_): return None
    config = types.SimpleNamespace(data_dir=str(tmp_path), replay_data=lambda _: b"replay", start=lambda **_: Controller())
    report = prepare_replay_version(replay, metadata, run_configs=types.SimpleNamespace(get=lambda: config),
                                    sc_pb=types.SimpleNamespace(RequestReplayInfo=lambda **kwargs: types.SimpleNamespace(**kwargs)))
    assert report.download_requested and not report.executable_present_after and report.replay_info_error is None
    assert sent[0].download_data is True and sent[0].replay_data == b"replay"
