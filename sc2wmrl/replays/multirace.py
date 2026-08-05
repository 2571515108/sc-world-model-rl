"""Multi-view P/T/Z replay extraction with a shared, masked feature schema."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sc2wmrl.envs.action_mask import action_mask
from sc2wmrl.envs.base_macro_env import ACTION_COUNT, MacroAction
from sc2wmrl.envs.pysc2_backend import PySC2StateAdapter
from sc2wmrl.envs.unit_semantics import (BUILDING_CATEGORIES, COMBAT_CATEGORIES, SUPPLY_IDS, TOWNHALL_IDS, WORKER_IDS,
                                         building_category, combat_category, normalise_race, race_id, unit_value)
from sc2wmrl.replays.extractor import (ReplayExtractionConfig, _map_data, _map_size, _matching_run_config, _require_sc2,
                                       _start_replay, configure_sc2_path)
from sc2wmrl.replays.macro_labeler import ReplayMacroLabeler
from sc2wmrl.replays.metadata import ReplayMetadata, ReplayPlayer, inspect_replay_metadata
from sc2wmrl.replays.multirace_labels import (MultiRaceReplayLabeler, RaceMacroAction, UniversalIntent,
                                               universal_event_vector)
from sc2wmrl.replays.observation import controller_observation_dict, player_outcome, raw_unit_index


MULTIRACE_SCHEMA = "sc2wmrl.multirace.features.v1"
BUILDING_FEATURES = ("base", "supply", "basic_production", "tech_production", "gas", "tech", "defense", "addon")
ENEMY_BUILDING_FEATURES = ("base", "production", "ground_tech", "air_tech")
STRATEGY_FEATURES = ("rush", "economy", "defensive", "ground_tech", "air_tech", "unknown")
REGIONS = ("own_main", "own_natural", "map_center", "enemy_natural", "enemy_main", "left_route", "right_route", "expansion_regions")


def multirace_feature_names() -> tuple[str, ...]:
    """Return the fixed, race-conditioned feature names used by v1 datasets."""
    names = ["game_time", "minerals", "vespene", "current_supply", "maximum_supply", "worker_count", "base_count",
             "idle_worker_count", "army_value", "lost_army_value", "enemy_army_value_estimate"]
    names += [f"building_{name}" for name in BUILDING_FEATURES]
    names += [f"upgrade_{index}" for index in range(6)]
    names += [f"queue_{index}" for index in range(5)]
    names += [f"own_{name}" for name in COMBAT_CATEGORIES]
    names += ["average_army_health", "army_center_x", "army_center_y", "number_of_army_groups"]
    names += [f"enemy_observed_{name}" for name in COMBAT_CATEGORIES]
    names += [f"enemy_building_{name}" for name in ENEMY_BUILDING_FEATURES]
    names += ["enemy_position_x", "enemy_position_y", "enemy_position_present", "time_since_last_scout",
              "estimated_enemy_army_value", "estimated_enemy_worker_count"]
    names += [f"enemy_strategy_{name}" for name in STRATEGY_FEATURES]
    for region in REGIONS:
        names += [f"{region}_{name}" for name in ("friendly_power", "visible_enemy_power", "visibility", "control_score", "last_scout_time")]
    names += ["player_race_terran", "player_race_protoss", "player_race_zerg", "opponent_race_terran", "opponent_race_protoss", "opponent_race_zerg"]
    return tuple(names)


def _field(value: Any, name: str, default: Any = 0) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _ratio(value: Any, scale: float) -> float:
    return float(np.clip(float(value) / scale, 0.0, 1.0))


def _log(value: Any, scale: float) -> float:
    return float(np.clip(np.log1p(max(0.0, float(value))) / np.log1p(scale), 0.0, 1.0))


class MultiRaceStateAdapter:
    """Build a canonical state from one player's fog-of-war observation only."""

    SELF, ENEMY = 1, 4

    def __init__(self, map_width: float, map_height: float) -> None:
        self.map_width, self.map_height = float(map_width), float(map_height)
        self.reset()

    def reset(self) -> None:
        self._enemy_counts: Counter[int] = Counter()
        self._enemy_buildings: Counter[str] = Counter()
        self._enemy_position: tuple[float, float] | None = None
        self._last_seen_loop: int | None = None
        self._previous_own: Counter[int] = Counter()
        self._lost_army_value = 0.0

    @staticmethod
    def _center(units: list[Any]) -> tuple[float, float]:
        if not units:
            return 0.0, 0.0
        return float(np.mean([float(_field(unit, "x")) for unit in units])), float(np.mean([float(_field(unit, "y")) for unit in units]))

    def extract(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        raw = list(observation.get("raw_units", ()))
        own = [unit for unit in raw if int(_field(unit, "alliance")) == self.SELF]
        enemy = [unit for unit in raw if int(_field(unit, "alliance")) == self.ENEMY]
        loop_values = np.asarray(observation.get("game_loop", [0])).reshape(-1)
        game_loop = int(loop_values[0]) if loop_values.size else 0
        if enemy:
            self._enemy_counts = Counter(int(_field(unit, "unit_type")) for unit in enemy)
            self._enemy_buildings = Counter(category for unit in enemy if (category := building_category(int(_field(unit, "unit_type")))) is not None)
            points = [(float(_field(unit, "x")), float(_field(unit, "y"))) for unit in enemy]
            self._enemy_position = (float(np.mean([point[0] for point in points])), float(np.mean([point[1] for point in points])))
            self._last_seen_loop = game_loop
        counts = Counter(int(_field(unit, "unit_type")) for unit in own)
        army_counts = Counter(int(_field(unit, "unit_type")) for unit in own if combat_category(int(_field(unit, "unit_type"))) is not None)
        for unit_type, previous in self._previous_own.items():
            self._lost_army_value += max(0, previous - army_counts.get(unit_type, 0)) * unit_value(unit_type)
        self._previous_own = army_counts
        player = np.asarray(observation.get("player", ()), dtype=np.float32)
        value_at = lambda index: float(player[index]) if player.size > index else 0.0
        own_categories = Counter(combat_category(unit_type) for unit_type, count in counts.items() for _ in range(count) if combat_category(unit_type) is not None)
        enemy_categories = Counter(combat_category(unit_type) for unit_type, count in self._enemy_counts.items() for _ in range(count) if combat_category(unit_type) is not None)
        army = [unit for unit in own if combat_category(int(_field(unit, "unit_type"))) is not None]
        buildings = Counter(category for unit_type, count in counts.items() for category in [building_category(unit_type)] if category is not None for _ in range(count))
        orders = [int(_field(unit, "order_length")) for unit in own if building_category(int(_field(unit, "unit_type"))) is not None]
        enemy_value = sum(unit_value(unit_type) * count for unit_type, count in self._enemy_counts.items())
        return {
            "game_time": game_loop / 22.4, "minerals": value_at(1), "vespene": value_at(2), "current_supply": value_at(3),
            "maximum_supply": value_at(4), "worker_count": sum(counts[value] for value in WORKER_IDS),
            "base_count": sum(counts[value] for value in TOWNHALL_IDS), "idle_worker_count": value_at(7),
            "army_value": sum(unit_value(int(_field(unit, "unit_type"))) for unit in army), "lost_army_value": self._lost_army_value,
            "enemy_army_value_estimate": enemy_value, "buildings": {name: int(buildings[name]) for name in BUILDING_FEATURES},
            "upgrades": np.asarray(observation.get("upgrades", ()), dtype=np.int32), "queue": orders[:5],
            "own_categories": own_categories, "enemy_categories": enemy_categories,
            "average_army_health": 1.0 if not army else float(np.mean([float(_field(unit, "health")) / max(1.0, float(_field(unit, "health_max", 1))) for unit in army])),
            "army_center": self._center(army), "number_of_army_groups": int(bool(army)), "enemy_buildings": self._enemy_buildings,
            "enemy_position": self._enemy_position, "time_since_last_scout": 3600.0 if self._last_seen_loop is None else max(0.0, (game_loop - self._last_seen_loop) / 22.4),
            "estimated_enemy_worker_count": float(sum(self._enemy_counts[value] for value in WORKER_IDS)), "enemy_seen": self._last_seen_loop is not None,
        }


class MultiRaceFeatureExtractor:
    """Encode semantic P/T/Z observations and an explicit value-validity mask."""

    names = multirace_feature_names()

    def __init__(self, map_width: float, map_height: float) -> None:
        self.map_width, self.map_height = float(map_width), float(map_height)

    @property
    def dimension(self) -> int:
        return len(self.names)

    def extract(self, state: Mapping[str, Any], player_race: str, opponent_race: str) -> tuple[np.ndarray, np.ndarray]:
        values: list[float] = []; valid: list[bool] = []
        def add(value: float, is_valid: bool = True) -> None:
            values.append(float(value)); valid.append(bool(is_valid))
        for value, scale, logarithmic in ((state.get("game_time", 0), 3600, True), (state.get("minerals", 0), 4000, True),
                                           (state.get("vespene", 0), 3000, True), (state.get("current_supply", 0), 200, False),
                                           (state.get("maximum_supply", 0), 200, False), (state.get("worker_count", 0), 80, False),
                                           (state.get("base_count", 0), 8, False), (state.get("idle_worker_count", 0), 80, False),
                                           (state.get("army_value", 0), 12000, True), (state.get("lost_army_value", 0), 12000, True),
                                           (state.get("enemy_army_value_estimate", 0), 12000, True)):
            add(_log(value, scale) if logarithmic else _ratio(value, scale), value is not None)
        buildings = state.get("buildings", {})
        for name in BUILDING_FEATURES:
            add(_ratio(buildings.get(name, 0), 20))
        upgrades = np.asarray(state.get("upgrades", ()), dtype=np.int32)
        for index in range(6):
            add(float(index < len(upgrades)), bool(len(upgrades)))
        queues = list(state.get("queue", ()))
        for index in range(5):
            add(_ratio(queues[index] if index < len(queues) else 0, 20), True)
        own = state.get("own_categories", {})
        for name in COMBAT_CATEGORIES:
            add(_ratio(own.get(name, 0), 200))
        center = state.get("army_center", (0.0, 0.0))
        add(_ratio(state.get("average_army_health", 0), 1)); add(_ratio(center[0], self.map_width)); add(_ratio(center[1], self.map_height)); add(_ratio(state.get("number_of_army_groups", 0), 10))
        enemy_seen = bool(state.get("enemy_seen", False)); enemy = state.get("enemy_categories", {})
        for name in COMBAT_CATEGORIES:
            add(_ratio(enemy.get(name, 0), 200), enemy_seen)
        enemy_buildings = state.get("enemy_buildings", {})
        building_map = {"base": enemy_buildings.get("base", 0), "production": enemy_buildings.get("basic_production", 0) + enemy_buildings.get("tech_production", 0),
                        "ground_tech": enemy_buildings.get("tech", 0), "air_tech": enemy_buildings.get("tech_production", 0)}
        for name in ENEMY_BUILDING_FEATURES:
            add(_ratio(building_map[name], 20), enemy_seen)
        position = state.get("enemy_position")
        add(0.0 if position is None else _ratio(position[0], self.map_width), enemy_seen)
        add(0.0 if position is None else _ratio(position[1], self.map_height), enemy_seen)
        add(float(position is not None), enemy_seen)
        add(_ratio(state.get("time_since_last_scout", 3600), 3600), enemy_seen)
        add(_log(state.get("enemy_army_value_estimate", 0), 12000), enemy_seen)
        add(_ratio(state.get("estimated_enemy_worker_count", 0), 80), enemy_seen)
        # No strategy classifier is available during a replay observation.
        # Preserve zeros and mark them invalid rather than inventing "unknown" labels.
        for _ in STRATEGY_FEATURES:
            add(0.0, False)
        for _region in REGIONS:
            for _field_name in range(5):
                add(0.0, False)
        own_id, opponent_id = race_id(player_race), race_id(opponent_race)
        for identifier in (1, 2, 3):
            add(float(own_id == identifier), own_id != 0)
        for identifier in (1, 2, 3):
            add(float(opponent_id == identifier), opponent_id != 0)
        array, mask = np.asarray(values, dtype=np.float32), np.asarray(valid, dtype=np.bool_)
        if array.shape != (self.dimension,) or mask.shape != (self.dimension,) or not np.isfinite(array).all():
            raise ValueError("invalid multi-race feature extraction")
        return array, mask


@dataclass(frozen=True)
class MultiRaceExtractionSummary:
    replay_path: str
    player_id: int
    player_race: str
    opponent_race: str
    steps: int
    valid_universal_labels: int
    valid_terran_actor_labels: int
    outcome: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiRaceTransition:
    observation: np.ndarray; next_observation: np.ndarray; feature_valid_mask: np.ndarray; next_feature_valid_mask: np.ndarray
    action: int; universal_intent_valid: bool; race_action: int; race_action_valid: bool; reward: float; terminated: bool
    events: np.ndarray; game_loop: int; episode_id: int; replay_id: int; player_id: int; player_race: int; opponent_race: int
    label_confidence: float; terran_action: int = 0; terran_action_valid: bool = False
    terran_action_mask: np.ndarray = field(default_factory=lambda: np.ones(ACTION_COUNT, dtype=np.bool_))
    source: str = "none"; evidence: tuple[str, ...] = (); ability_ids: tuple[int, ...] = ()
    map_name: str = "unknown"


def _townhall_position(response: Any) -> tuple[float, float] | None:
    for unit in response.observation.raw_data.units:
        if int(unit.alliance) == 1 and int(unit.unit_type) in TOWNHALL_IDS:
            return float(unit.pos.x), float(unit.pos.y)
    return None


def _enemy_start(game_info: Any, own_main: tuple[float, float] | None) -> tuple[float, float] | None:
    locations = [(float(point.x), float(point.y)) for point in getattr(getattr(game_info, "start_raw", None), "start_locations", ())]
    if not locations:
        return None
    if own_main is None:
        return locations[0]
    return max(locations, key=lambda point: (point[0] - own_main[0]) ** 2 + (point[1] - own_main[1]) ** 2)


def extract_multirace_viewpoint(path: str | Path, player: ReplayPlayer, *, episode_id: int, replay_id: int,
                                config: ReplayExtractionConfig, metadata: ReplayMetadata | None = None) -> tuple[list[MultiRaceTransition], MultiRaceExtractionSummary]:
    """Extract a single player's local replay view; never concatenate both POVs."""
    replay_path = Path(path).expanduser().resolve(); metadata = metadata or inspect_replay_metadata(replay_path)
    opponents = [candidate for candidate in metadata.players if candidate.player_id != player.player_id]
    opponent_race = opponents[0].assigned_race if opponents else "Unknown"
    configure_sc2_path(config.sc2_path)
    run_configs, replay_lib, sc_pb = _require_sc2(); default_config = run_configs.get(); replay_data = default_config.replay_data(str(replay_path))
    run_config = _matching_run_config(run_configs, replay_lib, sc_pb, replay_path, replay_data, metadata, download_missing_version=config.download_missing_version)
    transitions: list[MultiRaceTransition] = []; labeler = MultiRaceReplayLabeler(); terran_labeler = ReplayMacroLabeler()
    valid_labels = 0; valid_terran = 0; terminal_outcome: str | None = None
    with run_config.start(want_rgb=False) as controller:
        request = sc_pb.RequestStartReplay(replay_data=replay_data, options=sc_pb.InterfaceOptions(raw=True, score=True, show_cloaked=True, show_burrowed_shadows=True), disable_fog=False, observed_player_id=int(player.player_id))
        _start_replay(controller, request, _map_data(run_config, config.map_file))
        game_info = controller.game_info(); width, height = _map_size(game_info, config)
        adapter = MultiRaceStateAdapter(width, height); extractor = MultiRaceFeatureExtractor(width, height)
        terran_adapter = PySC2StateAdapter(width, height) if normalise_race(player.assigned_race) == "Terran" else None
        previous_response = controller.observe(); previous_input = controller_observation_dict(previous_response); previous_state = adapter.extract(previous_input)
        previous_terran = terran_adapter.extract(previous_input, {}) if terran_adapter is not None else None
        own_main = _townhall_position(previous_response); enemy_main = _enemy_start(game_info, own_main)
        while not previous_response.player_result:
            controller.step(config.step_mul); current_response = controller.observe(); current_input = controller_observation_dict(current_response); current_state = adapter.extract(current_input)
            label = labeler.infer(current_response.actions, previous_state, current_state, own_main=own_main, enemy_main=enemy_main)
            observation, feature_mask = extractor.extract(previous_state, player.assigned_race, opponent_race)
            next_observation, next_feature_mask = extractor.extract(current_state, player.assigned_race, opponent_race)
            outcome = player_outcome(current_response, player.player_id); terminal_outcome = outcome or terminal_outcome
            reward = 1.0 if outcome == "win" else -1.0 if outcome == "loss" else -0.1 if outcome == "draw" else 0.0
            terran_action, terran_valid, terran_mask = 0, False, np.ones(ACTION_COUNT, dtype=np.bool_)
            if terran_adapter is not None and previous_terran is not None:
                current_terran = terran_adapter.extract(current_input, {})
                terran = terran_labeler.infer(current_response.actions, previous_terran, current_terran, raw_unit_index(previous_response), own_main=own_main, enemy_main=enemy_main)
                terran_mask = action_mask(previous_terran)
                terran_mask = terran_mask.copy(); terran_mask[int(terran.action)] = True
                terran_action = int(terran.action); terran_valid = bool(terran.confidence >= config.minimum_label_confidence and terran.action != MacroAction.NO_OP)
                valid_terran += int(terran_valid); previous_terran = current_terran
            valid_labels += int(label.valid and label.confidence >= config.minimum_label_confidence and label.universal_intent != UniversalIntent.NO_OP)
            transitions.append(MultiRaceTransition(observation, next_observation, feature_mask, next_feature_mask, int(label.universal_intent),
                bool(label.valid and label.confidence >= config.minimum_label_confidence), int(label.race_action), bool(label.valid and label.confidence >= config.minimum_label_confidence), reward, outcome is not None,
                universal_event_vector(label.universal_intent, outcome), int(previous_response.observation.game_loop), episode_id, replay_id, player.player_id,
                race_id(player.assigned_race), race_id(opponent_race), float(label.confidence), terran_action, terran_valid, terran_mask, label.source, label.evidence, label.ability_ids, metadata.title))
            previous_response, previous_state = current_response, current_state
            if outcome is not None:
                break
    return transitions, MultiRaceExtractionSummary(str(replay_path), player.player_id, normalise_race(player.assigned_race), normalise_race(opponent_race), len(transitions), valid_labels, valid_terran, terminal_outcome)


def save_multirace_dataset(path: str | Path, transitions: list[MultiRaceTransition], summaries: list[MultiRaceExtractionSummary]) -> None:
    """Persist every training-critical label and mask in the NPZ, not JSON only."""
    if not transitions:
        raise ValueError("cannot save an empty multi-race dataset")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    paired = np.full(len(transitions), -1, dtype=np.int64); by_key: dict[tuple[int, int], list[int]] = {}
    for index, item in enumerate(transitions):
        by_key.setdefault((item.replay_id, item.game_loop), []).append(index)
    for indexes in by_key.values():
        if len(indexes) == 2:
            paired[indexes[0]], paired[indexes[1]] = indexes[1], indexes[0]
    arrays = {
        "observations": np.stack([item.observation for item in transitions]), "next_observations": np.stack([item.next_observation for item in transitions]),
        "feature_valid_masks": np.stack([item.feature_valid_mask for item in transitions]), "next_feature_valid_masks": np.stack([item.next_feature_valid_mask for item in transitions]),
        "actions": np.asarray([item.action for item in transitions], dtype=np.int64), "universal_intents": np.asarray([item.action for item in transitions], dtype=np.int64),
        "universal_intent_valid": np.asarray([item.universal_intent_valid for item in transitions], dtype=np.bool_), "race_actions": np.asarray([item.race_action for item in transitions], dtype=np.int64),
        "race_action_valid": np.asarray([item.race_action_valid for item in transitions], dtype=np.bool_), "label_confidences": np.asarray([item.label_confidence for item in transitions], dtype=np.float32),
        "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32), "terminated": np.asarray([item.terminated for item in transitions], dtype=np.bool_), "truncated": np.zeros(len(transitions), dtype=np.bool_),
        "events": np.stack([item.events for item in transitions]), "action_masks": np.ones((len(transitions), ACTION_COUNT), dtype=np.bool_), "next_action_masks": np.ones((len(transitions), ACTION_COUNT), dtype=np.bool_),
        "next_action_mask_valid": np.zeros(len(transitions), dtype=np.bool_), "opponent_actions": np.zeros(len(transitions), dtype=np.int64), "opponent_action_valid": np.zeros(len(transitions), dtype=np.bool_),
        "episode_ids": np.asarray([item.episode_id for item in transitions], dtype=np.int64), "game_loops": np.asarray([item.game_loop for item in transitions], dtype=np.int64),
        "replay_ids": np.asarray([item.replay_id for item in transitions], dtype=np.int64), "player_ids": np.asarray([item.player_id for item in transitions], dtype=np.int8),
        "map_names": np.asarray([item.map_name for item in transitions]),
        "player_races": np.asarray([item.player_race for item in transitions], dtype=np.int8), "opponent_races": np.asarray([item.opponent_race for item in transitions], dtype=np.int8),
        "paired_view_indices": paired, "terran_macro_actions": np.asarray([item.terran_action for item in transitions], dtype=np.int64),
        "terran_macro_action_valid": np.asarray([item.terran_action_valid for item in transitions], dtype=np.bool_), "terran_action_masks": np.stack([item.terran_action_mask for item in transitions]),
    }
    np.savez_compressed(path, **arrays)
    metadata = {"format_version": 3, "schema": MULTIRACE_SCHEMA, "feature_names": list(MultiRaceFeatureExtractor.names), "race_ids": {name: value for name, value in {"Unknown": 0, "Terran": 1, "Protoss": 2, "Zerg": 3}.items()},
                "universal_intents": {item.name: int(item) for item in UniversalIntent}, "race_actions": {item.name: int(item) for item in RaceMacroAction},
                "episode_ids": arrays["episode_ids"].tolist(), "game_loops": arrays["game_loops"].tolist(), "opponent_ids": [f"replay-{item.replay_id}-opponent" for item in transitions],
                "opponent_types": ["human_replay"] * len(transitions), "policy_versions": ["multirace-expert-replay"] * len(transitions), "map_names": arrays["map_names"].tolist(), "environment_types": ["real_sc2"] * len(transitions),
                "infos": [{"label_source": item.source, "label_evidence": list(item.evidence), "ability_ids": list(item.ability_ids)} for item in transitions], "summaries": [item.to_dict() for item in summaries]}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
