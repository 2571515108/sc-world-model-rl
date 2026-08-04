"""Extract one fog-of-war replay viewpoint into macro replay transitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sc2wmrl.envs.action_mask import action_mask
from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.envs.feature_extractor import FeatureExtractor
from sc2wmrl.envs.pysc2_backend import PySC2StateAdapter
from sc2wmrl.envs.reward import RewardConfig, RewardTracker, reward_metrics_from_state
from sc2wmrl.replay.transition import MacroTransition
from sc2wmrl.replays.macro_labeler import ReplayMacroLabeler, replay_event_vector
from sc2wmrl.replays.metadata import ReplayMetadata, ReplayPlayer, inspect_replay_metadata
from sc2wmrl.replays.observation import controller_observation_dict, player_outcome, raw_unit_index
from sc2wmrl.replays.versioning import exact_replay_version, prepare_replay_version, replay_executable_path


_TERRAN_TOWNHALLS = {18, 130, 132}


@dataclass(frozen=True)
class ReplayExtractionConfig:
    """Extraction settings shared by every replay in an expert dataset."""

    step_mul: int = 32
    minimum_label_confidence: float = 0.70
    include_no_op_for_behavior_cloning: bool = False
    download_missing_version: bool = True
    map_width_fallback: float = 200.0
    map_height_fallback: float = 200.0
    reward: RewardConfig = field(default_factory=lambda: RewardConfig(successful_scout_scale=0.0001, terminal_draw=-0.1))

    def __post_init__(self) -> None:
        if self.step_mul <= 0:
            raise ValueError("replay step_mul must be positive")
        if not 0.0 <= self.minimum_label_confidence <= 1.0:
            raise ValueError("minimum_label_confidence must be in [0, 1]")


@dataclass(frozen=True)
class ReplayExtractionSummary:
    """Per-viewpoint conversion summary stored next to the output dataset."""

    replay_path: str
    map_name: str
    player_id: int
    player_race: str
    player_result: str
    player_apm: float
    game_version: str
    steps: int
    behavior_cloning_labels: int
    augmented_action_masks: int
    terminal_outcome: str | None
    action_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report value."""
        return asdict(self)


def _require_sc2() -> tuple[Any, Any, Any]:
    try:
        from absl import flags
        from pysc2 import run_configs
        from pysc2.lib import replay as replay_lib
        from s2clientprotocol import sc2api_pb2 as sc_pb
    except ImportError as exc:
        raise RuntimeError("Replay extraction requires PySC2 4, s2clientprotocol, mpyq, and Python 3.10.") from exc
    # The project uses argparse. PySC2 reads Abseil flags while selecting a
    # run configuration, so parse a harmless known-only argument list once.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["sc2wmrl-replay-extractor"], known_only=True)
    return run_configs, replay_lib, sc_pb


def _map_size(game_info: Any, config: ReplayExtractionConfig) -> tuple[float, float]:
    size = getattr(getattr(game_info, "start_raw", None), "map_size", None)
    return (float(getattr(size, "x", 0.0) or config.map_width_fallback),
            float(getattr(size, "y", 0.0) or config.map_height_fallback))


def _own_main(response: Any) -> tuple[float, float] | None:
    for unit in response.observation.raw_data.units:
        if int(unit.alliance) == 1 and int(unit.unit_type) in _TERRAN_TOWNHALLS:
            return float(unit.pos.x), float(unit.pos.y)
    return None


def _enemy_start(game_info: Any, own_main: tuple[float, float] | None) -> tuple[float, float] | None:
    locations = [(float(point.x), float(point.y)) for point in getattr(getattr(game_info, "start_raw", None), "start_locations", ())]
    if not locations:
        return None
    if own_main is None:
        return locations[0]
    return max(locations, key=lambda point: (point[0] - own_main[0]) ** 2 + (point[1] - own_main[1]) ** 2)


def _player_names(replay_info: Any, player_id: int) -> tuple[str, str]:
    names: dict[int, str] = {}
    for entry in getattr(replay_info, "player_info", ()):
        player = getattr(entry, "player_info", None)
        value = str(getattr(player, "player_name", "") or "").strip()
        if value:
            names[int(getattr(player, "player_id", 0))] = value
    expert = names.get(player_id, f"player-{player_id}")
    opponent = next((name for ident, name in names.items() if ident != player_id), "unknown")
    return expert, opponent


def _start_replay(controller: Any, run_config: Any, replay_data: bytes, replay_info: Any, request: Any) -> None:
    """Start a replay with its local map data when the install provides it."""
    local_map = getattr(replay_info, "local_map_path", "")
    if local_map:
        try:
            request.map_data = run_config.map_data(local_map)
        except (FileNotFoundError, OSError, ValueError):
            # Ladder maps are commonly known to the game client without a local
            # map payload. The controller supplies a precise error if not.
            pass
    controller.start_replay(request)


def _matching_run_config(run_configs: Any, replay_lib: Any, sc_pb: Any, replay_path: Path, replay_data: bytes,
                         metadata: ReplayMetadata, *, download_missing_version: bool) -> Any:
    """Resolve a replay build without relying on PySC2's static version table."""
    exact_version = exact_replay_version(replay_lib, replay_data, metadata)
    current_config = run_configs.get()
    executable = replay_executable_path(current_config.data_dir, metadata.base_build)
    if not executable.is_file() and download_missing_version:
        prepared = prepare_replay_version(replay_path, metadata, run_configs=run_configs, sc_pb=sc_pb)
        if prepared.replay_info_error:
            raise RuntimeError(
                f"Replay data download request failed for Base{metadata.base_build}: {prepared.replay_info_error}"
            )
    if not executable.is_file():
        raise RuntimeError(
            f"Replay {metadata.path} requires SC2 Base{metadata.base_build}, but {executable} is absent. "
            "The current client could not provision a runnable exact binary; install that build under "
            "StarCraft II/Versions before conversion."
        )
    try:
        return run_configs.get(version=exact_version)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to start exact replay build Base{metadata.base_build}: {exc}") from exc


def _align_next_action_masks(transitions: list[MacroTransition]) -> None:
    """Propagate an expert-corrected mask into the preceding model target.

    A replay label can expose a legal command that the coarse state-only mask
    did not infer.  The next-action-mask head must learn the corrected mask
    at the preceding transition too; otherwise its target could contradict the
    label used at the following decision.
    """
    for previous, current in zip(transitions, transitions[1:]):
        previous.next_action_mask = current.action_mask.copy()


def extract_replay_viewpoint(path: str | Path, player: ReplayPlayer, *, episode_id: int = 0,
                             config: ReplayExtractionConfig | None = None,
                             metadata: ReplayMetadata | None = None) -> tuple[list[MacroTransition], ReplayExtractionSummary]:
    """Extract exactly one player's non-omniscient replay viewpoint.

    The opponent action is intentionally unknown. It is persisted with an
    explicit validity flag so the world-model opponent-action loss ignores it.
    """
    config = config or ReplayExtractionConfig()
    replay_path = Path(path).expanduser().resolve()
    metadata = metadata or inspect_replay_metadata(replay_path)
    if player.player_id not in {candidate.player_id for candidate in metadata.players}:
        raise ValueError(f"player {player.player_id} is absent from {replay_path}")
    run_configs, replay_lib, sc_pb = _require_sc2()
    default_config = run_configs.get()
    replay_data = default_config.replay_data(str(replay_path))
    run_config = _matching_run_config(
        run_configs, replay_lib, sc_pb, replay_path, replay_data, metadata,
        download_missing_version=config.download_missing_version,
    )
    transitions: list[MacroTransition] = []
    labeler = ReplayMacroLabeler()
    augmented_masks = 0
    bc_labels = 0
    terminal_outcome: str | None = None

    with run_config.start(want_rgb=False) as controller:
        replay_info = controller.replay_info(replay_data)
        request = sc_pb.RequestStartReplay(
            replay_data=replay_data,
            options=sc_pb.InterfaceOptions(raw=True, score=True, show_cloaked=True, show_burrowed_shadows=True),
            disable_fog=False,
            observed_player_id=int(player.player_id),
        )
        _start_replay(controller, run_config, replay_data, replay_info, request)
        game_info = controller.game_info()
        width, height = _map_size(game_info, config)
        adapter, extractor = PySC2StateAdapter(width, height), FeatureExtractor(width, height)
        rewards = RewardTracker(config.reward)
        previous_response = controller.observe()
        previous_state = adapter.extract(controller_observation_dict(previous_response), {})
        rewards.reset(reward_metrics_from_state(previous_state))
        own_main, enemy_main = _own_main(previous_response), _enemy_start(game_info, _own_main(previous_response))
        expert_name, opponent_name = _player_names(replay_info, player.player_id)

        while not previous_response.player_result:
            controller.step(config.step_mul)
            current_response = controller.observe()
            current_state = adapter.extract(controller_observation_dict(current_response), {})
            label = labeler.infer(current_response.actions, previous_state, current_state, raw_unit_index(previous_response),
                                  own_main=own_main, enemy_main=enemy_main)
            outcome = player_outcome(current_response, player.player_id)
            terminal_outcome = outcome or terminal_outcome
            reward, reward_components = rewards.step(reward_metrics_from_state(current_state), outcome)
            previous_mask = action_mask(previous_state)
            augmented = not bool(previous_mask[int(label.action)])
            if augmented:
                previous_mask = previous_mask.copy(); previous_mask[int(label.action)] = True; augmented_masks += 1
            expert_label = label.confidence >= config.minimum_label_confidence and (
                config.include_no_op_for_behavior_cloning or label.action != MacroAction.NO_OP
            )
            bc_labels += int(expert_label)
            transitions.append(MacroTransition(
                observation=extractor.extract(previous_state), entity_observation=None, action=int(label.action), action_mask=previous_mask,
                reward=reward, terminated=outcome is not None, truncated=False, next_observation=extractor.extract(current_state),
                opponent_id=opponent_name, opponent_type="human_replay", policy_version=f"expert-replay:{expert_name}",
                map_name=metadata.title, game_loop=int(previous_response.observation.game_loop), events=replay_event_vector(label.action, outcome),
                info={"source": "sc2_replay", "source_replay": replay_path.name, "source_replay_path": str(replay_path),
                      "expert_player": expert_name, "expert_player_id": player.player_id, "expert_race": player.assigned_race,
                      "expert_result": player.result, "expert_apm": player.apm, "fog_of_war": True, "macro_step_mul": config.step_mul,
                      "expert_label": expert_label, "label_confidence": label.confidence, "label_source": label.source,
                      "label_evidence": list(label.evidence), "ability_ids": list(label.ability_ids),
                      "candidate_actions": list(label.candidate_actions), "mask_augmented_by_expert_command": augmented,
                      "reward_components": reward_components, "enemy_strategy": "unknown", "replay_game_version": metadata.game_version,
                      "replay_base_build": metadata.base_build},
                episode_id=episode_id, next_action_mask=action_mask(current_state), opponent_action=int(MacroAction.NO_OP),
                opponent_action_valid=False, environment_type="real_sc2",
            ))
            previous_response, previous_state = current_response, current_state
            if outcome is not None:
                break
    _align_next_action_masks(transitions)
    histogram = Counter(MacroAction(item.action).name for item in transitions)
    return transitions, ReplayExtractionSummary(
        replay_path=str(replay_path), map_name=metadata.title, player_id=player.player_id, player_race=player.assigned_race,
        player_result=player.result, player_apm=player.apm, game_version=metadata.game_version, steps=len(transitions),
        behavior_cloning_labels=bc_labels, augmented_action_masks=augmented_masks, terminal_outcome=terminal_outcome,
        action_histogram=dict(histogram),
    )
