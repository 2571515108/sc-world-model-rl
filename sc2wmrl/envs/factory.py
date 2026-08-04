"""Configuration-driven construction for synthetic and real macro environments."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from sc2wmrl.utils.config import load_yaml

from .real_sc2_macro_env import RealSC2MacroEnv
from .pysc2_backend import PySC2Backend, PySC2BackendConfig
from .reward import RewardConfig
from .synthetic_macro_env import SyntheticEnvConfig, SyntheticMacroEnv


def build_macro_env(config_path: str):
    """Build one environment from a YAML file without hard-coded SC2 paths."""
    raw = load_yaml(config_path)
    environment = raw.pop("environment", {})
    environment_type = str(environment.get("type", raw.pop("type", "synthetic")))
    reward = RewardConfig(**raw.pop("reward", {}))
    if environment_type == "synthetic":
        allowed = {field.name for field in fields(SyntheticEnvConfig)} - {"reward"}
        return SyntheticMacroEnv(SyntheticEnvConfig(**{key: value for key, value in raw.items() if key in allowed}, reward=reward))
    if environment_type != "real_sc2":
        raise ValueError(f"unknown environment type {environment_type!r}")
    values = {**raw, **{key: value for key, value in environment.items() if key != "type"}}
    opponent = values.pop("opponent", {})
    if isinstance(opponent, str):
        opponent = {"type": opponent}
    backend_name = str(values.get("backend", "pysc2")).lower()
    if backend_name != "pysc2":
        raise ValueError("the bundled real-SC2 environment supports only backend: pysc2")
    backend = PySC2Backend(PySC2BackendConfig(
        map_name=str(values.get("map_name", "AbyssalReef")), map_file=str(values["map_file"]) if values.get("map_file") else None, race=str(values.get("race", "Terran")),
        opponent_type=str(opponent.get("type", "builtin")), opponent_race=str(opponent.get("race", "Terran")),
        opponent_difficulty=str(opponent.get("difficulty", "Medium")), realtime=bool(values.get("realtime", False)),
        timeout_seconds=float(values.get("timeout_seconds", 120.0)), command_cooldown_game_loops=int(values.get("command_cooldown_game_loops", 64)),
        map_width=float(values.get("map_width", 200.0)), map_height=float(values.get("map_height", 200.0)),
    ))
    return RealSC2MacroEnv(backend, macro_interval_game_loops=int(values.get("macro_interval_game_loops", 32)),
                           max_macro_steps=int(values.get("max_macro_steps", 256)), map_width=float(values.get("map_width", 200.0)),
                           map_height=float(values.get("map_height", 200.0)), reward=reward)
