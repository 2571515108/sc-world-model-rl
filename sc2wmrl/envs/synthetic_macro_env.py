"""Fast deterministic macro-RTS environment used for Phase 0/1 validation.

It is intentionally an interpretable strategic simulator, not a substitute for
the SC2 engine.  Its contract, masks, rewards, opponent labels, and observation
schema are identical to the real adapter so algorithms transfer unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .action_mask import action_mask
from .base_macro_env import InfoDict, MacroAction, MacroSC2Env, Observation
from .feature_extractor import FeatureExtractor, REGIONS, STRATEGIES
from .reward import RewardConfig, RewardTracker


@dataclass(frozen=True)
class SyntheticEnvConfig:
    """Configuration for a reproducible synthetic macro match."""

    seed: int = 7
    macro_interval_game_loops: int = 32
    max_macro_steps: int = 96
    map_width: float = 200.0
    map_height: float = 200.0
    opponent: str = "economy_bot"
    info_mode: str = "minimal"
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.macro_interval_game_loops <= 0 or self.max_macro_steps <= 0:
            raise ValueError("macro interval and maximum steps must be positive")
        if self.opponent not in {f"{name}_bot" for name in STRATEGIES if name != "unknown"} | {"randomized_bot"}:
            raise ValueError(f"unknown scripted opponent {self.opponent!r}")
        if self.info_mode not in {"minimal", "full"}:
            raise ValueError("info mode must be minimal or full")


class SyntheticMacroEnv(MacroSC2Env):
    """Structured Terran-vs-scripted-opponent macro simulator."""

    def __init__(self, config: SyntheticEnvConfig | None = None) -> None:
        self.config = config or SyntheticEnvConfig()
        self.extractor = FeatureExtractor(self.config.map_width, self.config.map_height)
        self.observation_dim = self.extractor.dimension
        self._rng = np.random.default_rng(self.config.seed)
        self._state: dict[str, Any] = {}
        self._enemy: dict[str, Any] = {}
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._reward = RewardTracker(self.config.reward)

    def _initial_state(self) -> dict[str, Any]:
        return {
            "game_time": 0, "minerals": 500.0, "vespene": 0.0, "current_supply": 12,
            "maximum_supply": 15, "worker_count": 12, "base_count": 1, "idle_worker_count": 0,
            "army_value": 0.0, "lost_army_value": 0.0,
            "buildings": {"command_center": 1, "barracks": 0, "factory": 0, "starport": 0,
                          "refinery": 0, "engineering_bay": 0, "tech_lab": 0, "reactor": 0},
            "completed_upgrade_flags": [0] * 6, "production_queue_summary": [0] * 5,
            "units": {"marine": 0, "marauder": 0, "reaper": 0, "hellion": 0, "tank": 0,
                      "medivac": 0, "viking": 0, "battlecruiser": 0},
            "average_army_health": 1.0, "army_center_x": 30.0, "army_center_y": 30.0,
            "number_of_army_groups": 0, "enemy_army_value_estimate": 0.0,
            "enemy": {"observed_unit_counts": {}, "observed_buildings": {}, "last_seen_army_position": None,
                      "time_since_last_scout": 3600.0, "estimated_army_value": 0.0,
                      "estimated_worker_count": 12, "strategy_probabilities": self._strategy_probabilities()},
            "map_control": {region: {"friendly_power": 0.0, "visible_enemy_power": 0.0, "visibility": 0.0,
                                      "control_score": 0.0, "last_scout_time": 3600.0} for region in REGIONS},
        }

    def _initial_enemy(self) -> dict[str, Any]:
        strategy = self.config.opponent.removesuffix("_bot")
        if strategy == "randomized":
            strategy = self._rng.choice(["rush", "economy", "defensive", "ground_tech", "air_tech"]).item()
        return {"strategy": strategy, "workers": 12, "bases": 1, "army": 0.0, "tech": 0.0, "last_action": int(MacroAction.NO_OP),
                "visible": False, "last_seen": None, "attacking": False, "lost_army": 0.0}

    def _strategy_probabilities(self) -> dict[str, float]:
        strategy = self._enemy.get("strategy", self.config.opponent.removesuffix("_bot"))
        return {name: float(name == strategy) for name in STRATEGIES} if strategy in STRATEGIES else {"unknown": 1.0}

    def reset(self, *, seed: int | None = None) -> tuple[Observation, InfoDict]:
        """Reset to a deterministic opening when a seed is supplied."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._initial_state()
        self._enemy = self._initial_enemy()
        self._state["enemy"]["strategy_probabilities"] = self._strategy_probabilities()
        self._steps, self._terminated, self._truncated = 0, False, False
        self._refresh_visibility()
        self._reward.reset(self._metrics())
        return self._observation(), self._info({"reset": True})

    def get_action_mask(self) -> np.ndarray:
        """Get the current legal-action vector and verify the terminal invariant."""
        mask = action_mask(self._state)
        if mask.shape != (self.action_dim,) or not mask.any():
            raise RuntimeError("synthetic environment generated an invalid action mask")
        return mask

    def step(self, macro_action: int) -> tuple[Observation, float, bool, bool, InfoDict]:
        """Apply a legal macro action, advance both economies, and report reward."""
        if self._terminated or self._truncated:
            raise RuntimeError("reset must be called before stepping a completed episode")
        action = self.validate_action(macro_action)
        mask = self.get_action_mask()
        if not mask[action]:
            raise ValueError(f"illegal macro action {action.name}")
        before = self._metrics()
        execution = self._execute(action)
        self._advance_opponent()
        self._advance_economy()
        self._steps += 1
        self._state["game_time"] += self.config.macro_interval_game_loops
        self._refresh_visibility()
        outcome = self._outcome()
        self._terminated = outcome is not None
        self._truncated = not self._terminated and self._steps >= self.config.max_macro_steps
        if self._truncated:
            outcome = "draw"
        reward, components = self._reward.step(self._metrics(), outcome)
        info = self._info({"action": action.name, "execution": execution, "reward_components": components,
                           "metrics_before": before, "outcome": outcome})
        return self._observation(), reward, self._terminated, self._truncated, info

    def _execute(self, action: MacroAction) -> dict[str, Any]:
        s, b, u = self._state, self._state["buildings"], self._state["units"]
        result: dict[str, Any] = {"success": True, "event": "none"}
        if action == MacroAction.TRAIN_WORKERS:
            s["minerals"] -= 50; s["worker_count"] += 1; s["current_supply"] += 1
        elif action == MacroAction.BUILD_SUPPLY:
            s["minerals"] -= 100; s["maximum_supply"] = min(200, s["maximum_supply"] + 8)
            result["event"] = "supply_built"
        elif action == MacroAction.BUILD_BARRACKS:
            s["minerals"] -= 150; b["barracks"] += 1; result["event"] = "barracks_built"
        elif action == MacroAction.BUILD_REFINERY:
            s["minerals"] -= 75; b["refinery"] += 1; result["event"] = "refinery_built"
        elif action == MacroAction.BUILD_FACTORY:
            s["minerals"] -= 150; s["vespene"] -= 100; b["factory"] += 1; result["event"] = "factory_built"
        elif action == MacroAction.BUILD_STARPORT:
            s["minerals"] -= 150; s["vespene"] -= 100; b["starport"] += 1; result["event"] = "starport_built"
        elif action == MacroAction.EXPAND:
            s["minerals"] -= 400; s["base_count"] += 1; b["command_center"] += 1; result["event"] = "base_created"
        elif action == MacroAction.TRAIN_BASIC_ARMY:
            s["minerals"] -= 50; s["current_supply"] += 1; u["marine"] += 1; s["army_value"] += 50
        elif action == MacroAction.TRAIN_ANTI_GROUND:
            s["minerals"] -= 100; s["vespene"] -= 25; s["current_supply"] += 2; u["marauder"] += 1; s["army_value"] += 125
        elif action == MacroAction.TRAIN_ANTI_AIR:
            s["minerals"] -= 100; s["vespene"] -= 100; s["current_supply"] += 2; u["viking"] += 1; s["army_value"] += 200
        elif action == MacroAction.RESEARCH_UPGRADE:
            s["minerals"] -= 100; s["vespene"] -= 100
            index = s["completed_upgrade_flags"].index(0); s["completed_upgrade_flags"][index] = 1; result["event"] = "tech_completed"
        elif action in (MacroAction.SCOUT_ENEMY_MAIN, MacroAction.SCOUT_EXPANSION):
            self._enemy["visible"] = True; self._enemy["last_seen"] = self._state["game_time"]
            result["event"] = "scouted"
        elif action in (MacroAction.DEFEND_MAIN, MacroAction.DEFEND_NATURAL, MacroAction.RETREAT):
            self._enemy["attacking"] = False; s["average_army_health"] = min(1.0, s["average_army_health"] + 0.08)
            result["event"] = "defensive_posture"
        elif action in (MacroAction.HARASS, MacroAction.ATTACK_ENEMY_NATURAL, MacroAction.ATTACK_ENEMY_MAIN):
            aggression = {MacroAction.HARASS: 0.35, MacroAction.ATTACK_ENEMY_NATURAL: 0.65, MacroAction.ATTACK_ENEMY_MAIN: 1.0}[action]
            own_power = s["army_value"] * (1 + 0.06 * sum(s["completed_upgrade_flags"]))
            enemy_power = self._enemy["army"] * (1 + 0.08 * self._enemy["tech"])
            margin = own_power * aggression - enemy_power * (0.65 + 0.2 * aggression)
            noise = self._rng.normal(0.0, max(20.0, enemy_power * 0.08))
            if margin + noise >= 0:
                damage = min(self._enemy["army"], max(25.0, own_power * aggression * 0.45))
                self._enemy["army"] -= damage; self._enemy["lost_army"] += damage
                if action != MacroAction.HARASS and self._enemy["army"] < 80:
                    self._enemy["bases"] = max(0, self._enemy["bases"] - 1)
                result["event"] = "successful_attack"
            else:
                loss = min(s["army_value"], max(20.0, enemy_power * aggression * 0.25))
                s["army_value"] -= loss; s["lost_army_value"] += loss; s["average_army_health"] = max(0.2, s["average_army_health"] - 0.18)
                result["event"] = "failed_attack"
        return result

    def _advance_economy(self) -> None:
        s, b = self._state, self._state["buildings"]
        s["minerals"] += s["worker_count"] * 4.5 * s["base_count"] ** 0.15
        s["vespene"] += b["refinery"] * 4.0
        s["idle_worker_count"] = 0
        s["number_of_army_groups"] = int(s["army_value"] >= 100) + int(s["army_value"] >= 400)
        s["production_queue_summary"] = [b["barracks"], b["factory"], b["starport"], b["refinery"], s["base_count"]]

    def _advance_opponent(self) -> None:
        e, strategy = self._enemy, self._enemy["strategy"]
        e["workers"] = min(80, e["workers"] + (2 if strategy == "economy" else 1)); e["last_action"] = int(MacroAction.TRAIN_WORKERS)
        expansion = int(strategy == "economy" and self._steps in (12, 32, 55)); e["bases"] += expansion
        if expansion: e["last_action"] = int(MacroAction.EXPAND)
        growth = {"rush": 75, "economy": 43, "defensive": 50, "ground_tech": 58, "air_tech": 55}[strategy]
        if strategy == "rush" and self._steps < 15: growth *= 1.55
        if strategy in {"ground_tech", "air_tech"} and self._steps > 18: e["tech"] += 0.1
        e["army"] += growth + self._rng.normal(0, 4)
        e["attacking"] = bool(strategy == "rush" and 8 <= self._steps <= 25) or (e["army"] > 650 and self._steps % 9 == 0)
        if e["attacking"]: e["last_action"] = int(MacroAction.ATTACK_ENEMY_NATURAL)
        if e["attacking"]:
            s = self._state
            defend_power = s["army_value"] * (1.15 if s["average_army_health"] > 0.8 else 1.0)
            loss = max(0.0, e["army"] * 0.08 - defend_power * 0.05)
            if loss:
                s["army_value"] = max(0.0, s["army_value"] - loss); s["lost_army_value"] += loss
                if s["army_value"] < 30 and self._rng.random() < 0.08:
                    s["base_count"] = max(0, s["base_count"] - 1); s["buildings"]["command_center"] = s["base_count"]

    def _refresh_visibility(self) -> None:
        s, e = self._state, self._enemy
        enemy = s["enemy"]
        time_since = 3600.0 if e["last_seen"] is None else max(0.0, s["game_time"] - e["last_seen"])
        enemy["time_since_last_scout"] = time_since
        if e["visible"]:
            enemy["last_seen_army_position"] = (170.0, 170.0)
            enemy["estimated_army_value"] = e["army"]
            enemy["estimated_worker_count"] = e["workers"]
            enemy["observed_unit_counts"] = {"marine": int(e["army"] / 50)}
            enemy["observed_buildings"] = {"base": e["bases"], "production": int(e["tech"] + 1)}
        else:
            decay = float(np.exp(-time_since / 900.0))
            enemy["estimated_army_value"] = e["army"] * decay
            enemy["estimated_worker_count"] = e["workers"] * decay
        s["enemy_army_value_estimate"] = enemy["estimated_army_value"]
        probabilities = self._strategy_probabilities() if e["visible"] else {name: (1.0 / len(STRATEGIES)) for name in STRATEGIES}
        enemy["strategy_probabilities"] = probabilities
        for region in REGIONS:
            entry = s["map_control"][region]
            entry["friendly_power"] = s["army_value"] if region.startswith("own") else 0.0
            entry["visible_enemy_power"] = e["army"] if e["visible"] and region.startswith("enemy") else 0.0
            entry["visibility"] = 1.0 if e["visible"] and region.startswith("enemy") else (1.0 if region.startswith("own") else 0.2)
            entry["control_score"] = float(np.clip((s["army_value"] - e["army"]) / 1000.0, -1, 1))
            entry["last_scout_time"] = time_since if region.startswith("enemy") else 0.0

    def _metrics(self) -> dict[str, float]:
        s, e = self._state, self._enemy
        return {"army_advantage": (s["army_value"] - e["army"]) / 1000.0,
                "worker_advantage": (s["worker_count"] - e["workers"]) / 80.0,
                "base_advantage": (s["base_count"] - e["bases"]) / 8.0,
                "technology_progress": sum(s["completed_upgrade_flags"]) / 6.0,
                "successful_scout": float(e["visible"]),
                "map_control": float(np.mean([v["control_score"] for v in s["map_control"].values()]))}

    def _outcome(self) -> str | None:
        if self._enemy["bases"] <= 0 or (self._enemy["army"] <= 0 and self._enemy["workers"] < 5): return "win"
        if self._state["base_count"] <= 0: return "loss"
        return None

    def _observation(self) -> Observation:
        return self.extractor.extract(self._state)

    def _info(self, additional: dict[str, Any]) -> InfoDict:
        info: InfoDict = {"action_mask": self.get_action_mask().copy(), "opponent_id": self.config.opponent,
                          "opponent_type": "scripted", "game_loop": self._state["game_time"], "step": self._steps,
                          "environment_type": "synthetic", "enemy_strategy": self._enemy["strategy"], "opponent_action": self._enemy["last_action"]}
        if self.config.info_mode == "full":
            info["raw_state"] = deepcopy(self._state)
        info.update(additional)
        return info

    def config_dict(self) -> dict[str, Any]:
        """Return a serializable config snapshot for logs and checkpoints."""
        result = asdict(self.config)
        result["reward"] = asdict(self.config.reward)
        return result
