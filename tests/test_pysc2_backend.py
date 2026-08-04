"""Offline contract tests for the optional PySC2 real-game backend."""

from __future__ import annotations

import unittest

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction
from sc2wmrl.envs.pysc2_backend import PySC2BackendConfig, PySC2MacroActionExecutor, PySC2StateAdapter


class PySC2StateAdapterTests(unittest.TestCase):
    """Verify raw-unit conversion without launching a StarCraft II process."""

    def test_visible_units_create_fixed_canonical_state(self) -> None:
        adapter = PySC2StateAdapter(200.0, 200.0)
        observation = {
            "player": np.asarray([1, 500, 100, 12, 23], dtype=np.int32),
            "game_loop": np.asarray([224], dtype=np.int32),
            "raw_units": [
                {"alliance": 1, "unit_type": 18, "tag": 1, "x": 20, "y": 20, "health": 1500, "health_max": 1500, "order_length": 0},
                {"alliance": 1, "unit_type": 45, "tag": 2, "x": 21, "y": 20, "health": 45, "health_max": 45, "order_length": 0},
                {"alliance": 1, "unit_type": 48, "tag": 3, "x": 24, "y": 21, "health": 45, "health_max": 45, "order_length": 0},
                {"alliance": 4, "unit_type": 48, "tag": 4, "x": 160, "y": 160, "health": 45, "health_max": 45, "order_length": 0},
            ],
        }
        state = adapter.extract(observation, {"BUILD_BARRACKS": True})
        self.assertEqual(state["worker_count"], 1)
        self.assertEqual(state["units"]["marine"], 1)
        self.assertEqual(state["enemy"]["last_seen_army_position"], (160.0, 160.0))
        self.assertEqual(state["pending_actions"], {"BUILD_BARRACKS": True})
        self.assertGreater(state["army_value"], 0.0)

    def test_enemy_memory_persists_only_after_a_visible_observation(self) -> None:
        adapter = PySC2StateAdapter(200.0, 200.0)
        seen = {"game_loop": np.asarray([10]), "raw_units": [{"alliance": 4, "unit_type": 48, "x": 150, "y": 150}]}
        adapter.extract(seen, {})
        hidden = {"game_loop": np.asarray([110]), "raw_units": []}
        state = adapter.extract(hidden, {})
        self.assertEqual(state["enemy"]["last_seen_army_position"], (150.0, 150.0))
        self.assertAlmostEqual(state["enemy"]["time_since_last_scout"], 100 / 22.4)

    def test_config_keeps_client_setup_independent_from_imports(self) -> None:
        config = PySC2BackendConfig(map_name="AcropolisLE")
        self.assertEqual(config.race, "Terran")
        self.assertEqual(MacroAction.NO_OP, 0)

    def test_command_reservation_blocks_duplicate_macro_requests(self) -> None:
        executor = PySC2MacroActionExecutor(200.0, 200.0, cooldown_game_loops=64)
        executor.reserve(MacroAction.BUILD_BARRACKS, issued_loop=100, duration=32)
        self.assertFalse(executor.is_ready(MacroAction.BUILD_BARRACKS))
        executor.advance(163)
        self.assertFalse(executor.is_ready(MacroAction.BUILD_BARRACKS))
        executor.advance(164)
        self.assertTrue(executor.is_ready(MacroAction.BUILD_BARRACKS))

    def test_state_exposes_engineering_bay_availability(self) -> None:
        adapter = PySC2StateAdapter(200.0, 200.0)
        state = adapter.extract({"raw_units": [{"alliance": 1, "unit_type": 22, "tag": 1, "order_length": 0}]}, {})
        self.assertTrue(state["available"]["engineering_bay"])

    def test_unfinished_supply_depot_is_not_a_completed_prerequisite(self) -> None:
        adapter = PySC2StateAdapter(200.0, 200.0)
        state = adapter.extract({"raw_units": [{"alliance": 1, "unit_type": 19, "tag": 1, "order_length": 0,
                                                 "build_progress": 99.0}]}, {})
        self.assertFalse(state["available"]["supply_depot"])
        completed = adapter.extract({"raw_units": [{"alliance": 1, "unit_type": 19, "tag": 1, "order_length": 0,
                                                     "build_progress": 100.0}]}, {})
        self.assertTrue(completed["available"]["supply_depot"])

    def test_build_candidates_are_separated(self) -> None:
        executor = PySC2MacroActionExecutor(200.0, 200.0)
        candidates = executor.build_candidates([{"alliance": 1, "unit_type": 18, "x": 154, "y": 163}])
        self.assertGreater(len(candidates), 4)
        self.assertEqual(len(candidates), len(set(candidates)))


if __name__ == "__main__":
    unittest.main()
