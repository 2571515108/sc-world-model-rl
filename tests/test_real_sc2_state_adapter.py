"""Mock-only tests for real-state extraction without a game client."""

import unittest
from types import SimpleNamespace

from sc2wmrl.envs.action_mask import action_mask
from sc2wmrl.envs.real_sc2_state_adapter import RealSC2StateAdapter


class Unit:
    """Small python-sc2-like unit fixture."""

    def __init__(self, name: str, x: float = 1.0, y: float = 2.0, idle: bool = True) -> None:
        self.type_id = SimpleNamespace(name=name); self.position = SimpleNamespace(x=x, y=y)
        self.health_percentage = 1.0; self.is_idle = idle


class Bot:
    """Visible-state-only BotAI fixture."""

    time = 42.0; minerals = 300; vespene = 75; supply_used = 20; supply_cap = 31
    workers = [Unit("SCV")]; units = [Unit("MARINE")]
    structures = [Unit("COMMANDCENTER"), Unit("BARRACKS")]
    enemy_units = [Unit("MARINE", 160, 160)]; enemy_structures = [Unit("COMMANDCENTER", 170, 170)]
    enemy_start_locations = [SimpleNamespace(x=170, y=170)]
    state = SimpleNamespace(upgrades=set())

    @staticmethod
    def calculate_unit_value(unit_type):
        del unit_type
        return SimpleNamespace(minerals=50, vespene=0)


class RealStateAdapterTests(unittest.TestCase):
    """Verify visible enemy memory and shared action-mask compatibility."""

    def test_extracts_shared_schema_without_hidden_enemy_access(self) -> None:
        state = RealSC2StateAdapter().extract_raw_state(Bot())
        self.assertEqual(state["worker_count"], 1)
        self.assertEqual(state["enemy"]["observed_unit_counts"]["marine"], 1)
        self.assertEqual(action_mask(state).shape, (20,))
        self.assertTrue(action_mask(state).any())
