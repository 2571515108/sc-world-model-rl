"""Regression tests for milestone-preserving collection behavior."""

from __future__ import annotations

import unittest

import numpy as np

from sc2wmrl.agents.rule_based_agent import RuleBasedAgent
from sc2wmrl.envs.base_macro_env import MacroAction


class RuleBasedAgentTests(unittest.TestCase):
    """Ensure the opening does not spend away Barracks minerals."""

    def test_opening_waits_for_barracks_instead_of_training_workers(self) -> None:
        mask = np.zeros(len(MacroAction), dtype=np.bool_)
        mask[[MacroAction.NO_OP, MacroAction.TRAIN_WORKERS, MacroAction.BUILD_REFINERY]] = True
        action = RuleBasedAgent().act(np.zeros(4, dtype=np.float32), mask, {"buildings": {"supply_depot": 0, "barracks": 0}, "current_supply": 12, "maximum_supply": 15})
        self.assertEqual(action, MacroAction.NO_OP)

    def test_opening_builds_barracks_when_legal(self) -> None:
        mask = np.zeros(len(MacroAction), dtype=np.bool_)
        mask[[MacroAction.NO_OP, MacroAction.BUILD_BARRACKS, MacroAction.TRAIN_WORKERS]] = True
        action = RuleBasedAgent().act(np.zeros(4, dtype=np.float32), mask, {"buildings": {"supply_depot": 1, "barracks": 0}, "current_supply": 12, "maximum_supply": 15})
        self.assertEqual(action, MacroAction.BUILD_BARRACKS)

    def test_basic_army_precedes_refinery_after_opening(self) -> None:
        mask = np.zeros(len(MacroAction), dtype=np.bool_)
        mask[[MacroAction.NO_OP, MacroAction.TRAIN_BASIC_ARMY, MacroAction.BUILD_REFINERY]] = True
        action = RuleBasedAgent().act(np.zeros(4, dtype=np.float32), mask, {"buildings": {"barracks": 1, "refinery": 0}, "current_supply": 14, "maximum_supply": 23})
        self.assertEqual(action, MacroAction.TRAIN_BASIC_ARMY)

    def test_ready_army_attacks_before_queuing_more_marines(self) -> None:
        mask = np.zeros(len(MacroAction), dtype=np.bool_)
        mask[[MacroAction.NO_OP, MacroAction.TRAIN_BASIC_ARMY, MacroAction.ATTACK_ENEMY_MAIN]] = True
        action = RuleBasedAgent().act(np.zeros(4, dtype=np.float32), mask,
                                      {"buildings": {"barracks": 1}, "army_value": 250,
                                       "current_supply": 17, "maximum_supply": 23})
        self.assertEqual(action, MacroAction.ATTACK_ENEMY_MAIN)
