"""Mock-backend smoke test for the full real-time agent lifecycle."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sc2wmrl.agents.ppo_agent import PPOAgent, PPOConfig
from sc2wmrl.controllers.macro_skill_executor import MacroSkillExecutor, SkillResult
from sc2wmrl.deployment.realtime_agent import RealtimeAgentConfig, RealtimeRLAgent
from sc2wmrl.envs.base_macro_env import MacroAction


class MockBackend:
    """Minimal BotAI-like adapter exposing structured state and macro execution."""
    def __init__(self) -> None: self.loop = 0; self.executed = 0
    def structured_state(self): return {"minerals": 500, "maximum_supply": 15, "current_supply": 12, "worker_count": 12, "base_count": 1, "buildings": {"command_center": 1}}
    def action_mask(self): mask = np.zeros(len(MacroAction), dtype=bool); mask[MacroAction.NO_OP] = True; return mask
    def game_loop(self): return self.loop
    def urgent_event(self): return False
    def skill_finished(self): return False
    async def execute_macro(self, action, duration, context): self.executed += 1; return SkillResult(True, 1)


class RealtimeAgentSmokeTests(unittest.TestCase):
    """Load a real checkpoint, issue a legal action, and persist decision artifacts."""
    def test_start_step_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"; PPOAgent(106, PPOConfig(hidden_dim=8), device="cpu").save(path, training_state={"macro_action_names": [item.name for item in MacroAction]})
            backend = MockBackend(); agent = RealtimeRLAgent(RealtimeAgentConfig(path, device="cpu"), backend, MacroSkillExecutor(backend), Path(directory) / "games")
            asyncio.run(agent.on_start()); asyncio.run(agent.on_step(0)); asyncio.run(agent.on_end("Victory"))
            summaries = list((Path(directory) / "games").rglob("game_summary.json"))
        self.assertEqual(backend.executed, 1); self.assertEqual(len(summaries), 1)
