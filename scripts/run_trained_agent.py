"""Load a trained real-time agent using an explicit backend factory."""

from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path

from sc2wmrl.controllers.macro_skill_executor import MacroSkillExecutor
from sc2wmrl.deployment.realtime_agent import RealtimeAgentConfig, RealtimeRLAgent


def _factory(specification: str):
    """Load a user-owned BotAI/backend factory from ``module:attribute`` notation."""
    module_name, attribute = specification.split(":", 1); return getattr(importlib.import_module(module_name), attribute)


async def _run(args: argparse.Namespace) -> None:
    """Create an agent and call the hosting backend's asynchronous match lifecycle."""
    backend = _factory(args.backend_factory)(); executor = MacroSkillExecutor(backend)
    agent = RealtimeRLAgent(RealtimeAgentConfig(Path(args.checkpoint), args.deterministic, args.macro_interval, args.device, enable_overlay=args.overlay, record_decisions=args.record_decisions), backend, executor, args.output)
    await agent.on_start(); await backend.run_agent(agent, realtime=args.realtime, render=args.render)


def main() -> None:
    """Require a real backend factory instead of silently simulating an SC2 match."""
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--backend-factory", required=True, help="module:factory returning a configured SC2 backend")
    parser.add_argument("--output", default="outputs/realtime_games"); parser.add_argument("--device", default="auto"); parser.add_argument("--macro-interval", type=int, default=32)
    parser.add_argument("--deterministic", action="store_true"); parser.add_argument("--realtime", action="store_true"); parser.add_argument("--render", action="store_true"); parser.add_argument("--overlay", action="store_true"); parser.add_argument("--record-decisions", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__": main()
