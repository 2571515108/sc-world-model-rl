"""Threaded bridge from synchronous :class:`MacroSC2Env` to python-sc2 BotAI.

``run_game`` owns an asyncio loop.  This backend starts it in a dedicated
thread and exchanges macro requests through thread-safe queues, so callers do
not call ``asyncio.run`` from an already-running application event loop.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .action_mask import action_mask
from .base_macro_env import MacroAction
from .real_sc2_state_adapter import RealSC2StateAdapter
from .sc2_macro_action_executor import SC2MacroActionExecutor


@dataclass(frozen=True)
class PythonSC2BackendConfig:
    """Platform-independent launch settings; SC2 installation is external."""

    map_name: str
    race: str = "Terran"
    opponent_type: str = "builtin"
    opponent_race: str = "Terran"
    opponent_difficulty: str = "Medium"
    realtime: bool = False
    timeout_seconds: float = 120.0
    map_width: float = 200.0
    map_height: float = 200.0


class PythonSC2Backend:
    """Concrete real-game backend built on optional ``burnysc2``.

    A macro request is issued exactly once by ``SC2MacroActionExecutor``.  The
    BotAI callback then allows the client to advance a requested number of game
    loops before publishing the next state snapshot.
    """

    def __init__(self, config: PythonSC2BackendConfig) -> None:
        self.config = config
        self._requests: queue.Queue[tuple[MacroAction, int] | None] = queue.Queue()
        self._snapshots: queue.Queue[dict[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_state: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}

    def reset_game(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh SC2 process and block until BotAI exposes its first state."""
        self.close()
        self._stop.clear(); self._requests = queue.Queue(); self._snapshots = queue.Queue()
        self._thread = threading.Thread(target=self._run_game, args=(seed,), name="sc2wmrl-python-sc2", daemon=True)
        self._thread.start()
        packet = self._wait_snapshot()
        self._last_state, self._last_info = packet["state"], packet["info"]
        return self._last_state, dict(self._last_info)

    def execute_macro(self, action: MacroAction, duration: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Request one action and wait for the post-interval, real-game snapshot."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("real SC2 game is not running; call reset_game first")
        self._requests.put((action, duration))
        packet = self._wait_snapshot()
        self._last_state, self._last_info = packet["state"], packet["info"]
        return self._last_state, 0.0, bool(packet["terminated"]), False, dict(self._last_info)

    def get_action_mask(self) -> np.ndarray:
        """Derive legality from the latest visible state instead of client truth."""
        if self._last_state is None:
            mask = np.zeros(len(MacroAction), dtype=np.bool_); mask[MacroAction.NO_OP] = True; return mask
        return action_mask(self._last_state)

    def close(self) -> None:
        """Ask the BotAI callback to leave, then join without leaking a process."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._requests.put(None)
            self._thread.join(timeout=10.0)
        self._thread = None

    def _wait_snapshot(self) -> dict[str, Any]:
        try:
            packet = self._snapshots.get(timeout=self.config.timeout_seconds)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("timed out waiting for the SC2 client state") from exc
        if "error" in packet:
            raise RuntimeError(f"SC2 game process failed: {packet['error']}")
        return packet

    def _run_game(self, seed: int | None) -> None:
        try:
            from sc2 import Difficulty, Race, maps, run_game
            from sc2.bot_ai import BotAI
            from sc2.player import Bot, Computer
        except ImportError as exc:
            self._snapshots.put({"error": "burnysc2 is not installed. Install the optional real-SC2 dependency."})
            return
        owner = self
        adapter = RealSC2StateAdapter(self.config.map_width, self.config.map_height)
        executor = SC2MacroActionExecutor()

        class ControlledTerranBot(BotAI):
            """BotAI callback that pauses only at macro decision boundaries."""

            def __init__(self) -> None:
                super().__init__(); self.next_publish_iteration = 0; self.last_execution: dict[str, Any] = {"success": True, "action": "RESET"}; self.sent_initial = False

            async def on_start(self) -> None:
                adapter.reset()

            async def on_step(self, iteration: int) -> None:
                if owner._stop.is_set():
                    await self.client.leave(); return
                if iteration < self.next_publish_iteration:
                    return
                executor.clear_completed(self)
                state = adapter.extract_raw_state(self, pending_actions=executor.pending_actions)
                owner._snapshots.put({"state": state, "terminated": False, "info": {"environment_type": "real_sc2", "map_name": owner.config.map_name, "opponent_id": owner.config.opponent_type, "opponent_type": owner.config.opponent_type, "game_loop": int(getattr(self.state, "game_loop", iteration)), "execution": self.last_execution, "seed": seed}})
                request = await asyncio.to_thread(owner._requests.get)
                if request is None or owner._stop.is_set():
                    await self.client.leave(); return
                action, duration = request
                self.last_execution = await executor.execute(action, self)
                self.next_publish_iteration = iteration + max(1, duration)

            async def on_end(self, result: Any) -> None:
                state = adapter.extract_raw_state(self, pending_actions=executor.pending_actions)
                outcome = "win" if str(result).lower().endswith("victory") else "loss" if str(result).lower().endswith("defeat") else "draw"
                owner._snapshots.put({"state": state, "terminated": True, "info": {"environment_type": "real_sc2", "map_name": owner.config.map_name, "opponent_id": owner.config.opponent_type, "opponent_type": owner.config.opponent_type, "game_loop": int(getattr(self.state, "game_loop", 0)), "execution": self.last_execution, "outcome": outcome, "seed": seed}})

        try:
            race = getattr(Race, self.config.race.title())
            enemy_race = getattr(Race, self.config.opponent_race.title())
            difficulty = getattr(Difficulty, self.config.opponent_difficulty.title())
            opponent: Any = Computer(enemy_race, difficulty) if self.config.opponent_type == "builtin" else Computer(enemy_race, difficulty)
            run_game(maps.get(self.config.map_name), [Bot(race, ControlledTerranBot()), opponent], realtime=self.config.realtime)
        except Exception as exc:
            self._snapshots.put({"error": f"{type(exc).__name__}: {exc}"})
