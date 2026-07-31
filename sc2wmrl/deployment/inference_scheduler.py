"""Macro decision timing with safe urgent-event overrides."""

from __future__ import annotations


class InferenceScheduler:
    """Avoids full actor inference at every SC2 iteration."""
    def __init__(self, macro_interval_game_loops: int) -> None:
        if macro_interval_game_loops <= 0: raise ValueError("macro interval must be positive")
        self.interval = macro_interval_game_loops; self.last_inference_loop: int | None = None

    def reset(self) -> None:
        """Clear episode-local decision history."""
        self.last_inference_loop = None

    def should_infer(self, game_loop: int, *, urgent_event: bool, skill_finished: bool) -> bool:
        """Trigger first, periodic, completed-skill, and urgent-event decisions."""
        if game_loop < 0: raise ValueError("game loop cannot be negative")
        if self.last_inference_loop is None or urgent_event or skill_finished: return True
        return game_loop - self.last_inference_loop >= self.interval

    def mark_inference(self, game_loop: int) -> None:
        """Record the game loop of the accepted decision."""
        self.last_inference_loop = game_loop
