"""Optional backend-neutral game overlay renderer."""

from __future__ import annotations

from typing import Protocol


class OverlaySink(Protocol):
    """Minimal BotAI/debug overlay drawing contract."""
    def draw_text(self, text: str) -> None:
        """Draw one text block in the game debug UI."""


class GameOverlay:
    """Formats decision diagnostics; disabled overlays do not affect inference."""
    def __init__(self, sink: OverlaySink | None, enabled: bool) -> None:
        self.sink, self.enabled = sink, enabled
    def render(self, diagnostics: dict[str, object]) -> None:
        """Draw a compact diagnostics panel when configured and available."""
        if self.enabled and self.sink is not None:
            self.sink.draw_text("\n".join(f"{key}: {value}" for key, value in diagnostics.items()))
