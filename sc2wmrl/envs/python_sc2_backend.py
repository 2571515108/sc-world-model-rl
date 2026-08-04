"""Backward-compatible import aliases for the PySC2 real-game backend.

Historically this module hosted a BurnySC2 ``BotAI`` bridge.  The production
backend is now PySC2; aliases keep older application imports working without
leaving a second SC2 client implementation in the codebase.
"""

from .pysc2_backend import PySC2Backend, PySC2BackendConfig

PythonSC2Backend = PySC2Backend
PythonSC2BackendConfig = PySC2BackendConfig

__all__ = ["PySC2Backend", "PySC2BackendConfig", "PythonSC2Backend", "PythonSC2BackendConfig"]
