"""Prepare and resolve exact StarCraft II builds required by replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sc2wmrl.replays.metadata import ReplayMetadata


@dataclass(frozen=True)
class ReplayVersionPreparation:
    """Auditable result of an optional client-side replay-data download."""

    replay_path: str
    game_version: str
    base_build: int
    data_version: str | None
    executable_path: str
    executable_present_before: bool
    download_requested: bool
    executable_present_after: bool
    replay_info_error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe preparation report."""
        return asdict(self)


def exact_replay_version(replay_lib: Any, replay_data: bytes, metadata: ReplayMetadata) -> Any:
    """Return a fully specified PySC2 Version without a static-table lookup.

    Old PySC2 releases do not list every later SC2 patch in ``VERSIONS``. A
    fully specified ``Version`` is accepted by the run configuration directly,
    while a partially specified one is looked up by game-version string and
    fails for valid newer replay headers.
    """
    parsed = replay_lib.get_replay_version(replay_data)
    if int(parsed.build_version) != metadata.base_build:
        raise ValueError(
            f"replay header build {parsed.build_version} disagrees with metadata Base{metadata.base_build}"
        )
    if metadata.data_version and parsed.data_version and str(parsed.data_version).upper() != metadata.data_version.upper():
        raise ValueError("replay header data version disagrees with replay metadata")
    if not parsed.data_version:
        raise ValueError("replay header does not contain a required data version")
    # ``binary`` is only a completeness marker in PySC2's Version object. The
    # Windows run configuration resolves the executable from ``build_version``.
    return type(parsed)(str(parsed.game_version), int(parsed.build_version), str(parsed.data_version), "replay")


def replay_executable_path(data_dir: str | Path, base_build: int) -> Path:
    """Return the standard Windows executable path for a replay base build."""
    return Path(data_dir) / "Versions" / f"Base{int(base_build):05d}" / "SC2_x64.exe"


def _send_replay_info_download(controller: Any, sc_pb: Any, replay_data: bytes) -> Any:
    """Use the protocol field omitted by PySC2's public ``replay_info`` API."""
    return controller._client.send(  # pylint: disable=protected-access
        replay_info=sc_pb.RequestReplayInfo(replay_data=replay_data, download_data=True)
    )


def prepare_replay_version(replay_path: str | Path, metadata: ReplayMetadata, *, run_configs: Any, sc_pb: Any) -> ReplayVersionPreparation:
    """Ask the current SC2 client to fetch replay data and report exact state.

    ``download_data`` is a client request, not a guarantee that Blizzard still
    serves a retired binary. The returned report therefore checks the requested
    executable before and after the request and preserves any protocol error.
    """
    replay_path = Path(replay_path).expanduser().resolve()
    current_config = run_configs.get()
    executable = replay_executable_path(current_config.data_dir, metadata.base_build)
    before = executable.is_file()
    response = None
    error: str | None = None
    try:
        replay_data = current_config.replay_data(str(replay_path))
        with current_config.start(want_rgb=False) as controller:
            response = _send_replay_info_download(controller, sc_pb, replay_data)
        if int(getattr(response, "error", 0) or 0):
            name = sc_pb.ResponseReplayInfo.Error.Name(int(response.error))
            details = str(getattr(response, "error_details", "")).strip()
            error = f"{name}: {details}".rstrip(": ")
    except Exception as exc:  # Preserve a diagnostic report for command-line users.
        error = f"{type(exc).__name__}: {exc}"
    return ReplayVersionPreparation(
        replay_path=str(replay_path), game_version=metadata.game_version, base_build=metadata.base_build,
        data_version=metadata.data_version, executable_path=str(executable), executable_present_before=before,
        download_requested=True, executable_present_after=executable.is_file(), replay_info_error=error,
    )
