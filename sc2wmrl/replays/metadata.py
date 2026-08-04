"""Fast, engine-free inspection of StarCraft II replay metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


def _normalise_race(value: str) -> str:
    """Map replay abbreviations to the public race spelling."""
    aliases = {
        "terr": "Terran", "terran": "Terran", "zerg": "Zerg",
        "prot": "Protoss", "protoss": "Protoss", "rand": "Random", "random": "Random",
    }
    normalised = str(value).strip().lower()
    return aliases.get(normalised, normalised.title())


@dataclass(frozen=True)
class ReplayPlayer:
    """One player record embedded in a replay MPQ archive."""

    player_id: int
    apm: float
    result: str
    selected_race: str
    assigned_race: str

    @property
    def won(self) -> bool:
        """Whether this player won the recorded game."""
        return self.result.strip().lower() == "win"


@dataclass(frozen=True)
class ReplayMetadata:
    """Stable replay metadata available without launching SC2."""

    path: str
    title: str
    game_version: str
    base_build: int
    data_build: int | None
    data_version: str | None
    duration: int
    is_not_available: bool
    players: tuple[ReplayPlayer, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata for reports and command-line output."""
        value = asdict(self)
        value["players"] = [asdict(player) for player in self.players]
        return value


def _read_member(path: Path) -> dict[str, Any]:
    """Read only ``replay.gamemetadata.json`` from the replay archive."""
    try:
        import mpyq
    except ImportError as exc:
        raise RuntimeError("Replay metadata inspection requires mpyq; install the project's sc2 extra.") from exc
    archive = mpyq.MPQArchive(str(path))
    try:
        raw = archive.read_file("replay.gamemetadata.json")
        if raw is None:
            raw = archive.read_file(b"replay.gamemetadata.json")
    finally:
        close = getattr(archive, "close", None)
        if callable(close):
            close()
    if raw is None:
        raise ValueError(f"{path} does not contain replay.gamemetadata.json")
    return json.loads(raw.decode("utf-8"))


def inspect_replay_metadata(path: str | Path) -> ReplayMetadata:
    """Inspect a ``.SC2Replay`` archive without starting a game process."""
    replay_path = Path(path).expanduser().resolve()
    if replay_path.suffix.lower() != ".sc2replay":
        raise ValueError("replay path must end with .SC2Replay")
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    payload = _read_member(replay_path)
    base = str(payload.get("BaseBuild", ""))
    if not base.startswith("Base") or not base[4:].isdigit():
        raise ValueError(f"invalid replay BaseBuild value: {base!r}")
    players = tuple(
        ReplayPlayer(
            player_id=int(value["PlayerID"]), apm=float(value.get("APM", 0.0)),
            result=str(value.get("Result", "Unknown")),
            selected_race=_normalise_race(value.get("SelectedRace", "Unknown")),
            assigned_race=_normalise_race(value.get("AssignedRace", "Unknown")),
        )
        for value in payload.get("Players", [])
    )
    return ReplayMetadata(
        path=str(replay_path), title=str(payload.get("Title", "Unknown")),
        game_version=str(payload.get("GameVersion", "Unknown")), base_build=int(base[4:]),
        data_build=int(payload["DataBuild"]) if str(payload.get("DataBuild", "")).isdigit() else None,
        data_version=str(payload["DataVersion"]) if payload.get("DataVersion") else None,
        duration=int(payload.get("Duration", 0)), is_not_available=bool(payload.get("IsNotAvailable", False)),
        players=players,
    )


def select_expert_players(metadata: ReplayMetadata, *, target_race: str = "Terran", winners_only: bool = True,
                          min_apm: float = 0.0, player_id: int | None = None) -> tuple[ReplayPlayer, ...]:
    """Select player viewpoints compatible with a race-specific expert dataset."""
    race = _normalise_race(target_race)
    return tuple(
        player for player in metadata.players
        if (player_id is None or player.player_id == player_id)
        and player.assigned_race == race
        and (not winners_only or player.won)
        and player.apm >= min_apm
    )
