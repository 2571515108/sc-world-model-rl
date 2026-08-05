"""Race-neutral unit semantics used by replay datasets.

The IDs are stable StarCraft II raw-unit identifiers.  They intentionally
cover broad categories rather than every patch-specific unit variant, so an
unknown unit is represented as ``other`` instead of being silently treated as
one of the three races' common combat units.
"""

from __future__ import annotations

from collections.abc import Mapping


RACE_IDS = {"Unknown": 0, "Terran": 1, "Protoss": 2, "Zerg": 3}


def normalise_race(value: str) -> str:
    """Return one of the stable public race names."""
    aliases = {"terr": "Terran", "terran": "Terran", "prot": "Protoss", "protoss": "Protoss",
               "zerg": "Zerg", "rand": "Unknown", "random": "Unknown"}
    return aliases.get(str(value).strip().lower(), "Unknown")


def race_id(value: str) -> int:
    """Encode a race without making a random selection part of the policy input."""
    return RACE_IDS[normalise_race(value)]


# The legacy Terran feature schema names only these units.  Keeping this map
# separate lets existing 106-dimensional Terran checkpoints remain usable.
LEGACY_TERRAN_UNIT_IDS = {
    "marine": 48, "marauder": 51, "reaper": 49, "hellion": 53,
    "tank": 33, "medivac": 54, "viking": 35, "battlecruiser": 57,
}

WORKER_IDS = frozenset({45, 84, 104})  # SCV, Probe, Drone
TOWNHALL_IDS = frozenset({18, 59, 86, 100, 101, 130, 132})
SUPPLY_IDS = frozenset({19, 60, 106})  # Depot, Pylon, Overlord
GAS_IDS = frozenset({20, 61, 88})      # Refinery, Assimilator, Extractor

# ``basic_production`` means the first army-producing structure.  Later
# structures are placed in ``tech_production`` or ``tech``; this is more
# portable across P/T/Z than a Terran building-name vector.
BUILDING_CATEGORIES: dict[str, frozenset[int]] = {
    "base": TOWNHALL_IDS,
    "supply": SUPPLY_IDS,
    "gas": GAS_IDS,
    "basic_production": frozenset({21, 62, 89}),
    "tech_production": frozenset({27, 28, 67, 71, 91, 92}),
    "tech": frozenset({22, 29, 30, 31, 32, 63, 65, 68, 69, 70, 72, 90, 93, 94, 95, 96, 97}),
    "defense": frozenset({23, 24, 25, 26, 66, 98, 99}),
    "addon": frozenset({5, 6, 37, 38, 39, 40, 41, 42}),
}

COMBAT_CATEGORIES = (
    "light_ground", "armored_ground", "air", "support", "detector", "spellcaster", "transport", "capital",
)
COMBAT_CATEGORY_BY_ID: dict[int, str] = {
    # Terran.
    48: "light_ground", 49: "light_ground", 53: "light_ground", 51: "armored_ground", 33: "armored_ground",
    52: "armored_ground", 55: "armored_ground", 35: "air", 56: "detector", 54: "support", 57: "capital",
    58: "capital", 59: "air",
    # Protoss.
    73: "light_ground", 74: "light_ground", 83: "armored_ground", 4: "armored_ground", 75: "spellcaster",
    76: "spellcaster", 77: "support", 78: "air", 79: "capital", 80: "air", 81: "transport", 82: "detector",
    # Zerg.
    105: "light_ground", 110: "armored_ground", 107: "armored_ground", 108: "air", 109: "capital",
    111: "spellcaster", 112: "air", 114: "capital", 126: "support", 127: "spellcaster", 129: "detector",
}

# Mineral + gas values are only used as bounded estimates.  The fallback of
# 50 maintains a nonzero signal for units introduced in a later patch.
UNIT_VALUES: Mapping[int, float] = {
    45: 50.0, 84: 50.0, 104: 50.0, 48: 50.0, 49: 50.0, 51: 125.0, 53: 100.0, 33: 275.0,
    54: 200.0, 35: 225.0, 56: 300.0, 57: 700.0, 73: 100.0, 74: 125.0, 83: 275.0, 4: 500.0,
    75: 200.0, 76: 250.0, 77: 100.0, 78: 150.0, 79: 600.0, 80: 400.0, 81: 250.0, 82: 75.0,
    105: 50.0, 110: 100.0, 107: 100.0, 108: 200.0, 109: 500.0, 111: 250.0, 112: 250.0,
    114: 550.0, 126: 150.0,
}


def building_category(unit_type: int) -> str | None:
    """Return the broad building category for a raw unit type, if known."""
    return next((name for name, identifiers in BUILDING_CATEGORIES.items() if int(unit_type) in identifiers), None)


def combat_category(unit_type: int) -> str | None:
    """Return a cross-race combat category for a raw unit type, if known."""
    return COMBAT_CATEGORY_BY_ID.get(int(unit_type))


def unit_value(unit_type: int) -> float:
    """Return a conservative visible-unit value estimate."""
    return float(UNIT_VALUES.get(int(unit_type), 50.0))
