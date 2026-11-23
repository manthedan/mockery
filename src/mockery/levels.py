"""Role-neutral seniority levels shared by Practice sessions, job listings, and question banks.

The stored value is the label a person reads ("Senior"), so no display mapping is needed.
Company-ladder codes saved by earlier releases or written into user packs ("L5", "EM")
are folded onto the same five levels.
"""

from __future__ import annotations

SENIORITY_LEVELS: tuple[str, ...] = ("Entry", "Mid", "Senior", "Lead", "Manager")
DEFAULT_LEVEL = "Senior"

_LEVEL_ALIASES: dict[str, str] = {
    **{level.lower(): level for level in SENIORITY_LEVELS},
    "l3": "Entry",
    "e3": "Entry",
    "junior": "Entry",
    "l4": "Mid",
    "e4": "Mid",
    "l5": "Senior",
    "e5": "Senior",
    "l6": "Lead",
    "e6": "Lead",
    "l7": "Lead",
    "e7": "Lead",
    "staff": "Lead",
    "staff+": "Lead",
    "principal": "Lead",
    "em": "Manager",
}


def normalize_level(raw: object) -> str:
    """Return the seniority level for a known value or ladder code; other text passes through trimmed."""
    text = str(raw or "").strip()
    return _LEVEL_ALIASES.get(text.lower(), text)
