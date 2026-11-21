from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_data_root(explicit_root: Path | None = None) -> Path:
    """
    Resolve where Mockery stores `.env.local` and `sessions/`.

    Priority:
      1. Explicit `explicit_root` argument (used by tests and custom callers).
      2. `MOCKERY_HOME` environment variable.
      3. Local dev checkout detection: if CWD contains `pyproject.toml`,
         `sessions/`, or `.env.local`, use CWD so developers keep repo-local data.
      4. Standard OS app-data directory:
         - macOS: `~/Library/Application Support/Mockery`
         - Linux / other: `$XDG_DATA_HOME/mockery` or `~/.local/share/mockery`
    """
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()

    env_home = os.getenv("MOCKERY_HOME", "").strip()
    if env_home:
        root = Path(env_home).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() or (cwd / "sessions").is_dir() or (cwd / ".env.local").exists():
        return cwd

    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "Mockery"
    else:
        xdg = os.getenv("XDG_DATA_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else (Path.home() / ".local" / "share")
        root = base / "mockery"
    root.mkdir(parents=True, exist_ok=True)
    return root
