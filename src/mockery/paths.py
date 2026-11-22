from __future__ import annotations

import ipaddress
import os
import sys
import tomllib
from pathlib import Path


def is_loopback_host(host: str, *, allow_testserver: bool = False) -> bool:
    """Return True if `host` is a loopback address (`localhost`, `127.0.0.0/8`, or `::1`)."""
    clean = (host or "").strip().strip("[]").lower()
    if clean == "localhost":
        return True
    if allow_testserver and clean == "testserver":
        return True
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False


def _is_mockery_checkout(directory: Path) -> bool:
    """Return True only when `directory` is an actual Mockery source checkout."""
    pyproject = directory / "pyproject.toml"
    if not pyproject.is_file() or not (directory / "src" / "mockery").is_dir():
        return False
    try:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = parsed.get("project")
        return isinstance(project, dict) and str(project.get("name") or "").strip().lower() == "mockery"
    except Exception:
        return False


def resolve_data_root(explicit_root: Path | None = None) -> Path:
    """
    Resolve where Mockery stores `.env.local` and `sessions/`.

    Priority:
      1. Explicit `explicit_root` argument (used by tests and custom callers).
      2. `MOCKERY_HOME` environment variable.
      3. Local Mockery dev checkout detection (requires `pyproject.toml` with
         `project.name == "mockery"` and `src/mockery/`).
      4. Standard OS app-data directory:
         - macOS: `~/Library/Application Support/Mockery`
         - Linux / other: `$XDG_DATA_HOME/mockery` or `~/.local/share/mockery`
    """
    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    env_home = os.getenv("MOCKERY_HOME", "").strip()
    if env_home:
        root = Path(env_home).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    cwd = Path.cwd()
    if _is_mockery_checkout(cwd):
        return cwd

    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "Mockery"
    else:
        xdg = os.getenv("XDG_DATA_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else (Path.home() / ".local" / "share")
        root = base / "mockery"
    root.mkdir(parents=True, exist_ok=True)
    return root
