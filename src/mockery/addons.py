from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .asr import parakeet_snapshot_is_complete
from .redactor import RAMPART_APPROX_MB, RAMPART_REPO_ID, RampartRedactor
from .tts import KOKORO_APPROX_MB, KOKORO_REPO_ID, PHONON_APPROX_MB, PHONON_REPO_ID
from .state_machines import (
    AddonPackState,
    compute_allowed_addon_pack_events,
    derive_addon_pack_state,
)

logger = logging.getLogger("mockery.addons")

_HF_CACHE_TTL_SECONDS = 2.0
_hf_repo_status_cache: dict[tuple[str, str], tuple[float, bool]] = {}


DEFAULT_MODULES: dict[str, bool] = {
    "practice_mode": True,
    "system_design_studio": True,
    "coding_studio": True,
}

MODULE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "practice_mode": {
        "label": "AI Mock Practice",
        "description": "Solo mock interviews shaped by your resume and target job, with a supportive coach that leads with your strengths.",
    },
    "system_design_studio": {
        "label": "System Design Whiteboard",
        "description": "Sketch your architecture while you talk. Your interviewer sees the diagram and asks about what you actually drew.",
    },
    "coding_studio": {
        "label": "Coding Studio",
        "description": "Write and run JavaScript or Python against real tests, with friendly step-by-step coaching when you want it.",
    },
}


def _hf_hub_dir() -> Path:
    hub_env = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_env:
        return Path(hub_env)
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub"


def _invalidate_hf_repo_cache(repo_id: str | None = None) -> None:
    if repo_id is None:
        _hf_repo_status_cache.clear()
        return
    prefix = f"{repo_id}:"
    for key in [k for k in _hf_repo_status_cache if k[1] == repo_id or k[1].startswith(prefix)]:
        _hf_repo_status_cache.pop(key, None)


def _hf_repo_has_files(
    repo_dir: Path,
    *,
    snapshot_validator: Callable[[Path], bool] | None = None,
) -> bool:
    if not repo_dir.is_dir():
        return False
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        if snapshot_validator is not None:
            ref_main = repo_dir / "refs" / "main"
            if ref_main.is_file():
                try:
                    rev = ref_main.read_text(encoding="utf-8").strip()
                    if rev:
                        return snapshot_validator(snapshots_dir / rev)
                except OSError:
                    pass
        try:
            with os.scandir(snapshots_dir) as snap_it:
                for snap in snap_it:
                    if not snap.is_dir():
                        continue
                    if snapshot_validator is not None:
                        if snapshot_validator(Path(snap.path)):
                            return True
                        continue
                    try:
                        with os.scandir(snap.path) as file_it:
                            for entry in file_it:
                                try:
                                    if entry.is_file(follow_symlinks=True) and entry.stat(follow_symlinks=True).st_size > 0:
                                        return True
                                except OSError:
                                    continue
                    except OSError:
                        continue
        except OSError:
            return False
        return False
    if snapshot_validator is not None:
        return snapshot_validator(repo_dir)
    # Fallback for non-standard or minimal cache dirs without a snapshots/ subdirectory
    try:
        with os.scandir(repo_dir) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=True) and entry.stat(follow_symlinks=True).st_size > 0:
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def _hf_repo_cached(
    repo_id: str,
    *,
    bypass_cache: bool = False,
    snapshot_validator: Callable[[Path], bool] | None = None,
) -> bool:
    hub_dir = _hf_hub_dir()
    validator_tag = getattr(snapshot_validator, "__name__", "") if snapshot_validator is not None else ""
    cache_key = (str(hub_dir), f"{repo_id}:{validator_tag}")
    now = time.monotonic()
    if not bypass_cache:
        cached = _hf_repo_status_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < _HF_CACHE_TTL_SECONDS:
            return cached[1]

    repo_dir = hub_dir / f"models--{repo_id.replace('/', '--')}"
    result = _hf_repo_has_files(repo_dir, snapshot_validator=snapshot_validator)
    _hf_repo_status_cache[cache_key] = (now, result)
    return result


def _hf_repo_delete(repo_id: str) -> bool:
    _invalidate_hf_repo_cache(repo_id)
    hub_dir = _hf_hub_dir()
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir(cache_dir=hub_dir)
        revs = []
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                revs.extend(r.commit_hash for r in repo.revisions)
        if revs:
            cache.delete_revisions(*revs).execute()
            _invalidate_hf_repo_cache(repo_id)
            return True
    except Exception as exc:
        logger.warning("Failed to delete HF repo %s via scan_cache_dir: %s", repo_id, exc)
    repo_dir = hub_dir / f"models--{repo_id.replace('/', '--')}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
        _invalidate_hf_repo_cache(repo_id)
        return True
    return False


class AddonManager:
    """
    Xcode-style Add-Ons & Models manager:
      - Lightweight UI modules (instant enable/disable toggles persisted to `<data_root>/addons.json`)
      - Downloadable local AI packs (Speech models, Rampart PII, Local Neural TTS, Offline Python runner)
    """

    def __init__(
        self,
        data_root: Path,
        redactor: RampartRedactor,
        *,
        extra_pack_handlers: dict[str, dict[str, Callable[..., Any]]] | None = None,
    ) -> None:
        self.data_root = data_root
        self.config_path = data_root / "addons.json"
        self.redactor = redactor
        self.extra_handlers: dict[str, dict[str, Callable[..., Any]]] = extra_pack_handlers or {}
        self._installing: dict[str, str] = {}
        self._uninstalling: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._pack_locks: dict[str, asyncio.Lock] = {}
        self._install_tasks: dict[str, asyncio.Task[Any]] = {}
        self._modules: dict[str, bool] = self._load_modules()

    def _get_pack_lock(self, pack_id: str) -> asyncio.Lock:
        lock = self._pack_locks.get(pack_id)
        if lock is None:
            lock = asyncio.Lock()
            self._pack_locks[pack_id] = lock
        return lock

    def register_pack_handler(
        self,
        pack_id: str,
        *,
        is_installed: Callable[[], bool],
        install: Callable[[], Any],
        uninstall: Callable[[], bool],
        available: bool = True,
    ) -> None:
        self.extra_handlers[pack_id] = {
            "is_installed": is_installed,
            "install": install,
            "uninstall": uninstall,
            "available": lambda: available,
        }

    def _load_modules(self) -> dict[str, bool]:
        mods = dict(DEFAULT_MODULES)
        if self.config_path.is_file():
            try:
                raw = json.loads(self.config_path.read_text(encoding="utf-8"))
                saved = raw.get("modules") if isinstance(raw, dict) else None
                if isinstance(saved, dict):
                    for k in DEFAULT_MODULES:
                        if k in saved:
                            mods[k] = bool(saved[k])
            except Exception:
                pass
        return mods

    def _save_modules(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"modules": self._modules}, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.config_path)

    def update_modules(self, updates: dict[str, bool]) -> dict[str, bool]:
        for k, v in updates.items():
            if k in DEFAULT_MODULES:
                self._modules[k] = bool(v)
        self._save_modules()
        return dict(self._modules)

    def _pack_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = [
            {
                "id": "rampart_pii",
                "name": "Resume Privacy Scrubber",
                "model": "Rampart",
                "category": "privacy",
                "size_mb": RAMPART_APPROX_MB,
                "description": "Finds names, employers, and contact details on this Mac, so they are hidden before anything reaches a cloud AI.",
                "repo_id": RAMPART_REPO_ID,
                "installed": self.redactor.is_downloaded(),
                "available": True,
            },
            {
                "id": "parakeet_v2",
                "name": "English Speech Recognition",
                "model": "Parakeet TDT 0.6B v2",
                "category": "speech",
                "size_mb": 600,
                "description": "Offline speech recognition for English. Turns what you say into text without leaving this Mac.",
                "repo_id": "mlx-community/parakeet-tdt-0.6b-v2",
                "installed": _hf_repo_cached(
                    "mlx-community/parakeet-tdt-0.6b-v2",
                    snapshot_validator=parakeet_snapshot_is_complete,
                ),
                "available": True,
            },
            {
                "id": "parakeet_v3",
                "name": "Multilingual Speech Recognition",
                "model": "Parakeet TDT 0.6B v3",
                "category": "speech",
                "size_mb": 620,
                "description": "Offline speech recognition for 25 European languages.",
                "repo_id": "mlx-community/parakeet-tdt-0.6b-v3",
                "installed": _hf_repo_cached(
                    "mlx-community/parakeet-tdt-0.6b-v3",
                    snapshot_validator=parakeet_snapshot_is_complete,
                ),
                "available": True,
            },
            {
                "id": "phonon_tts",
                "name": "Phonon Fast Voice",
                "model": "Phonon-31M",
                "category": "voice",
                "size_mb": PHONON_APPROX_MB,
                "description": "A lightweight interviewer voice that answers instantly, fully offline.",
                "repo_id": PHONON_REPO_ID,
                "installed": (
                    bool(self.extra_handlers["phonon_tts"]["is_installed"]())
                    if "phonon_tts" in self.extra_handlers
                    else False
                ),
                "available": (
                    bool(self.extra_handlers["phonon_tts"]["available"]())
                    if "phonon_tts" in self.extra_handlers
                    else False
                ),
            },
            {
                "id": "kokoro_tts",
                "name": "Kokoro Studio Voice",
                "model": "Kokoro-82M",
                "category": "voice",
                "size_mb": KOKORO_APPROX_MB,
                "description": "A warm, natural interviewer voice with US and UK accents. Interrupt it any time, just like a real conversation.",
                "repo_id": KOKORO_REPO_ID,
                "installed": (
                    bool(self.extra_handlers["kokoro_tts"]["is_installed"]())
                    if "kokoro_tts" in self.extra_handlers
                    else False
                ),
                "available": (
                    bool(self.extra_handlers["kokoro_tts"]["available"]())
                    if "kokoro_tts" in self.extra_handlers
                    else False
                ),
            },
            {
                "id": "pyodide_python",
                "name": "Python for Coding Rounds",
                "model": "Pyodide",
                "category": "coding",
                "size_mb": 14,
                "description": "Runs your Python tests safely on this Mac during coding practice. No internet needed.",
                "repo_id": "",
                "installed": (
                    bool(self.extra_handlers["pyodide_python"]["is_installed"]())
                    if "pyodide_python" in self.extra_handlers
                    else False
                ),
                "available": (
                    bool(self.extra_handlers["pyodide_python"]["available"]())
                    if "pyodide_python" in self.extra_handlers
                    else False
                ),
            },
        ]
        for s in specs:
            pid = s["id"]
            is_installing = pid in self._installing
            is_uninstalling = pid in self._uninstalling
            err_msg = self._errors.get(pid)
            pack_state = derive_addon_pack_state(
                installed=bool(s["installed"]),
                available=bool(s["available"]),
                installing=is_installing,
                uninstalling=is_uninstalling,
                error=err_msg,
            )
            allowed, blocked = compute_allowed_addon_pack_events(
                pack_state,
                installed=bool(s["installed"]),
                available=bool(s["available"]),
            )
            s["state"] = pack_state.value
            s["allowed_events"] = allowed
            s["blocked_reasons"] = blocked
            if pack_state == AddonPackState.INSTALLING:
                s["status_detail"] = self._installing[pid]
            elif pack_state == AddonPackState.UNINSTALLING:
                s["status_detail"] = self._uninstalling[pid]
            elif pack_state == AddonPackState.INSTALLED:
                s["status_detail"] = "Installed"
            elif pack_state == AddonPackState.ERROR:
                s["status_detail"] = "Failed"
            elif pack_state == AddonPackState.COMING_SOON:
                s["status_detail"] = "Not enabled in this build"
            else:
                s["status_detail"] = "Not installed"
            if err_msg:
                s["error"] = err_msg
        return specs

    def get_pack_spec(self, pack_id: str) -> dict[str, Any] | None:
        for spec in self._pack_specs():
            if spec["id"] == pack_id:
                return spec
        return None

    def status_dict(self) -> dict[str, Any]:
        return {
            "modules": [
                {
                    "id": mod_id,
                    "enabled": bool(self._modules.get(mod_id, True)),
                    "label": meta["label"],
                    "description": meta["description"],
                }
                for mod_id, meta in MODULE_DESCRIPTIONS.items()
            ],
            "modules_map": dict(self._modules),
            "packs": self._pack_specs(),
        }

    @staticmethod
    async def _await_uninterruptible(awaitable: Any) -> Any:
        """
        Wait for a critical disk/pack mutation to complete even if the caller is cancelled,
        so pack locks and state flags are never released while threaded IO is still running.
        """
        task = asyncio.ensure_future(awaitable)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    def _is_pack_installable(self, pack_id: str) -> bool:
        if pack_id in ("rampart_pii", "parakeet_v2", "parakeet_v3"):
            return True
        if pack_id in self.extra_handlers:
            return bool(self.extra_handlers[pack_id]["available"]())
        return False

    def _is_pack_known(self, pack_id: str) -> bool:
        return pack_id in ("rampart_pii", "parakeet_v2", "parakeet_v3") or pack_id in self.extra_handlers

    async def _run_install_locked(self, pack_id: str) -> None:
        if not self._is_pack_installable(pack_id):
            raise ValueError(f"Unknown or unavailable add-on pack '{pack_id}'")
        lock = self._get_pack_lock(pack_id)
        async with lock:
            self._errors.pop(pack_id, None)
            self._installing[pack_id] = "Downloading…"
            try:
                if pack_id == "rampart_pii":
                    await self._await_uninterruptible(asyncio.to_thread(self.redactor.ensure_downloaded))
                elif pack_id in ("parakeet_v2", "parakeet_v3"):
                    repo_id = (
                        "mlx-community/parakeet-tdt-0.6b-v2"
                        if pack_id == "parakeet_v2"
                        else "mlx-community/parakeet-tdt-0.6b-v3"
                    )
                    from huggingface_hub import snapshot_download

                    await self._await_uninterruptible(asyncio.to_thread(snapshot_download, repo_id))
                    _invalidate_hf_repo_cache(repo_id)
                else:
                    fn = self.extra_handlers[pack_id]["install"]
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await self._await_uninterruptible(res)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors[pack_id] = str(exc)
                raise
            finally:
                if pack_id in ("parakeet_v2", "parakeet_v3"):
                    _invalidate_hf_repo_cache()
                self._installing.pop(pack_id, None)

    async def install_pack(self, pack_id: str) -> dict[str, Any]:
        if not self._is_pack_installable(pack_id):
            raise ValueError(f"Unknown or unavailable add-on pack '{pack_id}'")
        existing = self._install_tasks.get(pack_id)
        if existing is not None and not existing.done():
            await asyncio.shield(existing)
            return self.status_dict()

        task = asyncio.create_task(self._run_install_locked(pack_id))
        self._install_tasks[pack_id] = task
        try:
            await asyncio.shield(task)
        finally:
            if self._install_tasks.get(pack_id) is task and task.done():
                self._install_tasks.pop(pack_id, None)
        return self.status_dict()

    async def uninstall_pack(self, pack_id: str) -> dict[str, Any]:
        if not self._is_pack_known(pack_id):
            raise ValueError(f"Unknown add-on pack '{pack_id}'")
        lock = self._get_pack_lock(pack_id)
        async with lock:
            self._errors.pop(pack_id, None)
            self._uninstalling[pack_id] = "Removing…"
            try:
                if pack_id == "rampart_pii":
                    await self._await_uninterruptible(asyncio.to_thread(self.redactor.delete_cache))
                elif pack_id in ("parakeet_v2", "parakeet_v3"):
                    repo_id = (
                        "mlx-community/parakeet-tdt-0.6b-v2"
                        if pack_id == "parakeet_v2"
                        else "mlx-community/parakeet-tdt-0.6b-v3"
                    )
                    await self._await_uninterruptible(asyncio.to_thread(_hf_repo_delete, repo_id))
                else:
                    fn = self.extra_handlers[pack_id]["uninstall"]
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await self._await_uninterruptible(res)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors[pack_id] = str(exc)
                raise
            finally:
                if pack_id in ("parakeet_v2", "parakeet_v3"):
                    _invalidate_hf_repo_cache()
                self._uninstalling.pop(pack_id, None)
        return self.status_dict()
