from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .asr import parakeet_snapshot_is_complete
from .downloads import (
    DownloadCancelledError,
    DownloadProgress,
    format_download_error,
    make_hf_progress_tqdm_class,
)
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
        self._progress: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_op_ids: dict[str, int] = {}
        self._op_seq: int = 0
        self._server_epoch: str = secrets.token_hex(6)
        self._revision: int = 0
        self._pack_locks: dict[str, asyncio.Lock] = {}
        self._install_tasks: dict[str, asyncio.Task[Any]] = {}
        self._modules: dict[str, bool] = self._load_modules()
        self._on_change: Callable[[str | None, str | None], Any] | None = None
        self._post_install_hook: Callable[[str], Any] | None = None

    def set_change_listener(self, listener: Callable[[str | None, str | None], Any] | None) -> None:
        self._on_change = listener

    def set_post_install_hook(self, hook: Callable[[str], Any] | None) -> None:
        self._post_install_hook = hook

    def _bump_revision(self) -> int:
        self._revision += 1
        return self._revision

    def _emit_change(self, pack_id: str | None = None, event: str | None = None) -> None:
        cb = self._on_change
        if cb is None:
            return
        try:
            res = cb(pack_id, event)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(res)
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                except RuntimeError:
                    res.close()
        except Exception:
            pass

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
        self._bump_revision()
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
                prog = self._progress.get(pid)
                if prog is not None:
                    s["progress"] = dict(prog)
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
            "server_epoch": self._server_epoch,
            "revision": self._revision,
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

    def _is_pack_installed(self, pack_id: str) -> bool:
        if pack_id == "rampart_pii":
            return bool(self.redactor.is_downloaded())
        if pack_id == "parakeet_v2":
            return _hf_repo_cached(
                "mlx-community/parakeet-tdt-0.6b-v2",
                bypass_cache=True,
                snapshot_validator=parakeet_snapshot_is_complete,
            )
        if pack_id == "parakeet_v3":
            return _hf_repo_cached(
                "mlx-community/parakeet-tdt-0.6b-v3",
                bypass_cache=True,
                snapshot_validator=parakeet_snapshot_is_complete,
            )
        if pack_id in self.extra_handlers:
            return bool(self.extra_handlers[pack_id]["is_installed"]())
        return False

    def _is_pack_known(self, pack_id: str) -> bool:
        return pack_id in ("rampart_pii", "parakeet_v2", "parakeet_v3") or pack_id in self.extra_handlers

    @staticmethod
    def _call_install_handler(
        fn: Callable[..., Any],
        *,
        on_progress: Callable[[dict[str, Any]], None],
        cancel_event: threading.Event,
    ) -> Any:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn()
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: dict[str, Any] = {}
        if has_var_kw or "on_progress" in params:
            kwargs["on_progress"] = on_progress
        if has_var_kw or "cancel_event" in params:
            kwargs["cancel_event"] = cancel_event
        return fn(**kwargs)

    async def _run_install_locked(
        self,
        pack_id: str,
        *,
        op_id: int,
        cancel_event: threading.Event,
    ) -> None:
        if not self._is_pack_installable(pack_id):
            raise ValueError(f"Unknown or unavailable add-on pack '{pack_id}'")
        loop = asyncio.get_running_loop()

        def _on_progress_from_thread(prog: dict[str, Any]) -> None:
            def _apply() -> None:
                if self._active_op_ids.get(pack_id) != op_id or pack_id not in self._installing:
                    return
                self._progress[pack_id] = dict(prog)
                phase = prog.get("phase")
                retry_note = prog.get("retry_note")
                pct = prog.get("percent")
                if cancel_event.is_set() and prog.get("cancellable", True):
                    self._installing[pack_id] = "Cancelling…"
                elif phase == "retrying" and retry_note:
                    self._installing[pack_id] = str(retry_note)
                elif phase == "verifying":
                    self._installing[pack_id] = "Verifying…"
                elif phase == "loading":
                    self._installing[pack_id] = "Initializing…"
                elif isinstance(pct, (int, float)) and pct > 0:
                    self._installing[pack_id] = f"Downloading {int(round(pct))}%…"
                else:
                    self._installing[pack_id] = "Downloading…"
                self._bump_revision()
                self._emit_change(pack_id, "progress")

            try:
                loop.call_soon_threadsafe(_apply)
            except RuntimeError:
                pass

        lock = self._get_pack_lock(pack_id)
        terminal_event = "installed"
        async with lock:
            try:
                if cancel_event.is_set():
                    raise DownloadCancelledError("Download cancelled.")
                self._errors.pop(pack_id, None)
                self._installing[pack_id] = "Downloading…"
                self._progress[pack_id] = DownloadProgress(phase="downloading").to_dict()
                self._bump_revision()
                self._emit_change(pack_id, "started")

                if pack_id == "rampart_pii":
                    await self._await_uninterruptible(
                        asyncio.to_thread(
                            self.redactor.ensure_downloaded,
                            on_progress=_on_progress_from_thread,
                            cancel_event=cancel_event,
                        )
                    )
                elif pack_id in ("parakeet_v2", "parakeet_v3"):
                    repo_id = (
                        "mlx-community/parakeet-tdt-0.6b-v2"
                        if pack_id == "parakeet_v2"
                        else "mlx-community/parakeet-tdt-0.6b-v3"
                    )
                    approx_mb = 600 if pack_id == "parakeet_v2" else 620
                    from huggingface_hub import snapshot_download

                    tqdm_cls = make_hf_progress_tqdm_class(
                        _on_progress_from_thread,
                        approx_total_bytes=approx_mb * 1024 * 1024,
                        cancel_event=cancel_event,
                    )

                    def _dl_parakeet() -> Any:
                        supports_tqdm = False
                        supports_allow = False
                        try:
                            params = inspect.signature(snapshot_download).parameters
                            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                            supports_tqdm = has_var_kw or "tqdm_class" in params
                            supports_allow = has_var_kw or "allow_patterns" in params
                        except (TypeError, ValueError):
                            pass
                        dl_kwargs: dict[str, Any] = {}
                        if supports_allow:
                            dl_kwargs["allow_patterns"] = ["config.json", "model.safetensors"]
                        if supports_tqdm:
                            dl_kwargs["tqdm_class"] = tqdm_cls
                        res = snapshot_download(repo_id, **dl_kwargs)
                        if cancel_event.is_set():
                            raise DownloadCancelledError("Download cancelled.")
                        _on_progress_from_thread(
                            DownloadProgress(phase="verifying", percent=100.0, cancellable=False).to_dict()
                        )
                        return res

                    await self._await_uninterruptible(asyncio.to_thread(_dl_parakeet))
                    if cancel_event.is_set():
                        raise DownloadCancelledError("Download cancelled.")
                    _invalidate_hf_repo_cache(repo_id)
                else:
                    fn = self.extra_handlers[pack_id]["install"]
                    res = self._call_install_handler(
                        fn,
                        on_progress=_on_progress_from_thread,
                        cancel_event=cancel_event,
                    )
                    if asyncio.iscoroutine(res):
                        await self._await_uninterruptible(res)

                hook = self._post_install_hook
                if hook is not None:
                    hook_res = hook(pack_id)
                    if asyncio.iscoroutine(hook_res):
                        await self._await_uninterruptible(hook_res)
            except DownloadCancelledError:
                terminal_event = "cancelled"
                self._errors.pop(pack_id, None)
                logger.info("Cancelled add-on pack install: %s", pack_id)
            except asyncio.CancelledError:
                terminal_event = "cancelled"
                raise
            except Exception as exc:
                terminal_event = "error"
                clean_err = format_download_error(exc)
                self._errors[pack_id] = clean_err
                raise RuntimeError(clean_err) from exc
            finally:
                if pack_id in ("parakeet_v2", "parakeet_v3"):
                    _invalidate_hf_repo_cache()
                if self._active_op_ids.get(pack_id) == op_id:
                    self._active_op_ids.pop(pack_id, None)
                    self._installing.pop(pack_id, None)
                    self._progress.pop(pack_id, None)
                if self._cancel_events.get(pack_id) is cancel_event:
                    self._cancel_events.pop(pack_id, None)
                self._bump_revision()
                self._emit_change(pack_id, terminal_event)

    def _ensure_install_task(self, pack_id: str) -> asyncio.Task[Any]:
        existing = self._install_tasks.get(pack_id)
        if existing is not None and not existing.done():
            return existing

        self._op_seq += 1
        op_id = self._op_seq
        cancel_event = threading.Event()
        # Seed initial installing state synchronously so immediate HTTP 202 snapshots reflect it
        self._errors.pop(pack_id, None)
        self._active_op_ids[pack_id] = op_id
        self._installing[pack_id] = "Downloading…"
        self._progress[pack_id] = DownloadProgress(phase="downloading").to_dict()
        self._cancel_events[pack_id] = cancel_event
        self._bump_revision()

        task = asyncio.create_task(
            self._run_install_locked(pack_id, op_id=op_id, cancel_event=cancel_event)
        )
        self._install_tasks[pack_id] = task

        def _cleanup_done(done_task: asyncio.Task[Any]) -> None:
            if self._install_tasks.get(pack_id) is done_task:
                self._install_tasks.pop(pack_id, None)
            if not done_task.cancelled():
                exc = done_task.exception()
                if exc is not None:
                    logger.warning("Background install for %s failed: %s", pack_id, exc)

        task.add_done_callback(_cleanup_done)
        return task

    def start_install_pack(self, pack_id: str) -> dict[str, Any]:
        """Start installing `pack_id` in the background and return immediately."""
        if not self._is_pack_installable(pack_id):
            raise ValueError(f"Unknown or unavailable add-on pack '{pack_id}'")
        self._ensure_install_task(pack_id)
        return self.status_dict()

    async def install_pack(self, pack_id: str) -> dict[str, Any]:
        if not self._is_pack_installable(pack_id):
            raise ValueError(f"Unknown or unavailable add-on pack '{pack_id}'")
        task = self._ensure_install_task(pack_id)
        await asyncio.shield(task)
        return self.status_dict()

    def cancel_install(self, pack_id: str) -> dict[str, Any]:
        if not self._is_pack_known(pack_id):
            raise ValueError(f"Unknown add-on pack '{pack_id}'")
        ev = self._cancel_events.get(pack_id)
        if ev is not None:
            prog = self._progress.get(pack_id)
            if prog is not None and prog.get("cancellable") is False:
                return self.status_dict()
            ev.set()
            if pack_id in self._installing:
                self._installing[pack_id] = "Cancelling…"
            self._bump_revision()
            self._emit_change(pack_id, "cancelling")
        return self.status_dict()

    async def uninstall_pack(self, pack_id: str) -> dict[str, Any]:
        if not self._is_pack_known(pack_id):
            raise ValueError(f"Unknown add-on pack '{pack_id}'")
        lock = self._get_pack_lock(pack_id)
        terminal_event = "uninstalled"
        async with lock:
            self._errors.pop(pack_id, None)
            self._uninstalling[pack_id] = "Removing…"
            self._bump_revision()
            self._emit_change(pack_id, "uninstalling")
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
                if pack_id in ("parakeet_v2", "parakeet_v3"):
                    _invalidate_hf_repo_cache()
                if self._is_pack_installed(pack_id):
                    raise RuntimeError(f"Could not completely remove {pack_id} from disk.")
            except asyncio.CancelledError:
                terminal_event = "cancelled"
                raise
            except Exception as exc:
                terminal_event = "error"
                self._errors[pack_id] = format_download_error(exc)
                raise
            finally:
                if pack_id in ("parakeet_v2", "parakeet_v3"):
                    _invalidate_hf_repo_cache()
                self._uninstalling.pop(pack_id, None)
                self._bump_revision()
                self._emit_change(pack_id, terminal_event)
        return self.status_dict()
