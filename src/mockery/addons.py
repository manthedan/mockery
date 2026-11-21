from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from .models import PARAKEET_CHECKPOINTS
from .redactor import RAMPART_APPROX_MB, RAMPART_REPO_ID, RampartRedactor

logger = logging.getLogger("mockery.addons")


DEFAULT_MODULES: dict[str, bool] = {
    "practice_mode": True,
    "system_design_studio": True,
    "coding_studio": True,
}

MODULE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "practice_mode": {
        "label": "AI Practice Interviewer & CARL Coach",
        "description": "Solo mock interviews driven by your resume, target job listing, and curated banks with supportive CARL coaching.",
    },
    "system_design_studio": {
        "label": "System Design Whiteboard Studio",
        "description": "Interactive architecture canvas whose diagram state is read live by the AI interviewer for real bottleneck probes.",
    },
    "coding_studio": {
        "label": "Coding Interview Studio",
        "description": "Live code editor with in-browser JS/Python test execution and step-by-step AI proctoring.",
    },
}


def _hf_repo_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id and repo.size_on_disk > 0:
                return True
    except Exception:
        pass
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    return repo_dir.is_dir() and any(repo_dir.iterdir())


def _hf_repo_delete(repo_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        revs = []
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                revs.extend(r.commit_hash for r in repo.revisions)
        if revs:
            cache.delete_revisions(*revs).execute()
            return True
    except Exception as exc:
        logger.warning("Failed to delete HF repo %s via scan_cache_dir: %s", repo_id, exc)
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
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
                "name": "Rampart Local PII Classifier",
                "category": "privacy",
                "size_mb": RAMPART_APPROX_MB,
                "description": "America.gov's 14.7 MB quantized ONNX MiniLM classifier for local resume scrubbing before cloud LLM calls.",
                "repo_id": RAMPART_REPO_ID,
                "installed": self.redactor.is_downloaded(),
                "available": True,
            },
            {
                "id": "parakeet_v2",
                "name": PARAKEET_CHECKPOINTS["mlx-community/parakeet-tdt-0.6b-v2"],
                "category": "speech",
                "size_mb": 600,
                "description": "Default offline Apple Silicon MLX speech recognition engine for English.",
                "repo_id": "mlx-community/parakeet-tdt-0.6b-v2",
                "installed": _hf_repo_cached("mlx-community/parakeet-tdt-0.6b-v2"),
                "available": True,
            },
            {
                "id": "parakeet_v3",
                "name": PARAKEET_CHECKPOINTS["mlx-community/parakeet-tdt-0.6b-v3"],
                "category": "speech",
                "size_mb": 620,
                "description": "Optional multilingual Apple Silicon MLX speech recognition engine (25 European languages).",
                "repo_id": "mlx-community/parakeet-tdt-0.6b-v3",
                "installed": _hf_repo_cached("mlx-community/parakeet-tdt-0.6b-v3"),
                "available": True,
            },
            {
                "id": "kokoro_tts",
                "name": "Kokoro-82M Neural Voice (Local TTS)",
                "category": "voice",
                "size_mb": 95,
                "description": "Local Apple Silicon neural voice so the AI interviewer speaks questions aloud with instant barge-in (macOS voices work with zero download).",
                "repo_id": "onnx-community/Kokoro-82M-v1.0-ONNX",
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
                "name": "Offline Python Runner (Pyodide WASM)",
                "category": "coding",
                "size_mb": 14,
                "description": "Run Python unit tests sandboxed inside your browser during Coding Studio practice sessions.",
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
            if pid in self._installing:
                s["state"] = "installing"
                s["status_detail"] = self._installing[pid]
            elif s["installed"]:
                s["state"] = "installed"
                s["status_detail"] = "Installed"
            elif not s["available"]:
                s["state"] = "coming_soon"
                s["status_detail"] = "Not enabled in this build"
            else:
                s["state"] = "not_installed"
                s["status_detail"] = "Not installed"
            if pid in self._errors:
                s["error"] = self._errors[pid]
        return specs

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

    async def _run_install_locked(self, pack_id: str) -> None:
        lock = self._get_pack_lock(pack_id)
        async with lock:
            self._errors.pop(pack_id, None)
            self._installing[pack_id] = "Downloading…"
            try:
                if pack_id == "rampart_pii":
                    await asyncio.to_thread(self.redactor.ensure_downloaded)
                elif pack_id in ("parakeet_v2", "parakeet_v3"):
                    repo_id = (
                        "mlx-community/parakeet-tdt-0.6b-v2"
                        if pack_id == "parakeet_v2"
                        else "mlx-community/parakeet-tdt-0.6b-v3"
                    )
                    from huggingface_hub import snapshot_download

                    await asyncio.to_thread(snapshot_download, repo_id)
                elif pack_id in self.extra_handlers and self.extra_handlers[pack_id]["available"]():
                    fn = self.extra_handlers[pack_id]["install"]
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                else:
                    raise ValueError(f"Unknown or unavailable add-on pack '{pack_id}'")
            except Exception as exc:
                self._errors[pack_id] = str(exc)
                raise
            finally:
                self._installing.pop(pack_id, None)

    async def install_pack(self, pack_id: str) -> dict[str, Any]:
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
        lock = self._get_pack_lock(pack_id)
        async with lock:
            self._errors.pop(pack_id, None)
            if pack_id == "rampart_pii":
                await asyncio.to_thread(self.redactor.delete_cache)
            elif pack_id in ("parakeet_v2", "parakeet_v3"):
                repo_id = (
                    "mlx-community/parakeet-tdt-0.6b-v2"
                    if pack_id == "parakeet_v2"
                    else "mlx-community/parakeet-tdt-0.6b-v3"
                )
                await asyncio.to_thread(_hf_repo_delete, repo_id)
            elif pack_id in self.extra_handlers:
                fn = self.extra_handlers[pack_id]["uninstall"]
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            else:
                raise ValueError(f"Unknown add-on pack '{pack_id}'")
        return self.status_dict()
