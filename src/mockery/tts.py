from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .downloads import (
    DEFAULT_BACKOFF_BASE_S,
    DEFAULT_MAX_ATTEMPTS,
    DownloadCancelledError,
    MultiFileProgressTracker,
    download_verified_file,
)
from .speech_text import speak_text

logger = logging.getLogger("mockery.tts")

KOKORO_REPO_ID = "thewh1teagle/kokoro-onnx"
KOKORO_RELEASE_TAG = "model-files-v1.1"
KOKORO_APPROX_MB = 183
KOKORO_MODEL_FILENAME = "kokoro-v1.0.fp16.onnx"
KOKORO_VOICES_FILENAME = "voices-v1.0.bin"
KOKORO_MODEL_CANDIDATES = ("kokoro-v1.0.fp16.onnx", "kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx")

# Pinned GitHub Release `model-files-v1.1` assets + SHA256 integrity digests
KOKORO_ASSETS: tuple[tuple[str, str, str], ...] = (
    (
        KOKORO_MODEL_FILENAME,
        f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/{KOKORO_RELEASE_TAG}/{KOKORO_MODEL_FILENAME}",
        "f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96",
    ),
    (
        KOKORO_VOICES_FILENAME,
        f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/{KOKORO_RELEASE_TAG}/{KOKORO_VOICES_FILENAME}",
        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    ),
)

KOKORO_PREFERRED_VOICES: tuple[str, ...] = (
    "af_heart",
    "af_bella",
    "af_sarah",
    "am_michael",
    "am_fenrir",
    "bf_emma",
    "bm_george",
)

PHONON_REPO_ID = "gradium/phonon-31m-en"
PHONON_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f"
PHONON_APPROX_MB = 62
PHONON_MODEL_FILENAME = "model.q8.gguf"

# Pinned Hugging Face commit `92741c5b...` assets for gradium/phonon-31m-en (Apache 2.0) + SHA256 digests
PHONON_ASSETS: tuple[tuple[str, str, str], ...] = (
    (
        "config.json",
        f"https://huggingface.co/{PHONON_REPO_ID}/resolve/{PHONON_REVISION}/config.json",
        "b5d0f88353638370d94035b7d73d7d6577c0554d78f8d7ecfd40490aae39ae85",
    ),
    (
        "tokenizer.json",
        f"https://huggingface.co/{PHONON_REPO_ID}/resolve/{PHONON_REVISION}/tokenizer.json",
        "691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936",
    ),
    (
        "default-voice.safetensors",
        f"https://huggingface.co/{PHONON_REPO_ID}/resolve/{PHONON_REVISION}/default-voice.safetensors",
        "dc537478eaadc1a5257fcda2e3501ab81e14c111a5d95a5838f350db2afd0f1a",
    ),
    (
        PHONON_MODEL_FILENAME,
        f"https://huggingface.co/{PHONON_REPO_ID}/resolve/{PHONON_REVISION}/{PHONON_MODEL_FILENAME}",
        "a7d9aad2cd6dc66473f24a08bc6d2939030ef4d751ef82349361fbef3272dbae",
    ),
)

PHONON_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "default-voice.safetensors",
    PHONON_MODEL_FILENAME,
)

PHONON_PREFERRED_VOICES: tuple[str, ...] = (
    "Harper",
    "Hadley",
    "Easton",
    "Juniper",
    "Trevor",
    "Maeve",
    "Miles",
    "default",
)

MAX_TTS_CHARS = 1600
MAX_TTS_CACHE_ENTRIES = 32
MAX_TTS_CACHE_BYTES = 16 * 1024 * 1024
KOKORO_INTER_SENTENCE_SILENCE_MS = 140

# Curated crisp macOS local voices (fallback when no neural voice pack is installed)
MACOS_PREFERRED_VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Tessa", "Alex")

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^\s*#+\s*", flags=re.MULTILINE)
_MULTI_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")
_KOKORO_VOICE_ID_RE = re.compile(r"[ab][fm]_[a-z]{2,16}")
_PHONON_VOICE_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,31}")


@functools.lru_cache(maxsize=1)
def optimal_ort_threads() -> int:
    """
    Pick the optimal ONNX Runtime CPU intra-op thread count.
    Empirically on Apple Silicon (e.g. M4 4P+6E), using all logical cores causes ~35% slowdown
    from barrier contention, while `min(8, max(p_cores, min(6, total - 2)))` hits the throughput peak
    for both Kokoro-82M and Rampart Q4 ONNX.
    """
    total = os.cpu_count() or 4
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            p_cores = int((proc.stdout or "").strip())
            if 1 <= p_cores <= 16:
                return max(1, min(8, max(p_cores, min(6, max(1, total - 2)))))
        except Exception:
            pass
    return max(1, min(6, total))


def _clean_text_for_speech(text: str) -> str:
    """Strip markdown, read inline code the way a person would say it, and clamp length."""
    cleaned = _CODE_BLOCK_RE.sub(" ", text or "")
    cleaned = speak_text(cleaned)
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    cleaned = _ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _LINK_RE.sub(r"\1", cleaned)
    cleaned = _HEADING_RE.sub("", cleaned)
    cleaned = _MULTI_WS_RE.sub(" ", cleaned).strip()
    return cleaned[:MAX_TTS_CHARS]


def _split_speech_sentences(text: str, *, min_chunk_chars: int = 24, max_chunk_chars: int = 220) -> list[str]:
    """
    Split cleaned speech into natural sentence chunks for low Time-To-First-Audio streaming.
    Merges short opening fragments (< min_chunk_chars) with the following sentence to preserve
    natural prosody, and splits unusually long single sentences at clause boundaries.
    """
    clean = (text or "").strip()
    if not clean:
        return []
    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(clean) if p.strip()]
    if not raw_parts:
        return [clean]

    expanded: list[str] = []
    for part in raw_parts:
        if len(part) <= max_chunk_chars:
            expanded.append(part)
            continue
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(part) if c.strip()]
        buf = ""
        for cl in clauses:
            cand = f"{buf} {cl}".strip() if buf else cl
            if buf and len(cand) > max_chunk_chars:
                expanded.append(buf)
                buf = cl
            else:
                buf = cand
        if buf:
            expanded.append(buf)

    merged: list[str] = []
    pending = ""
    for chunk in expanded:
        if pending:
            chunk = f"{pending} {chunk}".strip()
            pending = ""
        if len(chunk) < min_chunk_chars:
            pending = chunk
        else:
            merged.append(chunk)
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}".strip()
        else:
            merged.append(pending)
    return merged


def _float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return b""
    np.clip(arr, -1.0, 1.0, out=arr) if arr.flags.writeable else None
    clipped = arr if arr.flags.writeable else np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified_asset(
    url: str,
    dest: Path,
    expected_sha256: str,
    *,
    client: httpx.Client | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    progress_cb: Callable[[int, int | None], None] | None = None,
    retry_cb: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Stream an HTTPS asset with retries/resume, verify SHA256, and atomically move into place."""
    path, _size = download_verified_file(
        url,
        dest,
        expected_sha256,
        client=client,
        max_attempts=max_attempts,
        backoff_base_s=backoff_base_s,
        progress_cb=progress_cb,
        retry_cb=retry_cb,
        cancel_event=cancel_event,
    )
    return path


class LocalTTSEngine:
    """
    Local-first TTS engine for Practice Mode AI Interviewer:
      1. Kokoro-82M FP16 ONNX neural voice pack (~183 MB, rich studio tone)
      2. Gradium Phonon-31M q8_0 neural voice pack (~62 MB, Apache 2.0, ~30x real-time on CPU)
      3. Zero-download macOS native `say` + `afconvert` fallback (22.05kHz mono WAV)
      4. Browser Web Speech API fallback on non-macOS systems
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_root = models_dir
        self.models_dir = models_dir / "kokoro"
        self.phonon_dir = models_dir / "phonon"
        self._kokoro: Any | None = None
        self._kokoro_error: str | None = None
        self._kokoro_lock = threading.Lock()
        self._phonon: Any | None = None
        self._phonon_error: str | None = None
        self._phonon_lock = threading.Lock()
        self._cached_macos_voices: list[str] | None = None
        self._cache_lock = threading.Lock()
        self._pcm_cache: OrderedDict[tuple[str, str, int, str], tuple[bytes, int]] = OrderedDict()
        self._pcm_cache_bytes: int = 0

    def _cache_get(self, key: tuple[str, str, int, str]) -> tuple[bytes, int] | None:
        with self._cache_lock:
            hit = self._pcm_cache.get(key)
            if hit is not None:
                self._pcm_cache.move_to_end(key)
            return hit

    def _cache_put(self, key: tuple[str, str, int, str], pcm_bytes: bytes, sample_rate: int) -> None:
        size = len(pcm_bytes)
        if size == 0 or size > MAX_TTS_CACHE_BYTES // 2:
            return
        with self._cache_lock:
            old = self._pcm_cache.pop(key, None)
            if old is not None:
                self._pcm_cache_bytes -= len(old[0])
            self._pcm_cache[key] = (pcm_bytes, sample_rate)
            self._pcm_cache_bytes += size
            while (
                len(self._pcm_cache) > MAX_TTS_CACHE_ENTRIES
                or self._pcm_cache_bytes > MAX_TTS_CACHE_BYTES
            ):
                _, (evicted, _) = self._pcm_cache.popitem(last=False)
                self._pcm_cache_bytes -= len(evicted)

    def _cache_clear_engine(self, engine_name: str) -> None:
        with self._cache_lock:
            drop_keys = [k for k in self._pcm_cache if k[0] == engine_name]
            for k in drop_keys:
                old = self._pcm_cache.pop(k, None)
                if old is not None:
                    self._pcm_cache_bytes -= len(old[0])

    def _has_kokoro_runtime(self) -> bool:
        return importlib.util.find_spec("kokoro_onnx") is not None

    def _has_phonon_runtime(self) -> bool:
        return importlib.util.find_spec("ptts") is not None

    def _resolve_model_path(self) -> Path | None:
        for fname in KOKORO_MODEL_CANDIDATES:
            cand = self.models_dir / fname
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
        return None

    def is_kokoro_installed(self) -> bool:
        voices_path = self.models_dir / KOKORO_VOICES_FILENAME
        return (
            self._resolve_model_path() is not None
            and voices_path.is_file()
            and voices_path.stat().st_size > 0
        )

    def is_kokoro_available(self) -> bool:
        return self._has_kokoro_runtime()

    def is_phonon_installed(self) -> bool:
        return all(
            (self.phonon_dir / fname).is_file() and (self.phonon_dir / fname).stat().st_size > 0
            for fname in PHONON_REQUIRED_FILES
        )

    def is_phonon_available(self) -> bool:
        return self._has_phonon_runtime()

    def install_kokoro(
        self,
        *,
        client: httpx.Client | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    ) -> bool:
        if not self._has_kokoro_runtime():
            raise RuntimeError(
                "Package 'kokoro-onnx' is not installed in this environment. "
                "Run `uv sync` to enable local neural voice synthesis."
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        tracker = MultiFileProgressTracker(
            file_count=len(KOKORO_ASSETS),
            approx_total_bytes=KOKORO_APPROX_MB * 1024 * 1024,
            on_update=on_progress,
        )
        for filename, url, sha256_hex in KOKORO_ASSETS:
            tracker.start_file(filename)
            dest_path = self.models_dir / filename
            _download_verified_asset(
                url,
                dest_path,
                sha256_hex,
                client=client,
                max_attempts=max_attempts,
                backoff_base_s=backoff_base_s,
                progress_cb=tracker.on_file_progress,
                retry_cb=tracker.on_file_retry,
                cancel_event=cancel_event,
            )
            tracker.finish_file(dest_path.stat().st_size if dest_path.is_file() else 0)

        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelledError("Download cancelled.")
        if not self.is_kokoro_installed():
            raise RuntimeError("Kokoro model files failed to download.")
        tracker.set_phase("loading", cancellable=False)
        if not self._ensure_kokoro_loaded():
            raise RuntimeError(self._kokoro_error or "Kokoro-82M failed to initialize after download.")
        return True

    def uninstall_kokoro(self) -> bool:
        self._cache_clear_engine("kokoro")
        with self._kokoro_lock:
            self._kokoro = None
            self._kokoro_error = None
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir, ignore_errors=True)
                return True
            return False

    def install_phonon(
        self,
        *,
        client: httpx.Client | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    ) -> bool:
        if not self._has_phonon_runtime():
            raise RuntimeError(
                "Package 'ptts' is not installed in this environment. "
                "Run `uv sync` to enable Phonon-31M neural voice synthesis."
            )

        self.phonon_dir.mkdir(parents=True, exist_ok=True)
        tracker = MultiFileProgressTracker(
            file_count=len(PHONON_ASSETS),
            approx_total_bytes=PHONON_APPROX_MB * 1024 * 1024,
            on_update=on_progress,
        )
        for filename, url, sha256_hex in PHONON_ASSETS:
            tracker.start_file(filename)
            dest_path = self.phonon_dir / filename
            _download_verified_asset(
                url,
                dest_path,
                sha256_hex,
                client=client,
                max_attempts=max_attempts,
                backoff_base_s=backoff_base_s,
                progress_cb=tracker.on_file_progress,
                retry_cb=tracker.on_file_retry,
                cancel_event=cancel_event,
            )
            tracker.finish_file(dest_path.stat().st_size if dest_path.is_file() else 0)

        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelledError("Download cancelled.")
        if not self.is_phonon_installed():
            raise RuntimeError("Phonon-31M model files failed to download.")
        tracker.set_phase("loading", cancellable=False)
        if not self._ensure_phonon_loaded():
            raise RuntimeError(self._phonon_error or "Phonon-31M failed to initialize after download.")
        return True

    def uninstall_phonon(self) -> bool:
        self._cache_clear_engine("phonon")
        with self._phonon_lock:
            self._phonon = None
            self._phonon_error = None
            if self.phonon_dir.exists():
                shutil.rmtree(self.phonon_dir, ignore_errors=True)
                return True
            return False

    def _ensure_kokoro_loaded(self) -> bool:
        with self._kokoro_lock:
            if self._kokoro is not None:
                return True
            model_path = self._resolve_model_path()
            voices_path = self.models_dir / KOKORO_VOICES_FILENAME
            if model_path is None or not voices_path.is_file() or not self._has_kokoro_runtime():
                return False
            try:
                import onnxruntime as ort
                from kokoro_onnx import Kokoro

                ort.set_default_logger_severity(3)
                if hasattr(Kokoro, "from_session"):
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = optimal_ort_threads()
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    # Explicitly stay on CPUExecutionProvider (CoreML segfaults on Kokoro FP16 ONNX)
                    sess = ort.InferenceSession(
                        str(model_path),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    kokoro = Kokoro.from_session(sess, str(voices_path))
                else:
                    kokoro = Kokoro(str(model_path), str(voices_path))
                # Pre-warm the espeak phonemizer once so the ~580ms init never stalls an interactive turn
                try:
                    kokoro.tokenizer.phonemize("Ready.", lang="en-us")
                except Exception:
                    pass
                self._kokoro = kokoro
                self._kokoro_error = None
                return True
            except Exception as exc:
                self._kokoro_error = str(exc)
                logger.warning("Failed to initialize Kokoro-82M TTS: %s", exc)
                return False

    def _ensure_phonon_loaded(self) -> bool:
        with self._phonon_lock:
            if self._phonon is not None:
                return True
            if not self.is_phonon_installed() or not self._has_phonon_runtime():
                return False
            try:
                import ptts

                self._phonon = ptts.TTS(
                    model=str(self.phonon_dir),
                    quant="q8_0",
                    lang="en",
                )
                self._phonon_error = None
                return True
            except Exception as exc:
                self._phonon_error = str(exc)
                logger.warning("Failed to initialize Phonon-31M TTS: %s", exc)
                return False

    def warm_up_installed(self) -> list[str]:
        """Pre-load installed local neural TTS models off the event loop to eliminate cold-start stalls."""
        warmed: list[str] = []
        if self.is_phonon_installed() and self._has_phonon_runtime():
            if self._ensure_phonon_loaded():
                warmed.append("phonon")
        if self.is_kokoro_installed() and self._has_kokoro_runtime():
            if self._ensure_kokoro_loaded():
                warmed.append("kokoro")
        return warmed

    def list_macos_voices(self) -> list[str]:
        if sys.platform != "darwin" or not shutil.which("say"):
            return []
        if self._cached_macos_voices is not None:
            return list(self._cached_macos_voices)
        available: list[str] = []
        try:
            proc = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            found: set[str] = set()
            for line in proc.stdout.splitlines():
                m = re.match(r"^([A-Za-z][A-Za-z0-9 _\-]{1,28}?)\s+[a-z]{2}_[A-Z]{2}\b", line)
                if m:
                    found.add(m.group(1).strip())
            for pref in MACOS_PREFERRED_VOICES:
                if pref in found:
                    available.append(pref)
            if not available and found:
                available = sorted(found)[:6]
        except Exception:
            available = ["Samantha"]
        self._cached_macos_voices = available or ["Samantha"]
        return list(self._cached_macos_voices)

    def status_dict(self) -> dict[str, Any]:
        from .state_machines import derive_tts_backend_state

        kokoro_installed = self.is_kokoro_installed()
        kokoro_runtime = self._has_kokoro_runtime()
        kokoro_ready = kokoro_installed and kokoro_runtime and not self._kokoro_error
        phonon_installed = self.is_phonon_installed()
        phonon_runtime = self._has_phonon_runtime()
        phonon_ready = phonon_installed and phonon_runtime and not self._phonon_error
        macos_ok = sys.platform == "darwin" and bool(shutil.which("say")) and bool(shutil.which("afconvert"))

        if kokoro_ready:
            engine = "kokoro"
        elif phonon_ready:
            engine = "phonon"
        elif macos_ok:
            engine = "macos_say"
        else:
            engine = "browser"

        backend_state = derive_tts_backend_state(
            kokoro_installed=kokoro_installed,
            kokoro_runtime=kokoro_runtime,
            kokoro_error=self._kokoro_error,
            phonon_installed=phonon_installed,
            phonon_runtime=phonon_runtime,
            phonon_error=self._phonon_error,
            macos_say_available=macos_ok,
        )
        if engine == "kokoro":
            voices = list(KOKORO_PREFERRED_VOICES)
        elif engine == "phonon":
            voices = list(PHONON_PREFERRED_VOICES)
        elif macos_ok:
            voices = (
                list(self._cached_macos_voices)
                if self._cached_macos_voices is not None
                else list(MACOS_PREFERRED_VOICES)
            )
        else:
            voices = []
        return {
            "engine": engine,
            "max_chars": MAX_TTS_CHARS,
            "backend_state": backend_state.value,
            "kokoro_installed": kokoro_installed,
            "kokoro_available": kokoro_runtime,
            "phonon_installed": phonon_installed,
            "phonon_available": phonon_runtime,
            "neural_installed": kokoro_ready or phonon_ready,
            "macos_say_available": macos_ok,
            "engines": {
                "kokoro": {
                    "installed": kokoro_installed,
                    "available": kokoro_runtime,
                    "ready": kokoro_ready,
                    "size_mb": KOKORO_APPROX_MB,
                    "voices": list(KOKORO_PREFERRED_VOICES) if kokoro_ready else [],
                    "error": self._kokoro_error,
                },
                "phonon": {
                    "installed": phonon_installed,
                    "available": phonon_runtime,
                    "ready": phonon_ready,
                    "size_mb": PHONON_APPROX_MB,
                    "voices": list(PHONON_PREFERRED_VOICES) if phonon_ready else [],
                    "error": self._phonon_error,
                },
            },
            "voices": voices,
            "default_voice": voices[0] if voices else "",
            "error": self._kokoro_error or self._phonon_error,
        }

    @staticmethod
    def _resolve_kokoro_voice(voice: str | None) -> tuple[str, str]:
        kokoro_voice = (
            voice
            if voice and voice in KOKORO_PREFERRED_VOICES
            else (
                voice
                if voice and _KOKORO_VOICE_ID_RE.fullmatch(voice)
                else KOKORO_PREFERRED_VOICES[0]
            )
        )
        lang = "en-gb" if kokoro_voice.startswith(("bf_", "bm_")) else "en-us"
        return kokoro_voice, lang

    @staticmethod
    def _resolve_phonon_voice(voice: str | None) -> str:
        is_kokoro_id = bool(voice and _KOKORO_VOICE_ID_RE.fullmatch(voice))
        return (
            voice
            if voice
            and not is_kokoro_id
            and (voice in PHONON_PREFERRED_VOICES or _PHONON_VOICE_ID_RE.fullmatch(voice))
            else PHONON_PREFERRED_VOICES[0]
        )

    @staticmethod
    def _parse_request(
        text: str,
        *,
        voice: str | None,
        rate: float,
        engine: str | None,
    ) -> tuple[str, float, str | None, tuple[str, str]]:
        speech = _clean_text_for_speech(text)
        if not speech:
            raise ValueError("Cannot synthesize empty text.")
        clamped_rate = max(0.75, min(1.35, float(rate or 1.0)))

        requested_engine = (engine or "").strip().lower()
        clean_voice = (voice or "").strip() or None
        if clean_voice and ":" in clean_voice:
            prefix, rest = clean_voice.split(":", 1)
            if prefix.lower() in ("kokoro", "phonon") and rest.strip():
                if not requested_engine:
                    requested_engine = prefix.lower()
                clean_voice = rest.strip()

        if not requested_engine and clean_voice:
            if clean_voice in PHONON_PREFERRED_VOICES:
                requested_engine = "phonon"
            elif clean_voice in KOKORO_PREFERRED_VOICES or _KOKORO_VOICE_ID_RE.fullmatch(clean_voice):
                requested_engine = "kokoro"

        order = ("phonon", "kokoro") if requested_engine == "phonon" else ("kokoro", "phonon")
        return speech, clamped_rate, clean_voice, order

    async def _try_kokoro(self, speech: str, *, voice: str | None, clamped_rate: float) -> bytes | None:
        if not (self.is_kokoro_installed() and self._has_kokoro_runtime() and not self._kokoro_error):
            return None
        if not await asyncio.to_thread(self._ensure_kokoro_loaded):
            return None
        kokoro_voice, lang = self._resolve_kokoro_voice(voice)
        cache_key = ("kokoro", kokoro_voice, int(round(clamped_rate * 100)), speech)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return _pcm16_to_wav_bytes(cached[0], sample_rate=cached[1])

        def _run_kokoro() -> tuple[bytes, int]:
            with self._kokoro_lock:
                if self._kokoro is None:
                    raise RuntimeError("Kokoro-82M engine was unloaded.")
                samples, sr = self._kokoro.create(
                    speech,
                    voice=kokoro_voice,
                    speed=clamped_rate,
                    lang=lang,
                )
            return _float32_to_pcm16_bytes(samples), int(sr)

        try:
            pcm_bytes, sr = await asyncio.to_thread(_run_kokoro)
            self._cache_put(cache_key, pcm_bytes, sr)
            return _pcm16_to_wav_bytes(pcm_bytes, sample_rate=sr)
        except Exception as exc:
            with self._kokoro_lock:
                self._kokoro_error = str(exc)
                self._kokoro = None
            logger.warning("Kokoro-82M synthesis failed; falling back to next voice engine: %s", exc)
            return None

    async def _try_phonon(self, speech: str, *, voice: str | None) -> bytes | None:
        if not (self.is_phonon_installed() and self._has_phonon_runtime() and not self._phonon_error):
            return None
        if not await asyncio.to_thread(self._ensure_phonon_loaded):
            return None
        phonon_voice = self._resolve_phonon_voice(voice)
        cache_key = ("phonon", phonon_voice, 100, speech)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return _pcm16_to_wav_bytes(cached[0], sample_rate=cached[1])

        def _run_phonon() -> tuple[bytes, int]:
            with self._phonon_lock:
                if self._phonon is None:
                    raise RuntimeError("Phonon-31M engine was unloaded.")
                sr = int(getattr(self._phonon, "sample_rate", 24000))
                try:
                    samples = self._phonon.synth(speech, voice=phonon_voice)
                except Exception as voice_exc:
                    if phonon_voice == "default":
                        raise
                    # Offline fallback: use the bundled local default-voice.safetensors if catalog lookup fails
                    logger.info(
                        "Phonon voice %r unavailable (%s); falling back to bundled 'default' voice.",
                        phonon_voice,
                        voice_exc,
                    )
                    samples = self._phonon.synth(speech, voice="default")
            return _float32_to_pcm16_bytes(samples), sr

        try:
            pcm_bytes, sr = await asyncio.to_thread(_run_phonon)
            self._cache_put(cache_key, pcm_bytes, sr)
            return _pcm16_to_wav_bytes(pcm_bytes, sample_rate=sr)
        except Exception as exc:
            with self._phonon_lock:
                self._phonon_error = str(exc)
                self._phonon = None
            logger.warning("Phonon-31M synthesis failed; falling back to next voice engine: %s", exc)
            return None

    async def _try_kokoro_stream(
        self,
        speech: str,
        *,
        voice: str | None,
        clamped_rate: float,
    ) -> tuple[int, AsyncIterator[bytes]] | None:
        if not (self.is_kokoro_installed() and self._has_kokoro_runtime() and not self._kokoro_error):
            return None
        if not await asyncio.to_thread(self._ensure_kokoro_loaded):
            return None
        kokoro_voice, lang = self._resolve_kokoro_voice(voice)
        cache_key = ("kokoro", kokoro_voice, int(round(clamped_rate * 100)), speech)
        cached = self._cache_get(cache_key)
        if cached is not None:
            pcm_bytes, sr = cached

            async def _cached_gen() -> AsyncIterator[bytes]:
                yield pcm_bytes

            return sr, _cached_gen()

        sentences = _split_speech_sentences(speech) or [speech]

        def _synth_one(sent: str) -> tuple[bytes, int]:
            with self._kokoro_lock:
                if self._kokoro is None:
                    raise RuntimeError("Kokoro-82M engine was unloaded.")
                samples, sr = self._kokoro.create(
                    sent,
                    voice=kokoro_voice,
                    speed=clamped_rate,
                    lang=lang,
                )
            return _float32_to_pcm16_bytes(samples), int(sr)

        # Synthesize first sentence eagerly so errors fall back before HTTP headers are sent
        try:
            first_pcm, sr = await asyncio.to_thread(_synth_one, sentences[0])
        except Exception as exc:
            with self._kokoro_lock:
                self._kokoro_error = str(exc)
                self._kokoro = None
            logger.warning("Kokoro-82M stream failed on first sentence; falling back: %s", exc)
            return None

        silence_gap = b"\x00\x00" * int(sr * KOKORO_INTER_SENTENCE_SILENCE_MS / 1000)

        async def _sentence_gen() -> AsyncIterator[bytes]:
            collected: list[bytes] = [first_pcm]
            yield first_pcm
            completed = True
            try:
                for sent in sentences[1:]:
                    chunk_pcm, _ = await asyncio.to_thread(_synth_one, sent)
                    if not chunk_pcm:
                        continue
                    if silence_gap:
                        collected.append(silence_gap)
                        yield silence_gap
                    collected.append(chunk_pcm)
                    yield chunk_pcm
            except asyncio.CancelledError:
                completed = False
                raise
            except Exception as exc:
                completed = False
                logger.warning("Kokoro-82M stream aborted mid-utterance: %s", exc)
            if completed:
                self._cache_put(cache_key, b"".join(collected), sr)

        return sr, _sentence_gen()

    async def _try_phonon_stream(
        self,
        speech: str,
        *,
        voice: str | None,
    ) -> tuple[int, AsyncIterator[bytes]] | None:
        if not (self.is_phonon_installed() and self._has_phonon_runtime() and not self._phonon_error):
            return None
        if not await asyncio.to_thread(self._ensure_phonon_loaded):
            return None
        phonon_voice = self._resolve_phonon_voice(voice)
        cache_key = ("phonon", phonon_voice, 100, speech)
        cached = self._cache_get(cache_key)
        if cached is not None:
            pcm_bytes, sr = cached

            async def _cached_gen() -> AsyncIterator[bytes]:
                yield pcm_bytes

            return sr, _cached_gen()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=6)
        stop_event = threading.Event()
        sr = int(getattr(self._phonon, "sample_rate", 24000))

        def _put_sync(item: tuple[str, Any]) -> bool:
            if stop_event.is_set():
                return False
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stop_event.is_set():
                try:
                    fut.result(timeout=0.05)
                    return True
                except TimeoutError:
                    continue
                except Exception:
                    return False
            fut.cancel()
            return False

        def _producer() -> None:
            try:
                with self._phonon_lock:
                    if self._phonon is None:
                        raise RuntimeError("Phonon-31M engine was unloaded.")
                    has_stream = hasattr(self._phonon, "stream") and callable(self._phonon.stream)
                    voice_to_use = phonon_voice

                    def _iter_voice(vname: str):
                        if has_stream:
                            for arr in self._phonon.stream(speech, voice=vname):
                                yield arr
                        else:
                            yield self._phonon.synth(speech, voice=vname)

                    yielded_any = False
                    try:
                        for arr in _iter_voice(voice_to_use):
                            if stop_event.is_set():
                                break
                            pcm = _float32_to_pcm16_bytes(arr)
                            if pcm:
                                yielded_any = True
                                if not _put_sync(("chunk", pcm)):
                                    break
                    except Exception as voice_exc:
                        if yielded_any or voice_to_use == "default" or stop_event.is_set():
                            raise
                        logger.info(
                            "Phonon stream voice %r unavailable (%s); falling back to bundled 'default' voice.",
                            voice_to_use,
                            voice_exc,
                        )
                        for arr in _iter_voice("default"):
                            if stop_event.is_set():
                                break
                            pcm = _float32_to_pcm16_bytes(arr)
                            if pcm:
                                if not _put_sync(("chunk", pcm)):
                                    break
                _put_sync(("done", None))
            except Exception as exc:
                _put_sync(("error", exc))

        worker = threading.Thread(target=_producer, name="phonon-tts-stream", daemon=True)
        worker.start()

        # Pull the very first chunk eagerly (~29ms) so startup errors fall back cleanly before headers are sent.
        # Guard with BaseException so task cancellation during startup always signals stop_event and releases _phonon_lock.
        try:
            kind, payload = await queue.get()
        except BaseException:
            stop_event.set()
            raise
        if kind == "error":
            stop_event.set()
            def _record_phonon_error() -> None:
                with self._phonon_lock:
                    self._phonon_error = str(payload)
                    self._phonon = None

            await asyncio.to_thread(_record_phonon_error)
            logger.warning("Phonon-31M stream failed on first chunk; falling back: %s", payload)
            return None
        if kind == "done":
            stop_event.set()
            return None

        first_chunk: bytes = payload
        engine_ref = self

        class _ManagedPhononStream:
            """Cycle-free async iterator so `__del__` always triggers `stop_event.set()` if abandoned."""

            def __init__(self) -> None:
                self._closed = False
                self._first_pending: bytes | None = first_chunk
                self._collected: list[bytes] = [first_chunk]
                self._completed = False

            def close(self) -> None:
                if not self._closed:
                    self._closed = True
                    stop_event.set()

            async def aclose(self) -> None:
                self.close()

            def __aiter__(self) -> AsyncIterator[bytes]:
                return self

            async def __anext__(self) -> bytes:
                if self._closed:
                    raise StopAsyncIteration
                if self._first_pending is not None:
                    chunk = self._first_pending
                    self._first_pending = None
                    return chunk
                try:
                    k, val = await queue.get()
                except BaseException:
                    self.close()
                    raise
                if k == "chunk":
                    self._collected.append(val)
                    return val
                if k == "done":
                    self._completed = True
                    self.close()
                    engine_ref._cache_put(cache_key, b"".join(self._collected), sr)
                    raise StopAsyncIteration
                logger.warning("Phonon-31M stream errored mid-utterance: %s", val)
                self.close()
                raise StopAsyncIteration

            def __del__(self) -> None:
                self.close()

        return sr, _ManagedPhononStream()

    async def open_pcm_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        rate: float = 1.0,
        engine: str | None = None,
    ) -> tuple[str, int, AsyncIterator[bytes]]:
        """
        Open a low-TTFA streaming PCM16 (mono little-endian int16) synthesis stream.
        Returns `(engine_used, sample_rate, async_chunk_iterator)`.
        """
        speech, clamped_rate, clean_voice, order = self._parse_request(
            text, voice=voice, rate=rate, engine=engine
        )
        for eng_name in order:
            if eng_name == "phonon":
                res = await self._try_phonon_stream(speech, voice=clean_voice)
                if res is not None:
                    sr, gen = res
                    return "phonon", sr, gen
            elif eng_name == "kokoro":
                res = await self._try_kokoro_stream(speech, voice=clean_voice, clamped_rate=clamped_rate)
                if res is not None:
                    sr, gen = res
                    return "kokoro", sr, gen

        # Fallback: macOS native `say` WAV -> strip header and yield raw PCM16 frames
        wav_bytes, used = await self._synthesize_macos_wav(speech, voice=voice, clamped_rate=clamped_rate)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())

        async def _single_gen() -> AsyncIterator[bytes]:
            yield pcm

        return used, sr, _single_gen()

    async def _synthesize_macos_wav(
        self,
        speech: str,
        *,
        voice: str | None,
        clamped_rate: float,
    ) -> tuple[bytes, str]:
        if sys.platform == "darwin" and shutil.which("say") and shutil.which("afconvert"):
            macos_voices = await asyncio.to_thread(self.list_macos_voices)
            allowed_voices = set(macos_voices)
            chosen_voice = voice if voice in allowed_voices else (macos_voices[0] if macos_voices else None)
            wpm = int(round(185 * clamped_rate))
            with tempfile.TemporaryDirectory(prefix="mockery_tts_") as td:
                aiff_path = os.path.join(td, "speech.aiff")
                wav_path = os.path.join(td, "speech.wav")
                cmd = ["say", "-r", str(wpm), "-o", aiff_path]
                if chosen_voice:
                    cmd.extend(["-v", chosen_voice])
                # Pass text via `--` separator so leading dashes can never be parsed as flags
                cmd.extend(["--", speech])
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise RuntimeError("Local speech synthesis timed out.")
                if proc.returncode != 0:
                    raise RuntimeError(f"macOS say failed: {stderr.decode('utf-8', errors='ignore').strip()}")

                conv = await asyncio.create_subprocess_exec(
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@22050",
                    "-c",
                    "1",
                    aiff_path,
                    wav_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, c_err = await asyncio.wait_for(conv.communicate(), timeout=10.0)
                except asyncio.TimeoutError:
                    conv.kill()
                    await conv.wait()
                    raise RuntimeError("Audio conversion timed out.")
                if conv.returncode != 0 or not os.path.isfile(wav_path):
                    raise RuntimeError(f"afconvert failed: {c_err.decode('utf-8', errors='ignore').strip()}")
                return Path(wav_path).read_bytes(), "macos_say"

        raise RuntimeError("No server-side local TTS backend available on this platform; use browser speech synthesis.")

    async def synthesize_wav(
        self,
        text: str,
        *,
        voice: str | None = None,
        rate: float = 1.0,
        engine: str | None = None,
    ) -> tuple[bytes, str]:
        """Synthesize natural local speech to WAV bytes. Returns `(wav_bytes, engine_used)`."""
        speech, clamped_rate, clean_voice, order = self._parse_request(
            text, voice=voice, rate=rate, engine=engine
        )
        for eng_name in order:
            if eng_name == "kokoro":
                wav = await self._try_kokoro(speech, voice=clean_voice, clamped_rate=clamped_rate)
                if wav is not None:
                    return wav, "kokoro"
            elif eng_name == "phonon":
                wav = await self._try_phonon(speech, voice=clean_voice)
                if wav is not None:
                    return wav, "phonon"

        return await self._synthesize_macos_wav(speech, voice=voice, clamped_rate=clamped_rate)
