from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np

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

MAX_TTS_CHARS = 1600

# Curated crisp macOS local voices (fallback when Kokoro pack is not installed)
MACOS_PREFERRED_VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Tessa", "Alex")


def _clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting and clamp length so spoken questions sound natural."""
    cleaned = re.sub(r"```[\s\S]*?```", " ", text or "")
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_TTS_CHARS]


def _float32_to_wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
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
) -> Path:
    """Stream an HTTPS asset to a temp file, verify SHA256, and atomically move into place."""
    if dest.is_file() and _sha256_file(dest) == expected_sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    own_client = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=20.0),
    )
    try:
        hasher = hashlib.sha256()
        with http.stream("GET", url) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as out:
                for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                    if chunk:
                        out.write(chunk)
                        hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: expected {expected_sha256[:12]}…, got {digest[:12]}…"
            )
        os.replace(tmp_path, dest)
        return dest
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if own_client:
            http.close()


class LocalTTSEngine:
    """
    Local-first TTS engine for Practice Mode AI Interviewer:
      1. Kokoro-82M FP16 ONNX neural voice pack (installed via Add-Ons, runs locally via onnxruntime)
      2. Zero-download macOS native `say` + `afconvert` fallback (22.05kHz mono WAV)
      3. Browser Web Speech API fallback on non-macOS systems
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir / "kokoro"
        self._kokoro: Any | None = None
        self._kokoro_error: str | None = None
        self._kokoro_lock = threading.Lock()
        self._cached_macos_voices: list[str] | None = None

    def _has_kokoro_runtime(self) -> bool:
        try:
            import kokoro_onnx  # noqa: F401

            return True
        except ImportError:
            return False

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
        # Pack is installable whenever kokoro-onnx runtime is present
        return self._has_kokoro_runtime()

    def install_kokoro(self, *, client: httpx.Client | None = None) -> bool:
        if not self._has_kokoro_runtime():
            raise RuntimeError(
                "Package 'kokoro-onnx' is not installed in this environment. "
                "Run `uv sync` to enable local neural voice synthesis."
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        for filename, url, sha256_hex in KOKORO_ASSETS:
            _download_verified_asset(url, self.models_dir / filename, sha256_hex, client=client)

        if not self.is_kokoro_installed():
            raise RuntimeError("Kokoro model files failed to download.")
        if not self._ensure_kokoro_loaded():
            raise RuntimeError(self._kokoro_error or "Kokoro-82M failed to initialize after download.")
        return True

    def uninstall_kokoro(self) -> bool:
        with self._kokoro_lock:
            self._kokoro = None
            self._kokoro_error = None
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir, ignore_errors=True)
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
                self._kokoro = Kokoro(str(model_path), str(voices_path))
                self._kokoro_error = None
                return True
            except Exception as exc:
                self._kokoro_error = str(exc)
                logger.warning("Failed to initialize Kokoro-82M TTS: %s", exc)
                return False

    def list_macos_voices(self) -> list[str]:
        if sys.platform != "darwin" or not shutil.which("say"):
            return []
        if self._cached_macos_voices is not None:
            return list(self._cached_macos_voices)
        available: list[str] = []
        try:
            import subprocess

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
        macos_ok = sys.platform == "darwin" and bool(shutil.which("say")) and bool(shutil.which("afconvert"))
        engine = "kokoro" if kokoro_ready else ("macos_say" if macos_ok else "browser")
        backend_state = derive_tts_backend_state(
            kokoro_installed=kokoro_installed,
            kokoro_runtime=kokoro_runtime,
            kokoro_error=self._kokoro_error,
            macos_say_available=macos_ok,
        )
        if kokoro_ready:
            voices = list(KOKORO_PREFERRED_VOICES)
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
            "backend_state": backend_state.value,
            "kokoro_installed": kokoro_installed,
            "kokoro_available": kokoro_runtime,
            "macos_say_available": macos_ok,
            "voices": voices,
            "default_voice": voices[0] if voices else "",
            "error": self._kokoro_error,
        }

    async def synthesize_wav(
        self,
        text: str,
        *,
        voice: str | None = None,
        rate: float = 1.0,
    ) -> tuple[bytes, str]:
        """Synthesize natural local speech to WAV bytes. Returns `(wav_bytes, engine_used)`."""
        speech = _clean_text_for_speech(text)
        if not speech:
            raise ValueError("Cannot synthesize empty text.")
        clamped_rate = max(0.75, min(1.35, float(rate or 1.0)))

        # 1. Kokoro-82M local neural TTS (if installed)
        if self.is_kokoro_installed() and self._has_kokoro_runtime() and not self._kokoro_error:
            if await asyncio.to_thread(self._ensure_kokoro_loaded):
                kokoro_voice = (
                    voice
                    if voice and voice in KOKORO_PREFERRED_VOICES
                    else (
                        voice
                        if voice and re.fullmatch(r"[ab][fm]_[a-z]{2,16}", voice)
                        else KOKORO_PREFERRED_VOICES[0]
                    )
                )
                lang = "en-gb" if kokoro_voice.startswith(("bf_", "bm_")) else "en-us"

                def _run_kokoro() -> bytes:
                    with self._kokoro_lock:
                        if self._kokoro is None:
                            raise RuntimeError("Kokoro-82M engine was unloaded.")
                        samples, sr = self._kokoro.create(
                            speech,
                            voice=kokoro_voice,
                            speed=clamped_rate,
                            lang=lang,
                        )
                    return _float32_to_wav_bytes(samples, sample_rate=int(sr))

                try:
                    wav_bytes = await asyncio.to_thread(_run_kokoro)
                    return wav_bytes, "kokoro"
                except Exception as exc:
                    with self._kokoro_lock:
                        self._kokoro_error = str(exc)
                        self._kokoro = None
                    logger.warning("Kokoro-82M synthesis failed; falling back to local system voice: %s", exc)

        # 2. Zero-download macOS native `say` + `afconvert`
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
