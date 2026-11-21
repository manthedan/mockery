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

# Curated crisp macOS local voices (fallback when no neural voice pack is installed)
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

    def _has_kokoro_runtime(self) -> bool:
        try:
            import kokoro_onnx  # noqa: F401

            return True
        except ImportError:
            return False

    def _has_phonon_runtime(self) -> bool:
        try:
            import ptts  # noqa: F401

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

    def is_phonon_installed(self) -> bool:
        return all(
            (self.phonon_dir / fname).is_file() and (self.phonon_dir / fname).stat().st_size > 0
            for fname in PHONON_REQUIRED_FILES
        )

    def is_phonon_available(self) -> bool:
        return self._has_phonon_runtime()

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

    def install_phonon(self, *, client: httpx.Client | None = None) -> bool:
        if not self._has_phonon_runtime():
            raise RuntimeError(
                "Package 'ptts' is not installed in this environment. "
                "Run `uv sync` to enable Phonon-31M neural voice synthesis."
            )

        self.phonon_dir.mkdir(parents=True, exist_ok=True)
        for filename, url, sha256_hex in PHONON_ASSETS:
            _download_verified_asset(url, self.phonon_dir / filename, sha256_hex, client=client)

        if not self.is_phonon_installed():
            raise RuntimeError("Phonon-31M model files failed to download.")
        if not self._ensure_phonon_loaded():
            raise RuntimeError(self._phonon_error or "Phonon-31M failed to initialize after download.")
        return True

    def uninstall_phonon(self) -> bool:
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
                self._kokoro = Kokoro(str(model_path), str(voices_path))
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

    async def _try_kokoro(self, speech: str, *, voice: str | None, clamped_rate: float) -> bytes | None:
        if not (self.is_kokoro_installed() and self._has_kokoro_runtime() and not self._kokoro_error):
            return None
        if not await asyncio.to_thread(self._ensure_kokoro_loaded):
            return None
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
            return await asyncio.to_thread(_run_kokoro)
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
        is_kokoro_id = bool(voice and re.fullmatch(r"[ab][fm]_[a-z]{2,16}", voice))
        phonon_voice = (
            voice
            if voice
            and not is_kokoro_id
            and (voice in PHONON_PREFERRED_VOICES or re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{1,31}", voice))
            else PHONON_PREFERRED_VOICES[0]
        )

        def _run_phonon() -> bytes:
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
            return _float32_to_wav_bytes(samples, sample_rate=sr)

        try:
            return await asyncio.to_thread(_run_phonon)
        except Exception as exc:
            with self._phonon_lock:
                self._phonon_error = str(exc)
                self._phonon = None
            logger.warning("Phonon-31M synthesis failed; falling back to next voice engine: %s", exc)
            return None

    async def synthesize_wav(
        self,
        text: str,
        *,
        voice: str | None = None,
        rate: float = 1.0,
        engine: str | None = None,
    ) -> tuple[bytes, str]:
        """Synthesize natural local speech to WAV bytes. Returns `(wav_bytes, engine_used)`."""
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

        # Infer engine from voice name when caller passes a specific voice without explicit engine
        if not requested_engine and clean_voice:
            if clean_voice in PHONON_PREFERRED_VOICES:
                requested_engine = "phonon"
            elif clean_voice in KOKORO_PREFERRED_VOICES or re.fullmatch(r"[ab][fm]_[a-z]{2,16}", clean_voice):
                requested_engine = "kokoro"

        order = ("phonon", "kokoro") if requested_engine == "phonon" else ("kokoro", "phonon")
        for eng_name in order:
            if eng_name == "kokoro":
                wav = await self._try_kokoro(speech, voice=clean_voice, clamped_rate=clamped_rate)
                if wav is not None:
                    return wav, "kokoro"
            elif eng_name == "phonon":
                wav = await self._try_phonon(speech, voice=clean_voice)
                if wav is not None:
                    return wav, "phonon"

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
