from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
import os
import re
import secrets
import struct
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .addons import AddonManager
from .asr import ParakeetEngine
from .banks import QuestionBankStore
from .diarizer import OnlineTwoSpeakerDiarizer, WeSpeakerEmbedder
from .grouper import QAGrouper
from .llm import (
    InterviewCopilotLLM,
    LLMRuntimeConfig,
    PROVIDER_ENV_KEYS,
    discover_available_providers,
)
from .models import (
    InterviewerSelfRating,
    InterviewSession,
    LLM_PRESETS,
    SessionConfig,
    SpeakerRole,
    Utterance,
    short_id,
)
from .paths import resolve_data_root
from .practice import PracticeSessionStore
from .profiles import CandidateProfileStore
from .pyodide_pack import PyodidePackManager
from .redactor import RampartRedactor
from .routers import build_addons_router, build_practice_router, build_profile_router
from .routers.practice import PRACTICE_SESSION_ID_RE
from .sources import SegmentedSpeech, SessionAudioRouter
from .state_machines import (
    ASRModelState,
    AsyncJobState,
    CardProbeState,
    InvalidStateTransitionError,
    compute_allowed_session_events,
    derive_card_probe_state,
    derive_session_phase,
    export_machine_specs,
    validate_session_invariants,
)
from .storage import SessionAudioArchive, SessionStore
from .tts import LocalTTSEngine
from .vad import SileroVADRuntime

logger = logging.getLogger("mockery")

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]", "testserver"}
MAX_RESUME_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ASR_AUDIO_BYTES = 4 * 1024 * 1024
MAX_ASR_DURATION_SECONDS = 120
MAX_TTS_REQUEST_BYTES = 32 * 1024
MAX_PRACTICE_REQUEST_BYTES = 256 * 1024


def resolve_project_root() -> Path:
    return resolve_data_root()


def is_trusted_origin(
    origin: str | None,
    host_header: str | None,
    *,
    allow_same_host_lan: bool = False,
) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https"):
        return False
    origin_host = (parsed.hostname or "").lower()
    if host_header:
        expected_netloc = host_header.strip().lower()
        if parsed.netloc.lower() != expected_netloc:
            return False
        if allow_same_host_lan:
            return True
    if origin_host not in ALLOWED_LOOPBACK_HOSTS:
        return False
    return True


def _has_valid_token(
    expected_token: str,
    *,
    query_token: str | None,
    cookie_token: str | None,
    auth_header: str | None,
) -> bool:
    candidates: list[str] = []
    if query_token:
        candidates.append(query_token.strip())
    if cookie_token:
        candidates.append(cookie_token.strip())
    if auth_header and auth_header.lower().startswith("bearer "):
        candidates.append(auth_header[7:].strip())
    return any(secrets.compare_digest(c, expected_token) for c in candidates if c)


class LLMSettingsUpdate(BaseModel):
    provider: str
    base_url: str = ""
    fast_model: str = ""
    calibration_model: str = ""
    api_key: str | None = None
    parakeet_model: str | None = None


class LLMConnectionTestRequest(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    fast_model: str | None = None
    calibration_model: str | None = None
    api_key: str | None = None


@functools.lru_cache(maxsize=16)
def _lowpass_kernel(src_rate: int, dst_rate: int) -> np.ndarray:
    """Cached Hann-windowed sinc FIR low-pass kernel before downsampling."""
    cutoff = 0.45 * (float(dst_rate) / float(src_rate))  # normalized to Nyquist=0.5
    taps = min(31, max(7, int(round(src_rate / dst_rate)) * 4 + 1))
    if taps % 2 == 0:
        taps += 1
    half = taps // 2
    n = np.arange(-half, half + 1, dtype=np.float32)
    kernel = (np.sinc(2.0 * cutoff * n) * np.hanning(taps)).astype(np.float32)
    k_sum = float(kernel.sum())
    if k_sum > 0:
        kernel /= k_sum
    return kernel


def _lowpass_antialias(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Lightweight Hann-windowed sinc FIR low-pass filter before downsampling."""
    if src_rate <= dst_rate or pcm.size < 16:
        return pcm
    kernel = _lowpass_kernel(src_rate, dst_rate)
    return np.convolve(pcm, kernel, mode="same").astype(np.float32)


def _decimate_integer_fir(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Evaluate the symmetric Hann-sinc low-pass FIR filter ONLY at kept decimated output samples
    when `src_rate` is an exact integer multiple of `dst_rate` (e.g. 48kHz -> 16kHz, 32kHz -> 16kHz).
    Mathematically equivalent to `_lowpass_antialias(pcm, src_rate, dst_rate)[::factor]`.
    """
    factor = src_rate // dst_rate
    out_len = pcm.size // factor
    if out_len <= 0:
        return np.empty(0, dtype=np.float32)
    if pcm.size < 16:
        return np.ascontiguousarray(pcm[::factor], dtype=np.float32)
    kernel = _lowpass_kernel(src_rate, dst_rate)
    half = kernel.size // 2
    padded = np.pad(pcm, (half, half), mode="constant")
    out = np.zeros(out_len, dtype=np.float32)
    span = out_len * factor
    for k, w in enumerate(kernel):
        out += w * padded[k : k + span : factor]
    return out


def _decode_audio_to_float32_16k(raw_bytes: bytes, content_type: str = "") -> np.ndarray:
    """
    Decode either a standard RIFF/WAVE PCM16 upload or raw 16kHz little-endian int16 mono PCM
    into a contiguous float32 array normalized to [-1.0, 1.0] at 16kHz.
    """
    if not raw_bytes:
        raise ValueError("Audio payload is empty.")
    ct = (content_type or "").lower()
    is_wav = raw_bytes[:4] == b"RIFF" or "wav" in ct
    if is_wav:
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                rate = wf.getframerate()
                nframes = wf.getnframes()
                if channels not in (1, 2):
                    raise ValueError(f"Unsupported channel count: {channels}")
                if sampwidth != 2:
                    raise ValueError(f"Only 16-bit PCM WAV is supported (got {sampwidth * 8}-bit).")
                if rate < 8000 or rate > 96000:
                    raise ValueError(f"Unsupported sample rate: {rate} Hz")
                if nframes <= 0:
                    raise ValueError("WAV audio contains no frames.")
                if nframes > rate * MAX_ASR_DURATION_SECONDS:
                    raise ValueError(f"Audio clip exceeds maximum {MAX_ASR_DURATION_SECONDS}-second limit.")
                frames = wf.readframes(nframes)
        except (wave.Error, EOFError, struct.error, OSError) as exc:
            raise ValueError(f"Invalid WAV audio: {exc}") from exc
        expected_bytes = nframes * channels * sampwidth
        if len(frames) != expected_bytes:
            raise ValueError("Truncated or malformed WAV audio payload.")
        raw_i16 = np.frombuffer(frames, dtype="<i2")
        if channels == 2:
            stereo = raw_i16.reshape(-1, 2)
            pcm = np.add(stereo[:, 0], stereo[:, 1], dtype=np.float32)
            pcm *= np.float32(0.5 / 32768.0)
        else:
            pcm = raw_i16.astype(np.float32)
            pcm *= np.float32(1.0 / 32768.0)
        if rate != 16000 and pcm.size > 1:
            if rate > 16000 and rate % 16000 == 0 and pcm.size % (rate // 16000) == 0:
                pcm = _decimate_integer_fir(pcm, rate, 16000)
            else:
                if rate > 16000:
                    pcm = _lowpass_antialias(pcm, rate, 16000)
                target_len = max(1, int(round(pcm.size * (16000.0 / float(rate)))))
                pos = np.arange(target_len, dtype=np.float64) * (float(pcm.size) / float(target_len))
                np.minimum(pos, float(pcm.size - 1), out=pos)
                i0 = pos.astype(np.intp)
                i1 = np.minimum(i0 + 1, pcm.size - 1)
                frac = (pos - i0).astype(np.float32)
                pcm = pcm[i0] + frac * (pcm[i1] - pcm[i0])
        return np.ascontiguousarray(pcm, dtype=np.float32)

    # Raw 16kHz little-endian int16 mono PCM
    if len(raw_bytes) % 2 != 0:
        raw_bytes = raw_bytes[:-1]
    if not raw_bytes:
        raise ValueError("Audio payload is too short.")
    pcm = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
    pcm *= np.float32(1.0 / 32768.0)
    if pcm.size > 16000 * MAX_ASR_DURATION_SECONDS:
        raise ValueError(f"Audio clip exceeds maximum {MAX_ASR_DURATION_SECONDS}-second limit.")
    return np.ascontiguousarray(pcm, dtype=np.float32)


class AppRuntime:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.llm_config = LLMRuntimeConfig(project_root)
        self.vad_runtime = SileroVADRuntime()
        self.asr = ParakeetEngine(model_id=self.llm_config.parakeet_model)
        self.voice_embedder = WeSpeakerEmbedder()
        self.diarizer = OnlineTwoSpeakerDiarizer(self.voice_embedder)
        self.llm = InterviewCopilotLLM(self.llm_config)
        self.store = SessionStore(project_root / "sessions")
        self.grouper = QAGrouper()
        self.redactor = RampartRedactor()
        self.profiles = CandidateProfileStore(project_root / "profiles", self.redactor)
        self.banks = QuestionBankStore(project_root / "banks")
        self.addons = AddonManager(project_root, self.redactor)
        self.tts = LocalTTSEngine(project_root / "models")
        self.pyodide = PyodidePackManager(project_root / "models" / "pyodide")
        self.practice_store = PracticeSessionStore(project_root / "practice")
        self._practice_locks: dict[str, list[Any]] = {}
        self._asr_http_sem = asyncio.Semaphore(1)
        self._tts_sem = asyncio.Semaphore(2)

        self.addons.register_pack_handler(
            "phonon_tts",
            is_installed=self.tts.is_phonon_installed,
            install=lambda: asyncio.to_thread(self.tts.install_phonon),
            uninstall=lambda: asyncio.to_thread(self.tts.uninstall_phonon),
            available=self.tts.is_phonon_available(),
        )
        self.addons.register_pack_handler(
            "kokoro_tts",
            is_installed=self.tts.is_kokoro_installed,
            install=lambda: asyncio.to_thread(self.tts.install_kokoro),
            uninstall=lambda: asyncio.to_thread(self.tts.uninstall_kokoro),
            available=self.tts.is_kokoro_available(),
        )
        self.addons.register_pack_handler(
            "pyodide_python",
            is_installed=self.pyodide.is_installed,
            install=lambda: asyncio.to_thread(self.pyodide.install),
            uninstall=lambda: asyncio.to_thread(self.pyodide.uninstall),
            available=True,
        )

        self.active_session = InterviewSession()
        self.audio_router = SessionAudioRouter(
            self.vad_runtime, mode=self.active_session.config.mode
        )
        self.audio_archive: SessionAudioArchive | None = None

        # Canonical session-scoped state-machine trackers
        self.is_capturing: bool = False
        self.capture_started_epoch_ms: int | None = None
        self.regroup_job: AsyncJobState = AsyncJobState.IDLE
        self.calibration_job: AsyncJobState = AsyncJobState.IDLE
        self.practice_debrief_jobs: dict[str, AsyncJobState] = {}
        self.warming_card_ids: set[str] = set()
        self.loading_card_ids: set[str] = set()
        self.card_probe_gen: dict[str, int] = {}
        self.role_orientation_checked: set[str] = set()
        self.role_orientation_inflight: set[str] = set()
        self.regrouped_signature: dict[str, tuple[Any, ...]] = {}
        self.session_epoch: int = 0

        # Connected WebSocket senders & active capture owner
        self.ws_senders: set[Any] = set()
        self.capturing_sender: Any | None = None
        self.asr_watcher_task: asyncio.Task[Any] | None = None

    @asynccontextmanager
    async def practice_lock(self, session_id: str):
        if not PRACTICE_SESSION_ID_RE.match(session_id or ""):
            raise HTTPException(status_code=404, detail="Practice session not found.")
        entry = self._practice_locks.get(session_id)
        if entry is None:
            entry = [asyncio.Lock(), 0]
            self._practice_locks[session_id] = entry
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] <= 0 and self._practice_locks.get(session_id) is entry:
                self._practice_locks.pop(session_id, None)

    def is_live_session(self, target_session: InterviewSession, epoch: int) -> bool:
        return self.session_epoch == epoch and self.active_session is target_session

    @staticmethod
    def transcript_regroup_sig(session: InterviewSession) -> tuple[Any, ...]:
        return (
            session.config.interview_type,
            session.config.role_title,
            tuple((u.id, u.rev, u.speaker) for u in session.utterances),
            tuple(session.warmup_utterance_ids),
        )

    def current_phase_and_actions(self) -> tuple[str, list[str], dict[str, str]]:
        phase = derive_session_phase(
            self.active_session,
            is_capturing=self.is_capturing,
            calibration_job=self.calibration_job,
        )
        allowed, blocked = compute_allowed_session_events(
            self.active_session,
            phase=phase,
            regroup_job=self.regroup_job,
        )
        return phase.value, allowed, blocked

    def begin_capture(self, *, enforce_transition: bool = False, owner: Any | None = None) -> None:
        """
        Starts (or resumes) live capture. An explicit `start_capture` command passes
        `enforce_transition=True` so it is rejected while capture is already active or
        calibration is running; streamed audio frames call this idempotently without the check.
        """
        if enforce_transition:
            phase_str, allowed, blocked = self.current_phase_and_actions()
            if "start_capture" not in allowed:
                raise InvalidStateTransitionError(
                    machine="session_phase",
                    current_state=phase_str,
                    event="start_capture",
                    allowed_events=allowed,
                    remediation=blocked.get(
                        "start_capture",
                        "Live capture cannot start in the current session phase.",
                    ),
                )
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_started_epoch_ms = int(time.time() * 1000)
            # Clear stale ended timestamp when resuming a paused session
            self.active_session.ended_epoch_ms = None
        if owner is not None:
            self.capturing_sender = owner

    def pause_or_end_capture(self, *, mark_ended: bool = True) -> None:
        now_ms = int(time.time() * 1000)
        if self.is_capturing and self.capture_started_epoch_ms is not None:
            delta = max(0, now_ms - self.capture_started_epoch_ms)
            self.active_session.elapsed_capture_ms += delta
        self.is_capturing = False
        self.capture_started_epoch_ms = None
        self.capturing_sender = None
        self.close_archive()
        self.asr.clear_streams()
        if mark_ended:
            self.active_session.ended_epoch_ms = now_ms

    def live_elapsed_ms(self) -> int:
        base = self.active_session.elapsed_capture_ms
        if self.is_capturing and self.capture_started_epoch_ms is not None:
            base += max(0, int(time.time() * 1000) - self.capture_started_epoch_ms)
        return base

    def ensure_archive(self) -> SessionAudioArchive | None:
        cfg = self.active_session.config
        if cfg.save_raw_audio:
            if self.audio_archive is None or self.audio_archive.session_id != self.active_session.id:
                self.close_archive()
                self.audio_archive = SessionAudioArchive(
                    self.store.sessions_dir, self.active_session.id
                )
            return self.audio_archive
        return None

    def close_archive(self) -> None:
        if self.audio_archive is not None:
            self.audio_archive.close()
            self.audio_archive = None

    def switch_session(self, new_session: InterviewSession) -> None:
        self.pause_or_end_capture(mark_ended=False)
        self.session_epoch += 1
        self.regroup_job = AsyncJobState.IDLE
        self.calibration_job = AsyncJobState.IDLE
        self.warming_card_ids.clear()
        self.loading_card_ids.clear()
        self.card_probe_gen.clear()
        self.role_orientation_checked.discard(new_session.id)
        self.role_orientation_inflight.discard(new_session.id)
        self.active_session = new_session
        self.diarizer.reset()
        self.grouper.sync_card_kinds(self.active_session)
        initial_offset_ms = (
            max((u.end_ms for u in new_session.utterances), default=0) + 500
            if new_session.utterances
            else 0
        )
        self.audio_router = SessionAudioRouter(
            self.vad_runtime,
            mode=new_session.config.mode,
            initial_offset_ms=initial_offset_ms,
        )

    async def broadcast_all(self, payload: dict[str, Any]) -> None:
        for sender in list(self.ws_senders):
            try:
                await sender(payload)
            except Exception:
                pass

    def ensure_asr_watcher(self) -> None:
        if self.asr_watcher_task is not None and not self.asr_watcher_task.done():
            return

        async def _watch() -> None:
            while self.asr.state in (ASRModelState.LOADING, ASRModelState.SWITCHING):
                await asyncio.sleep(0.75)
                await self.broadcast_all({"type": "asr_status", "asr": self.asr.status_dict()})
            await self.broadcast_all({"type": "asr_status", "asr": self.asr.status_dict()})

        try:
            self.asr_watcher_task = asyncio.create_task(_watch())
        except RuntimeError:
            pass


def create_app(
    project_root: Path | None = None,
    preload_asr: bool = True,
    auth_token: str | None = None,
) -> FastAPI:
    root = project_root or resolve_project_root()
    runtime = AppRuntime(root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def _warm_tts_quietly() -> None:
            try:
                await asyncio.to_thread(runtime.tts.warm_up_installed)
            except Exception as exc:
                logger.debug("Background local TTS warmup skipped: %s", exc)

        tts_warmup_task: asyncio.Task[Any] | None = None
        if preload_asr:
            runtime.asr.start_background_load()
            runtime.voice_embedder.start_background_load()
            tts_warmup_task = asyncio.create_task(_warm_tts_quietly())
        yield
        if tts_warmup_task is not None and not tts_warmup_task.done():
            tts_warmup_task.cancel()
        if runtime.asr_watcher_task is not None and not runtime.asr_watcher_task.done():
            runtime.asr_watcher_task.cancel()
        runtime.close_archive()
        runtime.asr.clear_streams()
        runtime.store.close()
        await runtime.llm.aclose()

    app = FastAPI(title="Mockery — Local Mock Interview Studio", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.auth_token = auth_token

    @app.middleware("http")
    async def verify_http_origin_and_token(request: Request, call_next: Any):
        host = request.headers.get("host") or ""
        if not auth_token:
            # Defend loopback-only server against browser DNS rebinding attacks
            parsed_host = (urlparse(f"http://{host}").hostname or "").lower()
            if parsed_host not in ("localhost", "127.0.0.1", "::1", "testserver"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Untrusted Host header rejected in local loopback mode."},
                )
        if auth_token:
            query_token = request.query_params.get("token")
            cookie_token = request.cookies.get("mockery_token")
            auth_header = request.headers.get("authorization")
            if not _has_valid_token(
                auth_token,
                query_token=query_token,
                cookie_token=cookie_token,
                auth_header=auth_header,
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid Mockery LAN access token."},
                )
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if not is_trusted_origin(origin, host, allow_same_host_lan=bool(auth_token)):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Untrusted Origin rejected."},
                )
            cl = request.headers.get("content-length")
            cl_int = int(cl) if (cl and cl.isdigit()) else None
            if request.url.path.startswith("/api/profile/resume"):
                if cl_int is not None and cl_int > MAX_RESUME_UPLOAD_BYTES + 262_144:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Resume upload exceeds 10 MB limit."},
                    )
            elif request.url.path == "/api/asr/transcribe":
                if cl_int is not None and cl_int > MAX_ASR_AUDIO_BYTES + 65_536:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Audio upload exceeds 4 MB limit."},
                    )
            elif request.url.path in ("/api/tts/speak", "/api/tts/stream"):
                if cl_int is not None and cl_int > MAX_TTS_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "TTS request exceeds 32 KB limit."},
                    )
            elif request.url.path.startswith("/api/practice"):
                if cl_int is not None and cl_int > MAX_PRACTICE_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Practice request exceeds 256 KB limit."},
                    )
        response = await call_next(request)
        if request.url.path.startswith(("/api/practice", "/api/profile", "/api/asr/transcribe")):
            response.headers["Cache-Control"] = "no-store"
        if auth_token and request.query_params.get("token"):
            response.set_cookie(
                key="mockery_token",
                value=auth_token,
                httponly=True,
                samesite="strict",
            )
        return response

    @app.get("/api/machines")
    async def get_machine_specs() -> dict[str, Any]:
        return export_machine_specs()

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        active_jd = runtime.profiles.active_jd()
        resume = runtime.profiles.state.resume
        return {
            "asr": runtime.asr.status_dict(),
            "tts": runtime.tts.status_dict(),
            "diarizer": runtime.diarizer.status_dict(),
            "llm": runtime.llm_config.public_dict(),
            "addons": runtime.addons.status_dict(),
            "profile_summary": {
                "has_resume": bool(resume and resume.raw_text.strip()),
                "resume_filename": resume.filename if resume else "",
                "resume_redacted_count": resume.redacted_count if resume else 0,
                "resume_engine": resume.engine if resume else "rules_only",
                "active_jd": (
                    {
                        "id": active_jd.id,
                        "title": active_jd.title,
                        "company": active_jd.company,
                        "target_level": active_jd.target_level,
                    }
                    if active_jd
                    else None
                ),
                "jd_count": len(runtime.profiles.state.job_listings),
            },
            "data_dir": str(runtime.project_root),
            "sessions_dir": str(runtime.store.sessions_dir),
            "session": _serialize_session(runtime),
        }

    app.include_router(build_profile_router(runtime))
    app.include_router(build_addons_router(runtime, decode_audio_fn=_decode_audio_to_float32_16k))
    app.include_router(build_practice_router(runtime))

    @app.get("/api/providers/discover")
    async def discover_providers() -> dict[str, Any]:
        info = await discover_available_providers()
        info["data_dir"] = str(runtime.project_root)
        info["sessions_dir"] = str(runtime.store.sessions_dir)
        return info

    @app.post("/api/settings/llm/test")
    async def test_llm_settings(body: LLMConnectionTestRequest) -> dict[str, Any]:
        cfg = LLMRuntimeConfig(runtime.project_root)
        target_provider = (body.provider or runtime.llm_config.provider).strip()
        preset = LLM_PRESETS.get(target_provider, {})
        cfg.provider = target_provider
        cfg.base_url = (
            body.base_url.strip()
            if body.base_url is not None and body.base_url.strip()
            else (
                runtime.llm_config.base_url
                if target_provider == runtime.llm_config.provider
                else str(preset.get("base_url", ""))
            )
        )
        cfg.fast_model = (
            body.fast_model.strip()
            if body.fast_model is not None and body.fast_model.strip()
            else (
                runtime.llm_config.fast_model
                if target_provider == runtime.llm_config.provider
                else str(preset.get("fast_model", ""))
            )
        )
        cfg.calibration_model = (
            body.calibration_model.strip()
            if body.calibration_model is not None and body.calibration_model.strip()
            else (
                runtime.llm_config.calibration_model
                if target_provider == runtime.llm_config.provider
                else str(preset.get("calibration_model", ""))
            )
        )
        if body.api_key is not None and body.api_key.strip():
            cfg.api_key = body.api_key.strip()
        elif target_provider == runtime.llm_config.provider:
            cfg.api_key = runtime.llm_config.api_key
        else:
            env_var = PROVIDER_ENV_KEYS.get(target_provider)
            cfg.api_key = "" if not env_var else (os.getenv(env_var, "").strip())

        tester = InterviewCopilotLLM(cfg)
        try:
            return await tester.test_connection(timeout=20.0)
        finally:
            await tester.aclose()

    @app.post("/api/open-data-dir")
    async def open_data_dir() -> dict[str, Any]:
        if auth_token:
            raise HTTPException(status_code=403, detail="Opening local folders is disabled in LAN mode.")
        target = runtime.store.sessions_dir
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            await asyncio.create_subprocess_exec("open", str(target))
            return {"ok": True, "path": str(target)}
        return {"ok": False, "path": str(target), "detail": "Folder reveal is supported on macOS."}

    @app.post("/api/settings/llm")
    async def update_llm_settings(body: LLMSettingsUpdate) -> dict[str, Any]:
        runtime.llm_config.update_and_save(
            provider=body.provider,
            base_url=body.base_url,
            fast_model=body.fast_model,
            calibration_model=body.calibration_model,
            api_key=body.api_key,
            parakeet_model=body.parakeet_model,
        )
        if body.parakeet_model:
            runtime.asr.switch_model(body.parakeet_model)
            runtime.ensure_asr_watcher()
        await runtime.broadcast_all({"type": "asr_status", "asr": runtime.asr.status_dict()})
        return {
            "ok": True,
            "llm": runtime.llm_config.public_dict(),
            "asr": runtime.asr.status_dict(),
        }

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": runtime.store.list_recent()}

    @app.get("/api/sessions/{session_id}")
    async def get_saved_session(session_id: str) -> dict[str, Any]:
        phase_str, allowed, blocked = runtime.current_phase_and_actions()
        if "load_historical_session" not in allowed:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": blocked.get(
                        "load_historical_session",
                        "Stop live audio capture before switching to a historical session.",
                    ),
                    "state": phase_str,
                    "allowed_events": allowed,
                },
            )
        loaded = (
            runtime.active_session
            if runtime.active_session.id == session_id
            else runtime.store.load(session_id)
        )
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": _serialize_session_obj(runtime, loaded)}

    @app.post("/api/sessions/{session_id}/load")
    async def load_saved_session(session_id: str) -> dict[str, Any]:
        phase_str, allowed, blocked = runtime.current_phase_and_actions()
        if "load_historical_session" not in allowed:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": blocked.get(
                        "load_historical_session",
                        "Stop live audio capture before switching to a historical session.",
                    ),
                    "state": phase_str,
                    "allowed_events": allowed,
                },
            )
        loaded = runtime.store.load(session_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found")
        runtime.switch_session(loaded)
        serialized = _serialize_session(runtime)
        await runtime.broadcast_all(
            {
                "type": "session_state",
                "session": serialized,
                "asr": runtime.asr.status_dict(),
                "manual_speaker": runtime.audio_router.manual_speaker,
            }
        )
        return {"session": serialized}

    @app.get("/api/sessions/{session_id}/markdown")
    async def export_session_markdown(
        session_id: str,
        audience: Literal["interviewer", "candidate"] = Query("interviewer"),
    ) -> PlainTextResponse:
        session = (
            runtime.active_session
            if runtime.active_session.id == session_id
            else runtime.store.load(session_id)
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        suffix = "" if audience == "interviewer" else "_candidate"
        return PlainTextResponse(
            content=session.to_markdown(audience=audience),
            headers={"Content-Disposition": f'attachment; filename="{session.id}{suffix}.md"'},
        )

    @app.websocket("/ws/session")
    async def session_ws(ws: WebSocket) -> None:
        if auth_token:
            if not _has_valid_token(
                auth_token,
                query_token=ws.query_params.get("token"),
                cookie_token=ws.cookies.get("mockery_token"),
                auth_header=ws.headers.get("authorization"),
            ):
                await ws.close(code=1008)
                return
        if not is_trusted_origin(
            ws.headers.get("origin"),
            ws.headers.get("host"),
            allow_same_host_lan=bool(auth_token),
        ):
            await ws.close(code=1008)
            return

        await ws.accept()
        partial_inflight: dict[SpeakerRole, bool] = {"interviewer": False, "candidate": False}
        bg_tasks: set[asyncio.Task[Any]] = set()
        send_lock = asyncio.Lock()
        final_seg_queue: asyncio.Queue[
            tuple[InterviewSession, SessionAudioRouter, SegmentedSpeech] | None
        ] = asyncio.Queue()

        def spawn_bg(coro: Any) -> asyncio.Task[Any]:
            task = asyncio.create_task(coro)
            bg_tasks.add(task)
            task.add_done_callback(bg_tasks.discard)
            return task

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                try:
                    await ws.send_text(json.dumps(payload))
                except Exception:
                    pass

        runtime.ws_senders.add(send_json)

        async def broadcast_session(extra: dict[str, Any] | None = None) -> None:
            msg: dict[str, Any] = {
                "type": "session_state",
                "session": _serialize_session(runtime),
                "asr": runtime.asr.status_dict(),
                "manual_speaker": runtime.audio_router.manual_speaker,
            }
            if extra:
                msg.update(extra)
            await runtime.broadcast_all(msg)

        async def handle_segmented_speech(
            target_session: InterviewSession,
            router: SessionAudioRouter,
            seg: SegmentedSpeech,
        ) -> None:
            use_acoustic_diarizer = (
                target_session.config.mode == "in_person"
                and target_session.config.auto_diarize
            )

            if seg.kind == "partial":
                if (
                    partial_inflight[seg.speaker]
                    or runtime.asr.state != ASRModelState.READY
                    or not final_seg_queue.empty()
                ):
                    return
                partial_inflight[seg.speaker] = True
                try:
                    if use_acoustic_diarizer and runtime.voice_embedder.is_ready:
                        # Run CoreML WeSpeaker and MLX Parakeet concurrently on separate hardware units
                        speaker, res = await asyncio.gather(
                            asyncio.to_thread(
                                runtime.diarizer.classify_partial,
                                seg.audio,
                                default_speaker=seg.speaker,
                            ),
                            asyncio.to_thread(
                                runtime.asr.transcribe_pcm_partial,
                                seg.utterance_id,
                                seg.audio,
                            ),
                        )
                    else:
                        speaker = seg.speaker
                        res = await asyncio.to_thread(
                            runtime.asr.transcribe_pcm_partial,
                            seg.utterance_id,
                            seg.audio,
                        )
                    if res.text and runtime.active_session.id == target_session.id:
                        await upsert_utterance_and_route(
                            target_session=target_session,
                            router=router,
                            utt=Utterance(
                                id=seg.utterance_id,
                                rev=seg.rev,
                                final=False,
                                speaker=speaker,
                                text=res.text,
                                start_ms=seg.start_ms,
                                end_ms=seg.end_ms,
                            ),
                        )
                except Exception as exc:
                    logger.warning("Partial ASR transcription failed: %s", exc)
                finally:
                    partial_inflight[seg.speaker] = False
                return

            # Release incremental KV cache immediately when a speech turn finalizes;
            # finals always run exact full-clip `transcribe_pcm` (0.00% WER regression).
            runtime.asr.drop_stream(seg.utterance_id)

            if use_acoustic_diarizer:
                try:
                    sub_segs = await asyncio.to_thread(
                        runtime.diarizer.diarize_segment,
                        seg.audio,
                        seg.start_ms,
                        seg.end_ms,
                        default_speaker=seg.speaker,
                    )
                except Exception as exc:
                    logger.warning("Acoustic diarization failed (%s); falling back to manual speaker", exc)
                    sub_segs = []

                if sub_segs:
                    emitted_any = False
                    for sub in sub_segs:
                        res = await asyncio.to_thread(runtime.asr.transcribe_pcm, sub.audio)
                        if runtime.active_session.id != target_session.id:
                            return
                        if not res.text:
                            continue
                        utt_id = seg.utterance_id if not emitted_any else short_id("utt")
                        rev = seg.rev if not emitted_any else 1
                        emitted_any = True
                        runtime.diarizer.bind_utterance(utt_id, sub)
                        await upsert_utterance_and_route(
                            target_session=target_session,
                            router=router,
                            utt=Utterance(
                                id=utt_id,
                                rev=rev,
                                final=True,
                                speaker=sub.speaker,
                                text=res.text,
                                start_ms=sub.start_ms,
                                end_ms=sub.end_ms,
                            ),
                        )
                    if not emitted_any:
                        existing = target_session.utterance_map().get(seg.utterance_id)
                        if existing is not None and not existing.final:
                            target_session.remove_utterance(seg.utterance_id)
                            runtime.store.save_live(target_session)
                            await broadcast_session()
                    return

            # Remote Meet hardware-separated channels (or fallback when auto_diarize is off)
            res = await asyncio.to_thread(runtime.asr.transcribe_pcm, seg.audio)
            if runtime.active_session.id != target_session.id:
                return
            if not res.text:
                # If an earlier non-empty draft existed for this utterance, remove it so it doesn't stay stuck as live draft
                existing = target_session.utterance_map().get(seg.utterance_id)
                if existing is not None and not existing.final:
                    target_session.remove_utterance(seg.utterance_id)
                    runtime.store.save_live(target_session)
                    await broadcast_session()
                return
            await upsert_utterance_and_route(
                target_session=target_session,
                router=router,
                utt=Utterance(
                    id=seg.utterance_id,
                    rev=seg.rev,
                    final=True,
                    speaker=seg.speaker,
                    text=res.text,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                ),
            )

        async def upsert_utterance_and_route(
            *,
            target_session: InterviewSession,
            router: SessionAudioRouter,
            utt: Utterance,
        ) -> None:
            # Drop stale out-of-order revisions BEFORE bleed filtering, persistence, or grouping side effects
            existing = target_session.utterance_map().get(utt.id)
            if existing is not None and utt.rev < existing.rev:
                return

            if utt.speaker == "candidate":
                router.bleed_filter.record_candidate_text(utt.start_ms, utt.end_ms, utt.text)
            elif (
                target_session.config.mode == "remote_meet"
                and router.bleed_filter.is_echo_bleed(utt.start_ms, utt.end_ms, utt.text)
            ):
                had_draft = target_session.remove_utterance(utt.id)
                target_session.echo_bleed_dropped += 1
                if had_draft:
                    runtime.store.save_live(target_session)
                if runtime.active_session.id == target_session.id:
                    await broadcast_session()
                return

            decision = runtime.grouper.upsert_and_route(target_session, utt)
            if decision.ignored_stale:
                return

            is_inplace_draft = (
                not utt.final
                and existing is not None
                and not decision.opened_new_qa
            )

            # Always log new utterances, finals, and every 4th intermediate draft (~5s crash recovery)
            if utt.final or existing is None or (utt.rev % 4 == 0):
                runtime.store.append_utterance_event(target_session.id, utt)
            if utt.final:
                runtime.store.save_live(target_session)

            if runtime.active_session is target_session:
                if is_inplace_draft:
                    draft_msg: dict[str, Any] = {
                        "type": "utterance_draft",
                        "session_id": target_session.id,
                        "qa_id": decision.qa_id,
                        "utterance": utt.model_dump(),
                        "metrics": target_session.compute_metrics().model_dump(),
                    }
                    if decision.qa_id:
                        owner_qa = next(
                            (q for q in target_session.qa_groups if q.id == decision.qa_id), None
                        )
                        if owner_qa is not None:
                            p_state = derive_card_probe_state(
                                target_session,
                                owner_qa,
                                warming_ids=runtime.warming_card_ids,
                                loading_ids=runtime.loading_card_ids,
                            )
                            draft_msg["probe_state"] = p_state.value
                            draft_msg["has_speculative_cache"] = (
                                p_state == CardProbeState.READY_CACHED
                            )
                    await runtime.broadcast_all(draft_msg)
                else:
                    await broadcast_session(
                        {
                            "latest_qa_id": decision.qa_id,
                            "opened_new_qa": decision.opened_new_qa,
                        }
                    )

            job_epoch = runtime.session_epoch
            # LLM handles topic-shift card pivots (when needed) — never live speaker classification
            if (
                utt.final
                and decision.needs_async_topic_check
                and decision.qa_id
                and runtime.llm_config.is_configured
            ):
                spawn_bg(
                    run_turn_triage(
                        target_session=target_session,
                        utterance_id=utt.id,
                        diarize=False,
                        job_epoch=job_epoch,
                    )
                )

            if decision.should_speculate_probes and decision.qa_id:
                spawn_bg(precompute_speculative_probes(target_session, decision.qa_id, job_epoch))

            # One-shot single-mic speaker role-orientation guard: catches sessions where the
            # candidate spoke first at t=00:00 and inverted both acoustic voice clusters globally.
            if (
                utt.final
                and target_session.config.mode == "in_person"
                and target_session.config.auto_diarize
                and runtime.llm_config.is_configured
                and target_session.id not in runtime.role_orientation_checked
                and target_session.id not in runtime.role_orientation_inflight
                and not runtime.diarizer.has_manual_override
            ):
                finals = [u for u in target_session.utterances if u.final and u.text.strip()]
                n_int = sum(1 for u in finals if u.speaker == "interviewer")
                n_cand = sum(1 for u in finals if u.speaker == "candidate")
                total_words = sum(len(u.text.split()) for u in finals)
                if len(finals) >= 8 and n_int >= 2 and n_cand >= 2 and total_words >= 60:
                    spawn_bg(run_speaker_role_orientation_check(target_session, job_epoch))

        async def run_speaker_role_orientation_check(
            target_session: InterviewSession, job_epoch: int
        ) -> None:
            sid = target_session.id
            if sid in runtime.role_orientation_checked or sid in runtime.role_orientation_inflight:
                return
            runtime.role_orientation_inflight.add(sid)
            try:
                res = await runtime.llm.check_speaker_role_orientation(target_session)
                if not runtime.is_live_session(target_session, job_epoch):
                    return
                runtime.role_orientation_checked.add(sid)
                if (
                    res.get("inverted") is True
                    and res.get("confidence") == "high"
                    and not runtime.diarizer.has_manual_override
                ):
                    runtime.diarizer.swap_roles(mark_manual=False)
                    for u in target_session.utterances:
                        u.speaker = "candidate" if u.speaker == "interviewer" else "interviewer"
                    runtime.store.save(target_session)
                    logger.info(
                        "Auto-swapped inverted speaker roles for session %s (%s)",
                        sid,
                        res.get("rationale", ""),
                    )
                    await broadcast_session(
                        {
                            "roles_auto_swapped": True,
                            "roles_auto_swapped_rationale": res.get("rationale", ""),
                        }
                    )
            except Exception as exc:
                logger.warning("Speaker role-orientation background task failed: %s", exc)
            finally:
                runtime.role_orientation_inflight.discard(sid)

        async def precompute_speculative_probes(
            target_session: InterviewSession, qa_id: str, job_epoch: int
        ) -> None:
            if (
                not runtime.is_live_session(target_session, job_epoch)
                or qa_id in runtime.warming_card_ids
                or qa_id in runtime.loading_card_ids
                or not runtime.llm_config.is_configured
            ):
                return
            qa = next((q for q in target_session.qa_groups if q.id == qa_id), None)
            if qa is None or not qa.utterance_ids:
                return
            start_gen = runtime.card_probe_gen.get(qa_id, 0)
            start_hist_len = len(qa.dive_deeper_history)
            start_rev_sum = target_session.card_revision_sum(qa)
            runtime.warming_card_ids.add(qa_id)
            if runtime.is_live_session(target_session, job_epoch):
                await broadcast_session()
            should_rewarm = False
            try:
                cached = await runtime.llm.suggest_probes(target_session, qa)
                if (
                    runtime.is_live_session(target_session, job_epoch)
                    and qa_id not in runtime.loading_card_ids
                    and runtime.card_probe_gen.get(qa_id, 0) == start_gen
                    and len(qa.dive_deeper_history) == start_hist_len
                ):
                    qa.speculative_cache = cached
                    if target_session.card_revision_sum(qa) > start_rev_sum:
                        should_rewarm = True
            except Exception as exc:
                logger.warning("Speculative probe warming failed: %s", exc)
            finally:
                runtime.warming_card_ids.discard(qa_id)
                if runtime.is_live_session(target_session, job_epoch):
                    await broadcast_session()
            if should_rewarm and runtime.is_live_session(target_session, job_epoch):
                spawn_bg(precompute_speculative_probes(target_session, qa_id, job_epoch))

        triage_lock = asyncio.Lock()

        async def run_turn_triage(
            *,
            target_session: InterviewSession,
            utterance_id: str,
            diarize: bool,
            job_epoch: int,
        ) -> None:
            async with triage_lock:
                if not runtime.is_live_session(target_session, job_epoch):
                    return
                utt_obj = target_session.utterance_map().get(utterance_id)
                if utt_obj is None or not utt_obj.final:
                    return
                plan = await runtime.llm.triage_turn(target_session, utt_obj, diarize=diarize)
                if plan is None or not runtime.is_live_session(target_session, job_epoch):
                    return
                apply_res = runtime.grouper.apply_turn_triage(
                    target_session,
                    plan,
                    allow_speaker_mutation=diarize,
                    allow_card_split=True,
                )
                if not apply_res.mutated and not apply_res.queued_review_id:
                    return
                umap = target_session.utterance_map()
                if apply_res.split_utterance_ids:
                    for uid in apply_res.split_utterance_ids:
                        if uid in umap:
                            runtime.store.append_utterance_event(target_session.id, umap[uid])
                elif apply_res.mutated and utterance_id in umap:
                    runtime.store.append_utterance_event(target_session.id, umap[utterance_id])
                runtime.store.save(target_session)
                extra: dict[str, Any] = {}
                if apply_res.opened_new_qa_id:
                    extra["latest_qa_id"] = apply_res.opened_new_qa_id
                    extra["opened_new_qa"] = True
                await broadcast_session(extra or None)

        async def final_segment_worker() -> None:
            while True:
                item = await final_seg_queue.get()
                try:
                    if item is None:
                        return
                    sess_ref, router_ref, seg_item = item
                    await handle_segmented_speech(sess_ref, router_ref, seg_item)
                except Exception as exc:
                    logger.exception("Final segment transcription failed")
                    await send_json(
                        {"type": "error", "message": f"Transcription error (skipped segment): {exc}"}
                    )
                finally:
                    final_seg_queue.task_done()

        seg_worker_task = asyncio.create_task(final_segment_worker())

        await send_json(
            {
                "type": "session_state",
                "session": _serialize_session(runtime),
                "asr": runtime.asr.status_dict(),
                "manual_speaker": runtime.audio_router.manual_speaker,
            }
        )
        await send_json({"type": "asr_status", "asr": runtime.asr.status_dict()})
        if runtime.asr.state in (ASRModelState.LOADING, ASRModelState.SWITCHING):
            runtime.ensure_asr_watcher()

        capture_gate = {"closed": False}

        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"] is not None:
                    # Drop late in-flight worklet packets arriving after flush_and_end/cancel_capture
                    # or from a non-owner socket while another client owns active capture.
                    if capture_gate["closed"]:
                        continue
                    if (
                        runtime.is_capturing
                        and runtime.capturing_sender is not None
                        and runtime.capturing_sender is not send_json
                    ):
                        continue
                    runtime.begin_capture(owner=send_json)
                    raw_bytes = message["bytes"]
                    arch = runtime.ensure_archive()
                    if arch is not None:
                        arch.append_frame(raw_bytes)
                    cur_session = runtime.active_session
                    cur_router = runtime.audio_router
                    segments = cur_router.ingest_binary_frame(raw_bytes)
                    for seg in segments:
                        if seg.kind == "partial":
                            spawn_bg(handle_segmented_speech(cur_session, cur_router, seg))
                        else:
                            await final_seg_queue.put((cur_session, cur_router, seg))
                    continue

                if "text" in message and message["text"]:
                    try:
                        cmd = json.loads(message["text"])
                        if not isinstance(cmd, dict):
                            raise ValueError("WebSocket command must be a JSON object.")
                        await _handle_ws_command(
                            cmd=cmd,
                            runtime=runtime,
                            spawn_bg=spawn_bg,
                            final_seg_queue=final_seg_queue,
                            send_json=send_json,
                            broadcast_session=broadcast_session,
                            upsert_utterance_and_route=upsert_utterance_and_route,
                            capture_gate=capture_gate,
                        )
                    except InvalidStateTransitionError as exc:
                        await send_json(exc.to_dict())
                    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                        logger.warning("Rejected invalid WS command: %s", exc)
                        await send_json({"type": "error", "message": f"Invalid command: {exc}"})
        except WebSocketDisconnect:
            pass
        finally:
            runtime.ws_senders.discard(send_json)
            if runtime.capturing_sender is send_json or not runtime.ws_senders:
                runtime.pause_or_end_capture(mark_ended=False)
            await final_seg_queue.put(None)
            await seg_worker_task

    @app.get("/static/js_sandbox_worker.js")
    async def serve_js_sandbox_worker() -> FileResponse:
        worker_path = STATIC_DIR / "js_sandbox_worker.js"
        if not worker_path.is_file():
            raise HTTPException(status_code=404, detail="JS sandbox worker not found.")
        return FileResponse(
            path=worker_path,
            media_type="application/javascript; charset=utf-8",
            headers={
                "Content-Security-Policy": (
                    "default-src 'none'; "
                    "script-src 'unsafe-eval'; "
                    "connect-src 'none'; "
                    "worker-src 'none'; "
                    "child-src 'none';"
                ),
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/static/py_sandbox_worker.js")
    async def serve_py_sandbox_worker(request: Request) -> FileResponse:
        worker_path = STATIC_DIR / "py_sandbox_worker.js"
        if not worker_path.is_file():
            raise HTTPException(status_code=404, detail="Python sandbox worker not found.")
        scheme = "https" if request.url.scheme == "https" else "http"
        netloc = re.sub(r"[^A-Za-z0-9._:\[\]-]", "", request.url.netloc or "")
        pyodide_prefix = f"{scheme}://{netloc}/api/addons/pyodide/" if netloc else "'none'"
        return FileResponse(
            path=worker_path,
            media_type="application/javascript; charset=utf-8",
            headers={
                "Content-Security-Policy": (
                    "default-src 'none'; "
                    f"script-src {pyodide_prefix} 'unsafe-eval' 'wasm-unsafe-eval'; "
                    f"connect-src {pyodide_prefix}; "
                    "worker-src 'none'; "
                    "child-src 'none';"
                ),
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def root_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _require_allowed_event(runtime: AppRuntime, event: str) -> None:
    phase_str, allowed, blocked = runtime.current_phase_and_actions()
    if event not in allowed:
        raise InvalidStateTransitionError(
            machine="session_phase",
            current_state=phase_str,
            event=event,
            allowed_events=allowed,
            remediation=blocked.get(event, f"Event '{event}' is not permitted in state '{phase_str}'."),
        )


async def _handle_ws_command(
    *,
    cmd: dict[str, Any],
    runtime: AppRuntime,
    spawn_bg: Any,
    final_seg_queue: asyncio.Queue[Any],
    send_json: Any,
    broadcast_session: Any,
    upsert_utterance_and_route: Any,
    capture_gate: dict[str, bool] | None = None,
) -> None:
    action = str(cmd.get("type", ""))

    if action == "new_session":
        cfg_data = dict(cmd.get("config") or {})
        base_cfg = runtime.active_session.config.model_dump()
        cfg = SessionConfig.model_validate({**base_cfg, **cfg_data})
        runtime.switch_session(InterviewSession(config=cfg))
        if capture_gate is not None:
            capture_gate["closed"] = False
        await broadcast_session()
        return

    if action == "configure_session":
        cfg_data = cmd.get("config") or {}
        cfg = SessionConfig.model_validate({**runtime.active_session.config.model_dump(), **cfg_data})
        runtime.active_session.config = cfg
        runtime.grouper.sync_card_kinds(runtime.active_session)
        runtime.audio_router.mode = cfg.mode
        if not cfg.save_raw_audio:
            runtime.close_archive()
        if runtime.active_session.utterances:
            runtime.store.save(runtime.active_session)
        await broadcast_session()
        return

    if action == "start_capture":
        runtime.begin_capture(enforce_transition=True, owner=send_json)
        if capture_gate is not None:
            capture_gate["closed"] = False
        await broadcast_session()
        return

    if action == "cancel_capture":
        if capture_gate is not None:
            capture_gate["closed"] = True
        runtime.pause_or_end_capture(mark_ended=False)
        await broadcast_session()
        return

    if action == "set_manual_speaker":
        speaker = cmd.get("speaker")
        if speaker in ("interviewer", "candidate"):
            cur_session = runtime.active_session
            cur_router = runtime.audio_router
            flushed = cur_router.set_manual_speaker(speaker)
            for seg in flushed:
                await final_seg_queue.put((cur_session, cur_router, seg))
            await runtime.broadcast_all(
                {"type": "manual_speaker", "manual_speaker": cur_router.manual_speaker}
            )
        return

    if action == "inject_utterance":
        speaker = cmd.get("speaker", "interviewer")
        text = str(cmd.get("text", "")).strip()
        start_ms = int(cmd.get("start_ms", 0))
        end_ms = int(cmd.get("end_ms", start_ms + 2500))
        utt_id = cmd.get("id")
        rev = int(cmd.get("rev", 1))
        final = bool(cmd.get("final", True))
        if speaker in ("interviewer", "candidate") and text:
            utt = (
                Utterance(
                    id=str(utt_id),
                    rev=rev,
                    final=final,
                    speaker=speaker,
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
                if utt_id
                else Utterance(
                    rev=rev,
                    final=final,
                    speaker=speaker,
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )
            await upsert_utterance_and_route(
                target_session=runtime.active_session,
                router=runtime.audio_router,
                utt=utt,
            )
        return

    if action == "force_new_card":
        _require_allowed_event(runtime, "force_new_card")
        qa = runtime.grouper.force_new_card(runtime.active_session, title=str(cmd.get("title", "")))
        runtime.store.save(runtime.active_session)
        await broadcast_session({"latest_qa_id": qa.id, "opened_new_qa": True})
        return

    if action == "split_card":
        _require_allowed_event(runtime, "split_card")
        new_qa = runtime.grouper.split_utterance_to_new_card(
            runtime.active_session,
            qa_id=str(cmd.get("qa_id", "")),
            utterance_id=str(cmd.get("utterance_id", "")),
        )
        if new_qa:
            runtime.store.save(runtime.active_session)
            await broadcast_session({"latest_qa_id": new_qa.id, "opened_new_qa": True})
        return

    if action == "merge_card":
        _require_allowed_event(runtime, "merge_card")
        merged = runtime.grouper.merge_with_previous(
            runtime.active_session,
            qa_id=str(cmd.get("qa_id", "")),
        )
        if merged:
            runtime.store.save(runtime.active_session)
            await broadcast_session({"latest_qa_id": merged.id})
        return

    if action == "update_qa_meta":
        qa_id = str(cmd.get("qa_id", ""))
        qa = next((q for q in runtime.active_session.qa_groups if q.id == qa_id), None)
        if qa is not None:
            if "title" in cmd and str(cmd["title"]).strip():
                qa.title = str(cmd["title"]).strip()
            if "notes" in cmd:
                qa.notes = str(cmd["notes"])
            runtime.store.save(runtime.active_session)
            await broadcast_session()
        return

    if action == "flip_utterance_speaker":
        utt_id = str(cmd.get("utterance_id", ""))
        updated = runtime.grouper.flip_utterance_speaker(runtime.active_session, utt_id)
        if updated is not None:
            runtime.diarizer.record_manual_correction(utt_id, updated.speaker)
            runtime.store.append_utterance_event(runtime.active_session.id, updated)
            runtime.store.save(runtime.active_session)
            await broadcast_session()
        return

    if action == "swap_all_speakers":
        target_session = runtime.active_session
        runtime.diarizer.swap_roles(mark_manual=True)
        runtime.role_orientation_checked.add(target_session.id)
        for u in target_session.utterances:
            u.speaker = "candidate" if u.speaker == "interviewer" else "interviewer"
        runtime.store.save(target_session)
        await broadcast_session({"roles_swapped": True})
        return

    if action == "resolve_review":
        review_id = str(cmd.get("review_id", ""))
        accept = bool(cmd.get("accept", False))
        target_session = runtime.active_session
        review = next((r for r in target_session.pending_reviews if r.id == review_id), None)
        apply_res = runtime.grouper.resolve_pending_review(target_session, review_id, accept=accept)
        if apply_res.mutated:
            umap = target_session.utterance_map()
            if apply_res.split_utterance_ids:
                for uid in apply_res.split_utterance_ids:
                    if uid in umap:
                        runtime.store.append_utterance_event(target_session.id, umap[uid])
            elif accept and review is not None and review.utterance_id in umap:
                runtime.store.append_utterance_event(target_session.id, umap[review.utterance_id])
            runtime.store.save(target_session)
            extra: dict[str, Any] = {}
            if apply_res.opened_new_qa_id:
                extra["latest_qa_id"] = apply_res.opened_new_qa_id
                extra["opened_new_qa"] = True
            await broadcast_session(extra or None)
        return

    if action == "dive_deeper":
        _require_allowed_event(runtime, "dive_deeper")
        target_session = runtime.active_session
        qa_id = str(cmd.get("qa_id", ""))
        force_refresh = bool(cmd.get("force_refresh", False))
        qa = (
            next((q for q in target_session.qa_groups if q.id == qa_id), None)
            if qa_id
            else (target_session.qa_groups[-1] if target_session.qa_groups else None)
        )
        if qa is None:
            await send_json({"type": "error", "message": "No Q&A card selected yet."})
            return

        if not qa.utterance_ids:
            phase_str, allowed, _blocked = runtime.current_phase_and_actions()
            raise InvalidStateTransitionError(
                machine="card_probe",
                current_state=CardProbeState.EMPTY.value,
                event="dive_deeper",
                allowed_events=allowed,
                remediation="Wait for at least one utterance in this card before requesting Dive Deeper.",
            )

        p_state = derive_card_probe_state(
            target_session,
            qa,
            warming_ids=runtime.warming_card_ids,
            loading_ids=runtime.loading_card_ids,
        )
        if not force_refresh and p_state == CardProbeState.READY_CACHED and qa.speculative_cache is not None:
            result = qa.speculative_cache
            qa.speculative_cache = None
            runtime.card_probe_gen[qa.id] = runtime.card_probe_gen.get(qa.id, 0) + 1
            qa.dive_deeper_history.append(result)
            runtime.store.save(target_session)
            await runtime.broadcast_all(
                {"type": "dive_deeper_result", "result": result.model_dump(), "cached": True}
            )
            await broadcast_session()
            return

        if qa.id in runtime.loading_card_ids:
            return

        job_epoch = runtime.session_epoch
        runtime.card_probe_gen[qa.id] = runtime.card_probe_gen.get(qa.id, 0) + 1
        runtime.loading_card_ids.add(qa.id)
        await runtime.broadcast_all({"type": "dive_deeper_loading", "qa_id": qa.id})

        async def _run_dive_deeper() -> None:
            try:
                res = await runtime.llm.suggest_probes(target_session, qa)
                if not runtime.is_live_session(target_session, job_epoch):
                    return
                qa.speculative_cache = None
                qa.dive_deeper_history.append(res)
                runtime.store.save(target_session)
                await runtime.broadcast_all(
                    {"type": "dive_deeper_result", "result": res.model_dump(), "cached": False}
                )
            finally:
                runtime.loading_card_ids.discard(qa.id)
                if runtime.is_live_session(target_session, job_epoch):
                    await broadcast_session()

        spawn_bg(_run_dive_deeper())
        return

    if action == "submit_self_rating":
        rating_data = cmd.get("rating") or {}
        runtime.active_session.self_rating = InterviewerSelfRating.model_validate(rating_data)
        runtime.store.save(runtime.active_session)
        await broadcast_session()
        return

    if action == "regroup_cards":
        _require_allowed_event(runtime, "regroup_cards")
        target_session = runtime.active_session
        job_epoch = runtime.session_epoch
        runtime.regroup_job = AsyncJobState.RUNNING
        await runtime.broadcast_all({"type": "regroup_loading"})

        async def _run_regroup() -> None:
            try:
                sig_before = runtime.transcript_regroup_sig(target_session)
                boundaries = await runtime.llm.regroup_cards_full_context(target_session)
                if not runtime.is_live_session(target_session, job_epoch):
                    return
                applied = False
                if boundaries:
                    applied = runtime.grouper.apply_regroup_boundaries(target_session, boundaries)
                    if applied:
                        runtime.regrouped_signature[target_session.id] = sig_before
                        runtime.store.save(target_session)
                runtime.regroup_job = AsyncJobState.IDLE
                await broadcast_session({"regrouped": applied})
            except Exception:
                if runtime.is_live_session(target_session, job_epoch):
                    runtime.regroup_job = AsyncJobState.ERROR
                    await broadcast_session({"regrouped": False})

        spawn_bg(_run_regroup())
        return

    if action == "flush_and_end":
        if capture_gate is not None:
            capture_gate["closed"] = True
        cur_session = runtime.active_session
        cur_router = runtime.audio_router
        trailing = cur_router.flush_all()
        for seg in trailing:
            await final_seg_queue.put((cur_session, cur_router, seg))
        await final_seg_queue.join()
        runtime.pause_or_end_capture(mark_ended=True)
        runtime.store.save(cur_session)
        await broadcast_session({"session_ended": True})
        return

    if action == "generate_calibration":
        _require_allowed_event(runtime, "generate_calibration")
        target_session = runtime.active_session
        job_epoch = runtime.session_epoch
        if cmd.get("rating"):
            target_session.self_rating = InterviewerSelfRating.model_validate(cmd["rating"])
        regroup_first = bool(cmd.get("regroup_first", True))
        runtime.calibration_job = AsyncJobState.RUNNING
        await runtime.broadcast_all({"type": "calibration_loading"})

        async def _run_calibration() -> None:
            try:
                cur_sig = runtime.transcript_regroup_sig(target_session)
                needs_regroup = (
                    regroup_first
                    and runtime.llm_config.is_configured
                    and runtime.regrouped_signature.get(target_session.id) != cur_sig
                )
                if needs_regroup:
                    if runtime.is_live_session(target_session, job_epoch):
                        runtime.regroup_job = AsyncJobState.RUNNING
                    try:
                        boundaries = await runtime.llm.regroup_cards_full_context(target_session)
                        if not runtime.is_live_session(target_session, job_epoch):
                            return
                        if boundaries:
                            applied = runtime.grouper.apply_regroup_boundaries(
                                target_session, boundaries
                            )
                            if applied:
                                runtime.regrouped_signature[target_session.id] = cur_sig
                    finally:
                        if runtime.is_live_session(target_session, job_epoch):
                            runtime.regroup_job = AsyncJobState.IDLE

                delta_buf: list[str] = []
                last_flush = 0.0

                async def _on_calib_delta(chunk: str) -> None:
                    nonlocal last_flush
                    if not chunk or not runtime.is_live_session(target_session, job_epoch):
                        return
                    delta_buf.append(chunk)
                    now = time.monotonic()
                    if last_flush == 0.0 or (now - last_flush) >= 0.12 or sum(len(c) for c in delta_buf) >= 240:
                        text = "".join(delta_buf)
                        delta_buf.clear()
                        last_flush = now
                        await runtime.broadcast_all(
                            {
                                "type": "calibration_delta",
                                "session_id": target_session.id,
                                "delta": text,
                            }
                        )

                report_md = await runtime.llm.generate_calibration_report(
                    target_session,
                    on_delta=_on_calib_delta,
                )
                if not runtime.is_live_session(target_session, job_epoch):
                    return
                if delta_buf:
                    await runtime.broadcast_all(
                        {
                            "type": "calibration_delta",
                            "session_id": target_session.id,
                            "delta": "".join(delta_buf),
                        }
                    )
                    delta_buf.clear()
                target_session.calibration_markdown = report_md
                runtime.store.save(target_session)
                runtime.calibration_job = AsyncJobState.IDLE
                await runtime.broadcast_all({"type": "calibration_ready", "markdown": report_md})
                await broadcast_session()
            except Exception:
                if runtime.is_live_session(target_session, job_epoch):
                    runtime.calibration_job = AsyncJobState.ERROR
                    await broadcast_session()

        spawn_bg(_run_calibration())
        return


def _serialize_session(runtime: AppRuntime) -> dict[str, Any]:
    return _serialize_session_obj(runtime, runtime.active_session)


def _serialize_session_obj(runtime: AppRuntime, session: InterviewSession) -> dict[str, Any]:
    is_active = session.id == runtime.active_session.id
    data = session.model_dump()
    data["live_elapsed_ms"] = runtime.live_elapsed_ms() if is_active else session.elapsed_capture_ms
    data["metrics"] = session.compute_metrics().model_dump()

    # Reuse already-serialized utterance & card dicts instead of calling `.model_dump()` twice
    utt_dicts_by_id = {u["id"]: u for u in data["utterances"]}
    data["warmup_resolved"] = [
        utt_dicts_by_id[uid] for uid in session.warmup_utterance_ids if uid in utt_dicts_by_id
    ]

    warming_ids = runtime.warming_card_ids if is_active else set()
    loading_ids = runtime.loading_card_ids if is_active else set()
    card_probe_states: dict[str, str] = {}
    resolved_cards = []
    for qa, qa_dict in zip(session.qa_groups, data["qa_groups"]):
        p_state = derive_card_probe_state(
            session,
            qa,
            warming_ids=warming_ids,
            loading_ids=loading_ids,
        )
        card_probe_states[qa.id] = p_state.value
        qd = dict(qa_dict)
        qd["badge"] = f"{qa.badge_prefix()}{qa.index}"
        qd["utterances"] = [
            utt_dicts_by_id[uid] for uid in qa.utterance_ids if uid in utt_dicts_by_id
        ]
        qd["probe_state"] = p_state.value
        # Honest instant-cache flag: ONLY True when cache matches current utterance count & revisions!
        qd["has_speculative_cache"] = p_state == CardProbeState.READY_CACHED
        resolved_cards.append(qd)

    data["qa_groups_resolved"] = resolved_cards

    if is_active:
        phase_str, allowed_events, blocked_reasons = runtime.current_phase_and_actions()
        regroup_state = runtime.regroup_job.value
        calib_state = runtime.calibration_job.value
    else:
        phase = derive_session_phase(session, is_capturing=False, calibration_job=AsyncJobState.IDLE)
        allowed_events, blocked_reasons = compute_allowed_session_events(
            session, phase=phase, regroup_job=AsyncJobState.IDLE
        )
        phase_str = phase.value
        regroup_state = AsyncJobState.IDLE.value
        calib_state = AsyncJobState.IDLE.value

    violations = validate_session_invariants(session)
    data["state_machines"] = {
        "session": {
            "phase": phase_str,
            "allowed_events": allowed_events,
            "blocked_reasons": blocked_reasons,
        },
        "jobs": {
            "regroup": regroup_state,
            "calibration": calib_state,
        },
        "cards": card_probe_states,
        "invariants_ok": len(violations) == 0,
        "invariant_violations": violations,
    }
    return data
