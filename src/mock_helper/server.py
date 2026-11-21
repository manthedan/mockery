from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .asr import ParakeetEngine
from .diarizer import OnlineTwoSpeakerDiarizer, WeSpeakerEmbedder
from .grouper import QAGrouper
from .llm import InterviewCopilotLLM, LLMRuntimeConfig
from .models import (
    InterviewerSelfRating,
    InterviewSession,
    SessionConfig,
    SpeakerRole,
    Utterance,
    short_id,
)
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
from .vad import SileroVADRuntime

logger = logging.getLogger("mock_helper")

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]", "testserver"}


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    return PACKAGE_DIR.parent.parent


def is_trusted_origin(origin: str | None, host_header: str | None) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https"):
        return False
    origin_host = (parsed.hostname or "").lower()
    if origin_host not in ALLOWED_LOOPBACK_HOSTS:
        return False
    if host_header:
        expected_netloc = host_header.strip().lower()
        if parsed.netloc.lower() != expected_netloc:
            return False
    return True


class LLMSettingsUpdate(BaseModel):
    provider: str
    base_url: str
    fast_model: str
    calibration_model: str
    api_key: str | None = None
    parakeet_model: str | None = None


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
        self.warming_card_ids: set[str] = set()
        self.loading_card_ids: set[str] = set()
        self.card_probe_gen: dict[str, int] = {}
        self.role_orientation_checked: set[str] = set()
        self.role_orientation_inflight: set[str] = set()

        # Connected WebSocket senders & active capture owner
        self.ws_senders: set[Any] = set()
        self.capturing_sender: Any | None = None
        self.asr_watcher_task: asyncio.Task[Any] | None = None

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

    def begin_capture(self, *, require_consent: bool = False, owner: Any | None = None) -> None:
        phase_str, allowed, blocked = self.current_phase_and_actions()
        if require_consent and "start_capture" not in allowed:
            raise InvalidStateTransitionError(
                machine="session_phase",
                current_state=phase_str,
                event="start_capture",
                allowed_events=allowed,
                remediation=blocked.get(
                    "start_capture",
                    "Set `consent_confirmed=true` via `configure_session` first.",
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
        if mark_ended:
            self.active_session.ended_epoch_ms = now_ms

    def live_elapsed_ms(self) -> int:
        base = self.active_session.elapsed_capture_ms
        if self.is_capturing and self.capture_started_epoch_ms is not None:
            base += max(0, int(time.time() * 1000) - self.capture_started_epoch_ms)
        return base

    def ensure_archive(self) -> SessionAudioArchive | None:
        cfg = self.active_session.config
        if cfg.consent_confirmed and cfg.save_raw_audio:
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


def create_app(project_root: Path | None = None, preload_asr: bool = True) -> FastAPI:
    root = project_root or resolve_project_root()
    runtime = AppRuntime(root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if preload_asr:
            runtime.asr.start_background_load()
            runtime.voice_embedder.start_background_load()
        yield
        if runtime.asr_watcher_task is not None and not runtime.asr_watcher_task.done():
            runtime.asr_watcher_task.cancel()
        runtime.close_archive()
        await runtime.llm.aclose()

    app = FastAPI(title="Mock Helper — Live Interview Copilot", lifespan=lifespan)
    app.state.runtime = runtime

    @app.middleware("http")
    async def verify_http_origin(request: Request, call_next: Any):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            host = request.headers.get("host")
            if not is_trusted_origin(origin, host):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Untrusted Origin rejected."},
                )
        return await call_next(request)

    @app.get("/api/machines")
    async def get_machine_specs() -> dict[str, Any]:
        return export_machine_specs()

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        return {
            "asr": runtime.asr.status_dict(),
            "diarizer": runtime.diarizer.status_dict(),
            "llm": runtime.llm_config.public_dict(),
            "session": _serialize_session(runtime),
        }

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
        if not is_trusted_origin(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(code=1008)
            return

        await ws.accept()
        partial_inflight: dict[SpeakerRole, bool] = {"interviewer": False, "candidate": False}
        bg_tasks: set[asyncio.Task[Any]] = set()
        send_lock = asyncio.Lock()

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
                if partial_inflight[seg.speaker] or runtime.asr.state != ASRModelState.READY:
                    return
                partial_inflight[seg.speaker] = True
                try:
                    speaker = seg.speaker
                    if use_acoustic_diarizer and runtime.voice_embedder.is_ready:
                        speaker = await asyncio.to_thread(
                            runtime.diarizer.classify_partial,
                            seg.audio,
                            default_speaker=seg.speaker,
                        )
                    res = await asyncio.to_thread(runtime.asr.transcribe_pcm, seg.audio)
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
                            runtime.store.save(target_session)
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
                    runtime.store.save(target_session)
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
                    runtime.store.save(target_session)
                if runtime.active_session.id == target_session.id:
                    await broadcast_session()
                return

            decision = runtime.grouper.upsert_and_route(target_session, utt)
            if decision.ignored_stale:
                return

            runtime.store.append_utterance_event(target_session.id, utt)
            if utt.final:
                runtime.store.save(target_session)

            if runtime.active_session.id == target_session.id:
                await broadcast_session(
                    {
                        "latest_qa_id": decision.qa_id,
                        "opened_new_qa": decision.opened_new_qa,
                    }
                )

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
                    )
                )

            if decision.should_speculate_probes and decision.qa_id:
                spawn_bg(precompute_speculative_probes(target_session, decision.qa_id))

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
                    spawn_bg(run_speaker_role_orientation_check(target_session))

        async def run_speaker_role_orientation_check(target_session: InterviewSession) -> None:
            sid = target_session.id
            if sid in runtime.role_orientation_checked or sid in runtime.role_orientation_inflight:
                return
            runtime.role_orientation_inflight.add(sid)
            try:
                res = await runtime.llm.check_speaker_role_orientation(target_session)
                runtime.role_orientation_checked.add(sid)
                if (
                    res.get("inverted") is True
                    and res.get("confidence") == "high"
                    and runtime.active_session.id == sid
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
            target_session: InterviewSession, qa_id: str
        ) -> None:
            if (
                qa_id in runtime.warming_card_ids
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
            if runtime.active_session.id == target_session.id:
                await broadcast_session()
            should_rewarm = False
            try:
                cached = await runtime.llm.suggest_probes(target_session, qa)
                if (
                    runtime.active_session.id == target_session.id
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
                if runtime.active_session.id == target_session.id:
                    await broadcast_session()
            if should_rewarm and runtime.active_session.id == target_session.id:
                spawn_bg(precompute_speculative_probes(target_session, qa_id))

        triage_lock = asyncio.Lock()

        async def run_turn_triage(
            *,
            target_session: InterviewSession,
            utterance_id: str,
            diarize: bool,
        ) -> None:
            async with triage_lock:
                if runtime.active_session.id != target_session.id:
                    return
                utt_obj = target_session.utterance_map().get(utterance_id)
                if utt_obj is None or not utt_obj.final:
                    return
                plan = await runtime.llm.triage_turn(target_session, utt_obj, diarize=diarize)
                if plan is None or runtime.active_session.id != target_session.id:
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

        final_seg_queue: asyncio.Queue[tuple[InterviewSession, SessionAudioRouter, SegmentedSpeech] | None] = (
            asyncio.Queue()
        )

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

        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"] is not None:
                    runtime.begin_capture(require_consent=False, owner=send_json)
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
) -> None:
    action = str(cmd.get("type", ""))

    if action == "new_session":
        cfg_data = dict(cmd.get("config") or {})
        base_cfg = runtime.active_session.config.model_dump()
        # Require explicit fresh consent for each new mock candidate unless explicitly confirmed in cmd
        base_cfg["consent_confirmed"] = False
        cfg = SessionConfig.model_validate({**base_cfg, **cfg_data})
        runtime.switch_session(InterviewSession(config=cfg))
        await broadcast_session()
        return

    if action == "configure_session":
        cfg_data = cmd.get("config") or {}
        cfg = SessionConfig.model_validate({**runtime.active_session.config.model_dump(), **cfg_data})
        runtime.active_session.config = cfg
        runtime.grouper.sync_card_kinds(runtime.active_session)
        runtime.audio_router.mode = cfg.mode
        if not cfg.save_raw_audio or not cfg.consent_confirmed:
            runtime.close_archive()
        if runtime.active_session.utterances:
            runtime.store.save(runtime.active_session)
        await broadcast_session()
        return

    if action == "start_capture":
        runtime.begin_capture(require_consent=True, owner=send_json)
        await broadcast_session()
        return

    if action == "cancel_capture":
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

        runtime.card_probe_gen[qa.id] = runtime.card_probe_gen.get(qa.id, 0) + 1
        runtime.loading_card_ids.add(qa.id)
        await runtime.broadcast_all({"type": "dive_deeper_loading", "qa_id": qa.id})

        async def _run_dive_deeper() -> None:
            try:
                res = await runtime.llm.suggest_probes(target_session, qa)
                qa.speculative_cache = None
                qa.dive_deeper_history.append(res)
                runtime.store.save(target_session)
                if runtime.active_session.id == target_session.id:
                    await runtime.broadcast_all(
                        {"type": "dive_deeper_result", "result": res.model_dump(), "cached": False}
                    )
            finally:
                runtime.loading_card_ids.discard(qa.id)
                if runtime.active_session.id == target_session.id:
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
        runtime.regroup_job = AsyncJobState.RUNNING
        await runtime.broadcast_all({"type": "regroup_loading"})

        async def _run_regroup() -> None:
            try:
                boundaries = await runtime.llm.regroup_cards_full_context(target_session)
                if boundaries:
                    runtime.grouper.apply_regroup_boundaries(target_session, boundaries)
                    runtime.store.save(target_session)
                if runtime.active_session.id == target_session.id:
                    runtime.regroup_job = AsyncJobState.IDLE
                    await broadcast_session({"regrouped": bool(boundaries)})
            except Exception:
                if runtime.active_session.id == target_session.id:
                    runtime.regroup_job = AsyncJobState.ERROR
                    await broadcast_session({"regrouped": False})

        spawn_bg(_run_regroup())
        return

    if action == "flush_and_end":
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
        if cmd.get("rating"):
            target_session.self_rating = InterviewerSelfRating.model_validate(cmd["rating"])
        regroup_first = bool(cmd.get("regroup_first", True))
        runtime.calibration_job = AsyncJobState.RUNNING
        await runtime.broadcast_all({"type": "calibration_loading"})

        async def _run_calibration() -> None:
            try:
                if regroup_first and runtime.llm_config.is_configured:
                    if runtime.active_session.id == target_session.id:
                        runtime.regroup_job = AsyncJobState.RUNNING
                    try:
                        boundaries = await runtime.llm.regroup_cards_full_context(target_session)
                        if boundaries:
                            runtime.grouper.apply_regroup_boundaries(target_session, boundaries)
                    finally:
                        if runtime.active_session.id == target_session.id:
                            runtime.regroup_job = AsyncJobState.IDLE
                report_md = await runtime.llm.generate_calibration_report(target_session)
                target_session.calibration_markdown = report_md
                runtime.store.save(target_session)
                if runtime.active_session.id == target_session.id:
                    runtime.calibration_job = AsyncJobState.IDLE
                    await runtime.broadcast_all({"type": "calibration_ready", "markdown": report_md})
                    await broadcast_session()
            except Exception:
                if runtime.active_session.id == target_session.id:
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
    data["warmup_resolved"] = [u.model_dump() for u in session.warmup_utterances()]

    warming_ids = runtime.warming_card_ids if is_active else set()
    loading_ids = runtime.loading_card_ids if is_active else set()
    card_probe_states: dict[str, str] = {}
    resolved_cards = []
    for qa in session.qa_groups:
        p_state = derive_card_probe_state(
            session,
            qa,
            warming_ids=warming_ids,
            loading_ids=loading_ids,
        )
        card_probe_states[qa.id] = p_state.value
        qd = qa.model_dump()
        qd["badge"] = f"{qa.badge_prefix()}{qa.index}"
        qd["utterances"] = [u.model_dump() for u in session.utterances_for_card(qa, include_drafts=True)]
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
