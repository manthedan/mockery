"""AI Mock Practice Studio HTTP router."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..addons import AddonManager
from ..levels import DEFAULT_LEVEL, normalize_level
from ..models import DEFAULT_PRACTICE_ROLE_TITLE, short_id
from ..practice import (
    PracticeConfig,
    PracticeQuestionItem,
    PracticeSession,
    PracticeTurn,
    estimate_carl_heuristic,
)
from ..rubrics import framework_for
from ..practice_conversation import (
    MAX_DESIGN_NOTES_CHARS,
    build_practice_session_system_prompt,
    commit_practice_conversation_turn,
)
from ..state_machines import (
    AsyncJobState,
    InvalidStateTransitionError,
    PracticePhase,
    compute_allowed_practice_events,
    derive_practice_phase,
    serialize_practice_state_machines,
)

MAX_PRACTICE_TURN_CHARS = 16_000
MAX_WORKSPACE_CONTEXT_BYTES = 200_000
MAX_CUSTOM_QUESTION_CHARS = 6_000
PRACTICE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{3,80}$")


class PracticeStartRequest(BaseModel):
    role_title: str = ""
    company: str = ""
    interview_type: Literal["behavioral", "system_design", "coding", "general"] = "behavioral"
    question_source: Literal["bank_only", "resume_jd_tailored", "mixed"] = "mixed"
    target_level: str = DEFAULT_LEVEL
    coach_style: Literal["supportive", "balanced"] = "supportive"
    live_coach_nudges: bool = True
    voice_enabled: bool = True
    question_count: int = Field(default=4, ge=1, le=8)
    selected_question_id: str | None = None
    custom_question: str = Field(default="", max_length=MAX_CUSTOM_QUESTION_CHARS)
    answer_framework: Literal["carl", "star", "none"] = "carl"


class PracticeTurnRequest(BaseModel):
    text: str
    workspace_context: dict[str, Any] | None = None
    expected_question_id: str | None = None


class PracticeHintRequest(BaseModel):
    workspace_context: dict[str, Any] | None = None


class PracticeNextRequest(BaseModel):
    expected_question_id: str | None = None


class PracticeWorkspaceUpdateRequest(BaseModel):
    workspace_context: dict[str, Any]


class PracticeDebriefRequest(BaseModel):
    final_answer: str | None = None
    expected_question_id: str | None = None
    workspace_context: dict[str, Any] | None = None


class PracticeRetryRequest(BaseModel):
    answer_text: str
    question_id: str | None = None
    focus: str = ""


class _ClientDisconnectedError(Exception):
    """Raised when the HTTP client aborts before an LLM operation commits."""


async def _await_with_disconnect(request: Request, coro: Any) -> Any:
    """
    Await `coro` while watching `request.is_disconnected()`. If the browser aborts
    the fetch (e.g. clicking Cancel), cancel the in-flight LLM task immediately so
    nothing mutates or persists after cancel.
    """
    work_task = asyncio.create_task(coro)

    async def _watch_disconnect() -> bool:
        while not work_task.done():
            try:
                if await request.is_disconnected():
                    return True
            except Exception:
                return False
            await asyncio.sleep(0.1)
        return False

    watch_task = asyncio.create_task(_watch_disconnect())
    try:
        done, _ = await asyncio.wait(
            {work_task, watch_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if watch_task in done and watch_task.result() is True and not work_task.done():
            work_task.cancel()
            try:
                await work_task
            except BaseException:
                pass
            raise _ClientDisconnectedError()
        result = await work_task
        # Final disconnect check right before committing so a cancel within the last poll tick is honored
        try:
            if await request.is_disconnected():
                raise _ClientDisconnectedError()
        except _ClientDisconnectedError:
            raise
        except Exception:
            pass
        return result
    finally:
        if not watch_task.done():
            watch_task.cancel()
            try:
                await watch_task
            except BaseException:
                pass
        if not work_task.done():
            work_task.cancel()
            try:
                await work_task
            except BaseException:
                pass


def merge_workspace_context(session: PracticeSession, patch: dict[str, Any] | None) -> None:
    if not patch:
        return
    if not isinstance(patch, dict):
        raise ValueError("workspace_context must be a JSON object.")
    if "diagram_graph" in patch and patch["diagram_graph"] is not None:
        dg = patch["diagram_graph"]
        if not isinstance(dg, dict):
            raise ValueError("diagram_graph must be a JSON object.")
        if "nodes" in dg and not isinstance(dg["nodes"], list):
            raise ValueError("diagram_graph.nodes must be a list.")
        if "edges" in dg and not isinstance(dg["edges"], list):
            raise ValueError("diagram_graph.edges must be a list.")
    if "design_notes" in patch and patch["design_notes"] is not None:
        if not isinstance(patch["design_notes"], str):
            raise ValueError("design_notes must be a string.")
        if len(patch["design_notes"]) > MAX_DESIGN_NOTES_CHARS:
            raise ValueError(f"design_notes exceeds maximum length ({MAX_DESIGN_NOTES_CHARS:,} characters).")
    for map_key in ("code_by_lang", "tests_by_lang"):
        if map_key in patch and patch[map_key] is not None:
            m = patch[map_key]
            if not isinstance(m, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in m.items()
            ):
                raise ValueError(f"{map_key} must be an object mapping language names to strings.")
    merged = {**session.workspace_context, **patch}
    try:
        encoded = json.dumps(merged, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"Workspace context contains invalid text or data: {exc}") from exc
    if len(encoded) > MAX_WORKSPACE_CONTEXT_BYTES:
        raise ValueError("Workspace context exceeds size limit (200 KB).")
    session.workspace_context = merged


def serialize_practice_session(
    runtime: Any,
    session: PracticeSession,
) -> dict[str, Any]:
    """Serialize a PracticeSession with its canonical derived phase, allowed events, and invariants."""
    debrief_job = runtime.practice_debrief_jobs.get(session.id, AsyncJobState.IDLE)
    sm = serialize_practice_state_machines(session, debrief_job=debrief_job)
    data = session.model_dump()
    data.pop("llm_conversation", None)
    data["rubric"] = session.config.rubric.public_dict()
    data["phase"] = sm["practice"]["phase"]
    data["state_machines"] = sm
    return data


async def _safe_close_practice_worker(runtime: Any, session_id: str) -> None:
    close_fn = getattr(runtime.llm, "close_session_worker", None)
    if close_fn is None:
        return
    try:
        await close_fn(f"practice:{session_id}")
    except Exception:
        pass


def _check_practice_not_debriefing(
    runtime: Any,
    session_id: str,
    event: str,
) -> None:
    if not PRACTICE_SESSION_ID_RE.match(session_id or ""):
        raise HTTPException(status_code=404, detail="Practice session not found.")
    if runtime.practice_debrief_jobs.get(session_id) == AsyncJobState.RUNNING:
        err = InvalidStateTransitionError(
            machine="practice_phase",
            current_state=PracticePhase.DEBRIEFING.value,
            event=event,
            allowed_events=[],
            remediation="Wait for the debrief to finish generating.",
        )
        raise HTTPException(status_code=409, detail=err.to_dict())


def _require_practice_event(
    runtime: Any,
    session: PracticeSession,
    event: str,
) -> PracticePhase:
    debrief_job = runtime.practice_debrief_jobs.get(session.id, AsyncJobState.IDLE)
    phase = derive_practice_phase(session, debrief_job=debrief_job)
    allowed, blocked = compute_allowed_practice_events(session, phase=phase)
    if event not in allowed:
        err = InvalidStateTransitionError(
            machine="practice_phase",
            current_state=phase.value,
            event=event,
            allowed_events=allowed,
            remediation=blocked.get(
                event,
                f"Event '{event}' is not permitted in state '{phase.value}'.",
            ),
        )
        raise HTTPException(status_code=409, detail=err.to_dict())
    return phase


def build_practice_router(runtime: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/practice/sessions")
    async def list_practice_sessions() -> dict[str, Any]:
        items = await asyncio.to_thread(runtime.practice_store.list_recent, 25)
        return {"sessions": items}

    @router.get("/api/practice/sessions/{session_id}")
    async def get_practice_session(session_id: str) -> dict[str, Any]:
        if not PRACTICE_SESSION_ID_RE.match(session_id or ""):
            raise HTTPException(status_code=404, detail="Practice session not found.")
        sess = await asyncio.to_thread(runtime.practice_store.get, session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Practice session not found.")
        return {"session": serialize_practice_session(runtime, sess)}

    @router.post("/api/practice/start")
    async def start_practice_session(body: PracticeStartRequest) -> dict[str, Any]:
        if not runtime.addons._modules.get("practice_mode", True):
            raise HTTPException(
                status_code=403,
                detail="AI Mock Practice is switched off in Settings → Add-Ons.",
            )
        active_jd = runtime.profiles.active_jd()
        role_title = body.role_title.strip() or (active_jd.title if active_jd else DEFAULT_PRACTICE_ROLE_TITLE)
        company = body.company.strip() or (active_jd.company if active_jd else "")
        role_title = runtime.redactor.strip_contact_details(role_title)[0]
        company = runtime.redactor.strip_contact_details(company)[0]
        target_level = normalize_level(body.target_level or (active_jd.target_level if active_jd else "")) or DEFAULT_LEVEL
        jd_text = f"{active_jd.title} {active_jd.company}\n{active_jd.text}" if active_jd else ""
        redacted_resume = runtime.profiles.resume_for_llm()

        bank_candidates = runtime.banks.list_questions(
            category=body.interview_type,
            level=target_level,
            jd_text=jd_text or None,
        )
        if not bank_candidates:
            bank_candidates = runtime.banks.list_questions(
                category=body.interview_type,
                jd_text=jd_text or None,
            )

        # The candidate's saved questions are asked only when picked, and then exactly as written.
        bank_candidates = [q for q in bank_candidates if not q.get("mine")]

        custom_question, removed_details = runtime.redactor.strip_contact_details(body.custom_question.strip())
        selected_qid = body.selected_question_id.strip() if body.selected_question_id else None
        if custom_question:
            selected_qid = None
        if selected_qid:
            picked = runtime.banks.get_question(selected_qid, category=body.interview_type)
            if picked is not None and str(picked.get("category") or "").strip().lower() == body.interview_type:
                if picked.get("mine"):
                    # A saved question of the candidate's own is asked as written, like one typed in.
                    custom_question, removed_details = runtime.redactor.strip_contact_details(
                        str(picked.get("prompt") or "").strip()
                    )
                    selected_qid = None
                else:
                    bank_candidates = [picked] + [q for q in bank_candidates if q.get("id") != picked.get("id")]
            else:
                # A pick made under another format (or since removed) does not carry over; the format decides.
                selected_qid = None

        cfg = PracticeConfig(
            interview_type=body.interview_type,
            question_source=body.question_source,
            target_level=target_level,
            coach_style=body.coach_style,
            live_coach_nudges=body.live_coach_nudges,
            voice_enabled=body.voice_enabled,
            question_count=body.question_count,
            selected_question_id=selected_qid,
            answer_framework=framework_for(body.interview_type, body.answer_framework),
        )

        plan_fallback_reasons: list[str] = []
        try:
            planned_raw, opening = await runtime.llm.generate_practice_plan(
                role_title=role_title,
                company=company,
                interview_type=cfg.interview_type,
                target_level=cfg.target_level,
                question_source=cfg.question_source,
                question_count=cfg.question_count,
                bank_candidates=bank_candidates,
                redacted_resume=redacted_resume,
                jd_text=jd_text,
                selected_question_id=cfg.selected_question_id,
                custom_question=custom_question,
                rubric=cfg.rubric,
                fallback_reasons=plan_fallback_reasons,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"AI interviewer is temporarily unavailable while preparing questions: {exc}",
            ) from exc

        questions: list[PracticeQuestionItem] = []
        for idx, item in enumerate(planned_raw):
            q_item = PracticeQuestionItem.model_validate({**item, "status": "active" if idx == 0 else "pending"})
            questions.append(q_item)

        if not questions:
            questions.append(
                PracticeQuestionItem(
                    source="bank",
                    category=cfg.interview_type,
                    title="Core Experience Walkthrough",
                    competency="Ownership & Impact",
                    prompt=f"Tell me about a challenging project you led that best prepares you for {role_title}.",
                    carl_focus=cfg.rubric.default_focus,
                    status="active",
                )
            )

        first_q = questions[0]
        intro_text = f"{opening.strip()}\n\n{first_q.prompt.strip()}" if opening.strip() else first_q.prompt.strip()
        opening_turn = PracticeTurn(
            role="interviewer",
            question_id=first_q.id,
            text=intro_text,
            fallback_reason=plan_fallback_reasons[0] if plan_fallback_reasons else "",
        )

        session = PracticeSession(
            role_title=role_title,
            company=company,
            config=cfg,
            phase="active",
            questions=questions,
            current_question_index=0,
            turns=[opening_turn],
        )
        await asyncio.to_thread(runtime.practice_store.save, session)
        prewarm_session = getattr(runtime.llm, "prewarm_session_worker", None)
        if prewarm_session is not None:
            prewarm_session(
                f"practice:{session.id}",
                system=build_practice_session_system_prompt(session),
                expect_json=True,
            )
        return {
            "session": serialize_practice_session(runtime, session),
            "opening_turn": opening_turn.model_dump(),
            "removed_details": removed_details,
        }

    @router.post("/api/practice/{session_id}/turn")
    async def submit_practice_turn(
        session_id: str,
        body: PracticeTurnRequest,
        request: Request,
    ) -> dict[str, Any]:
        answer_text = (body.text or "").strip()
        if not answer_text:
            raise HTTPException(status_code=400, detail="Answer text cannot be empty.")
        if len(answer_text) > MAX_PRACTICE_TURN_CHARS:
            raise HTTPException(status_code=413, detail="Answer exceeds maximum allowed length.")
        _check_practice_not_debriefing(runtime, session_id, "submit_turn")

        async with runtime.practice_lock(session_id):
            session = await asyncio.to_thread(runtime.practice_store.get, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Practice session not found.")
            phase = _require_practice_event(runtime, session, "submit_turn")

            cur_q = session.current_question()
            if cur_q is None:
                err = InvalidStateTransitionError(
                    machine="practice_phase",
                    current_state=phase.value,
                    event="submit_turn",
                    allowed_events=[],
                    remediation="No active question in this practice session.",
                )
                raise HTTPException(status_code=409, detail=err.to_dict())
            if body.expected_question_id and body.expected_question_id != cur_q.id:
                allowed, _ = compute_allowed_practice_events(session, phase=phase)
                err = InvalidStateTransitionError(
                    machine="practice_phase",
                    current_state=phase.value,
                    event="submit_turn",
                    allowed_events=allowed,
                    remediation="Active question has already advanced; refresh and answer the current question.",
                )
                raise HTTPException(status_code=409, detail=err.to_dict())

            try:
                merge_workspace_context(session, body.workspace_context)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            next_q = (
                session.questions[session.current_question_index + 1]
                if (session.current_question_index + 1) < len(session.questions)
                else None
            )

            cand_turn = PracticeTurn(
                role="candidate",
                question_id=cur_q.id,
                text=answer_text,
            )
            prior_turns = [t.model_dump() for t in session.turns[-24:]]

            try:
                reply = await _await_with_disconnect(
                    request,
                    runtime.llm.generate_practice_interviewer_reply(
                        role_title=session.role_title,
                        company=session.company,
                        interview_type=session.config.interview_type,
                        target_level=session.config.target_level,
                        coach_style=session.config.coach_style,
                        current_question=cur_q.model_dump(),
                        next_question=next_q.model_dump() if next_q else None,
                        recent_turns=prior_turns,
                        candidate_answer=answer_text,
                        follow_up_count=cur_q.follow_up_count,
                        existing_carl=dict(cur_q.carl_signals),
                        workspace_context=session.workspace_context or None,
                        cache_key=f"mockery:practice:{session.id}",
                        session=session,
                    ),
                )
            except _ClientDisconnectedError:
                await _safe_close_practice_worker(runtime, session.id)
                raise HTTPException(status_code=499, detail="Request cancelled by client.")
            except Exception as exc:
                await _safe_close_practice_worker(runtime, session.id)
                raise HTTPException(
                    status_code=503,
                    detail=f"AI interviewer could not reply right now ({type(exc).__name__}).",
                ) from exc

            conv_commit = reply.pop("_conversation_commit", None)
            rubric = session.config.rubric
            if rubric.coached:
                carl_signals = reply.get("carl_signals") or estimate_carl_heuristic(
                    answer_text, cur_q.carl_signals, rubric
                )
                cur_q.carl_signals = dict(carl_signals)
                cand_turn.carl_signals = dict(carl_signals)
            cur_q.follow_up_count += 1

            nudge = reply.get("coach_nudge")
            if session.config.live_coach_nudges and isinstance(nudge, dict) and nudge.get("message"):
                session.live_nudge = nudge
                cand_turn.coach_tip = str(nudge["message"])
            else:
                session.live_nudge = None

            should_advance = bool(reply.get("should_advance", False))
            raw_reply_text = str(reply.get("interviewer_reply") or "").strip()
            reply_text = raw_reply_text
            target_q_id = cur_q.id
            server_modified_reply = bool(reply.get("reply_replaced"))

            if should_advance:
                cur_q.status = "completed"
                if next_q is not None:
                    session.current_question_index += 1
                    next_q.status = "active"
                    target_q_id = next_q.id
                    prompt_probe = next_q.prompt.strip()[:36].lower()
                    if prompt_probe and prompt_probe not in reply_text.lower():
                        reply_text = f"{reply_text}\n\n{next_q.prompt.strip()}" if reply_text else next_q.prompt.strip()
                        server_modified_reply = True
                else:
                    session.phase = PracticePhase.AWAITING_DEBRIEF.value

            interviewer_turn = PracticeTurn(
                role="interviewer",
                question_id=target_q_id,
                text=reply_text or "Thanks for sharing that — let's keep going.",
                fallback_reason=str(reply.get("fallback_reason") or ""),
            )
            session.turns.append(cand_turn)
            session.turns.append(interviewer_turn)
            if isinstance(conv_commit, dict):
                synced_count = len(session.turns) - 1 if server_modified_reply else len(session.turns)
                commit_practice_conversation_turn(
                    session,
                    system_prompt=str(conv_commit["system_prompt"]),
                    user_content=str(conv_commit["user_content"]),
                    assistant_raw=str(conv_commit["assistant_raw"]),
                    workspace_snapshot=dict(conv_commit["workspace_snapshot"]),
                    synced_turn_count=synced_count,
                    reset_conversation=bool(conv_commit["reset_conversation"]),
                )
            try:
                await AddonManager._await_uninterruptible(asyncio.to_thread(runtime.practice_store.save, session))
            except Exception:
                await _safe_close_practice_worker(runtime, session.id)
                raise
            if session.phase != PracticePhase.ACTIVE.value:
                await _safe_close_practice_worker(runtime, session.id)
            return {
                "session": serialize_practice_session(runtime, session),
                "reply": reply,
                "interviewer_turn": interviewer_turn.model_dump(),
            }

    @router.post("/api/practice/{session_id}/hint")
    async def request_practice_hint(
        session_id: str,
        request: Request,
        body: PracticeHintRequest | None = None,
    ) -> dict[str, Any]:
        _check_practice_not_debriefing(runtime, session_id, "request_hint")
        async with runtime.practice_lock(session_id):
            session = await asyncio.to_thread(runtime.practice_store.get, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Practice session not found.")
            phase = _require_practice_event(runtime, session, "request_hint")

            if body and body.workspace_context:
                try:
                    merge_workspace_context(session, body.workspace_context)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

            cur_q = session.current_question()
            if cur_q is None:
                err = InvalidStateTransitionError(
                    machine="practice_phase",
                    current_state=phase.value,
                    event="request_hint",
                    allowed_events=[],
                    remediation="No active question in this practice session.",
                )
                raise HTTPException(status_code=409, detail=err.to_dict())

            hint_fallback_reasons: list[str] = []
            try:
                hint_text = await _await_with_disconnect(
                    request,
                    runtime.llm.generate_practice_hint(
                        interview_type=session.config.interview_type,
                        target_level=session.config.target_level,
                        current_question=cur_q.model_dump(),
                        recent_turns=[t.model_dump() for t in session.turns[-8:]],
                        workspace_context=session.workspace_context or None,
                        rubric=session.config.rubric,
                        fallback_reasons=hint_fallback_reasons,
                    ),
                )
            except _ClientDisconnectedError:
                raise HTTPException(status_code=499, detail="Hint request cancelled by client.")
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"AI coach could not generate a hint right now: {exc}",
                ) from exc
            coach_turn = PracticeTurn(
                role="coach",
                question_id=cur_q.id,
                text=hint_text,
                fallback_reason=hint_fallback_reasons[0] if hint_fallback_reasons else "",
            )
            session.turns.append(coach_turn)
            session.live_nudge = {"tone": "tip", "message": hint_text}
            await asyncio.to_thread(runtime.practice_store.save, session)
            return {
                "session": serialize_practice_session(runtime, session),
                "hint": hint_text,
                "coach_turn": coach_turn.model_dump(),
            }

    @router.post("/api/practice/{session_id}/next")
    async def advance_practice_question(
        session_id: str,
        body: PracticeNextRequest | None = None,
    ) -> dict[str, Any]:
        _check_practice_not_debriefing(runtime, session_id, "advance_question")
        async with runtime.practice_lock(session_id):
            session = await asyncio.to_thread(runtime.practice_store.get, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Practice session not found.")
            phase = _require_practice_event(runtime, session, "advance_question")

            cur_q = session.current_question()
            if (
                body
                and body.expected_question_id
                and cur_q is not None
                and body.expected_question_id != cur_q.id
            ):
                allowed, _ = compute_allowed_practice_events(session, phase=phase)
                err = InvalidStateTransitionError(
                    machine="practice_phase",
                    current_state=phase.value,
                    event="advance_question",
                    allowed_events=allowed,
                    remediation="Active question has already advanced; refresh before advancing again.",
                )
                raise HTTPException(status_code=409, detail=err.to_dict())

            if cur_q is not None:
                cur_q.status = "completed"

            if session.current_question_index + 1 < len(session.questions):
                session.current_question_index += 1
                next_q = session.questions[session.current_question_index]
                next_q.status = "active"
                session.live_nudge = None
                turn = PracticeTurn(
                    role="interviewer",
                    question_id=next_q.id,
                    text=f"Great, let's move on to the next question:\n\n{next_q.prompt.strip()}",
                )
                session.turns.append(turn)
                await asyncio.to_thread(runtime.practice_store.save, session)
                return {
                    "session": serialize_practice_session(runtime, session),
                    "has_more": True,
                    "interviewer_turn": turn.model_dump(),
                }

            session.phase = PracticePhase.AWAITING_DEBRIEF.value
            session.live_nudge = None
            await asyncio.to_thread(runtime.practice_store.save, session)
            await _safe_close_practice_worker(runtime, session.id)
            return {"session": serialize_practice_session(runtime, session), "has_more": False}

    @router.post("/api/practice/{session_id}/workspace")
    async def update_practice_workspace(
        session_id: str,
        body: PracticeWorkspaceUpdateRequest,
    ) -> dict[str, Any]:
        _check_practice_not_debriefing(runtime, session_id, "update_workspace")
        async with runtime.practice_lock(session_id):
            session = await asyncio.to_thread(runtime.practice_store.get, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Practice session not found.")
            _require_practice_event(runtime, session, "update_workspace")
            try:
                merge_workspace_context(session, body.workspace_context)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            await asyncio.to_thread(runtime.practice_store.save, session)
            return {
                "ok": True,
                "workspace_context": session.workspace_context,
                "state_machines": serialize_practice_state_machines(session),
            }

    @router.post("/api/practice/{session_id}/debrief")
    async def generate_practice_debrief(
        session_id: str,
        body: PracticeDebriefRequest | None = None,
    ) -> dict[str, Any]:
        _check_practice_not_debriefing(runtime, session_id, "generate_debrief")
        async with runtime.practice_lock(session_id):
            session = await asyncio.to_thread(runtime.practice_store.get, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Practice session not found.")
            _require_practice_event(runtime, session, "generate_debrief")
            rubric = session.config.rubric
            if session.phase == PracticePhase.COMPLETED.value and session.debrief is not None:
                if body and body.final_answer and body.final_answer.strip():
                    phase = derive_practice_phase(session)
                    allowed, _ = compute_allowed_practice_events(session, phase=phase)
                    err = InvalidStateTransitionError(
                        machine="practice_phase",
                        current_state=phase.value,
                        event="generate_debrief",
                        allowed_events=allowed,
                        remediation="This practice session was already debriefed in another tab; refresh before continuing.",
                    )
                    raise HTTPException(status_code=409, detail=err.to_dict())
                return {
                    "session": serialize_practice_session(runtime, session),
                    "debrief": session.debrief,
                }
            if body and body.workspace_context:
                try:
                    merge_workspace_context(session, body.workspace_context)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            # If the candidate had an unsent final answer when clicking End & Debrief,
            # include it in-memory before grading (only persisted if debrief succeeds,
            # so a transient 503 never duplicates the turn on retry).
            final_answer = (body.final_answer or "").strip() if body and body.final_answer else ""
            if final_answer:
                if len(final_answer) > MAX_PRACTICE_TURN_CHARS:
                    raise HTTPException(status_code=413, detail="Final answer exceeds maximum allowed length.")
                cur_q = session.current_question()
                if cur_q is None or (body.expected_question_id and body.expected_question_id != cur_q.id):
                    phase = derive_practice_phase(session)
                    allowed, _ = compute_allowed_practice_events(session, phase=phase)
                    err = InvalidStateTransitionError(
                        machine="practice_phase",
                        current_state=phase.value,
                        event="generate_debrief",
                        allowed_events=allowed,
                        remediation="Active question changed before your final answer was recorded; refresh and review before ending.",
                    )
                    raise HTTPException(status_code=409, detail=err.to_dict())
                final_signals: dict[str, str] | None = None
                if rubric.coached:
                    final_signals = estimate_carl_heuristic(final_answer, cur_q.carl_signals, rubric)
                    cur_q.carl_signals = dict(final_signals)
                session.turns.append(
                    PracticeTurn(
                        role="candidate",
                        question_id=cur_q.id,
                        text=final_answer,
                        carl_signals=final_signals,
                    )
                )
            await _safe_close_practice_worker(runtime, session.id)
            questions_snapshot = [
                {**q.model_dump(), "status": "completed" if q.status == "active" else q.status}
                for q in session.questions
            ]
            runtime.practice_debrief_jobs[session.id] = AsyncJobState.RUNNING
            try:
                try:
                    debrief = await runtime.llm.generate_practice_debrief(
                        role_title=session.role_title,
                        company=session.company,
                        interview_type=session.config.interview_type,
                        target_level=session.config.target_level,
                        questions=questions_snapshot,
                        turns=[t.model_dump() for t in session.turns],
                        workspace_context=session.workspace_context or None,
                        rubric=rubric,
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Could not generate coaching debrief right now: {exc}",
                    ) from exc
                answered_ids = {
                    str(t.question_id)
                    for t in session.turns
                    if t.role == "candidate" and not t.is_retry and t.question_id
                }
                review_carl_by_qid: dict[str, dict[str, str]] = {}
                for qr in debrief.get("question_reviews") or []:
                    if isinstance(qr, dict) and qr.get("question_id") and isinstance(qr.get("carl"), dict):
                        qid = str(qr["question_id"])
                        if qid not in review_carl_by_qid and (not answered_ids or qid in answered_ids):
                            review_carl_by_qid[qid] = dict(qr["carl"])
                for q in session.questions:
                    if q.status == "active":
                        q.status = "completed"
                    if rubric.coached and q.id in review_carl_by_qid:
                        q.carl_signals = review_carl_by_qid[q.id]
                if not rubric.coached:
                    for qr in debrief.get("question_reviews") or []:
                        if isinstance(qr, dict):
                            qr.pop("carl", None)
                session.debrief = debrief
                session.phase = PracticePhase.COMPLETED.value
                session.live_nudge = None
                await AddonManager._await_uninterruptible(asyncio.to_thread(runtime.practice_store.save, session))
            finally:
                runtime.practice_debrief_jobs.pop(session.id, None)

            return {
                "session": serialize_practice_session(runtime, session),
                "debrief": debrief,
            }

    @router.post("/api/practice/{session_id}/retry")
    async def submit_practice_retry(
        session_id: str,
        body: PracticeRetryRequest,
        request: Request,
    ) -> dict[str, Any]:
        retry_text = (body.answer_text or "").strip()
        if not retry_text:
            raise HTTPException(status_code=400, detail="Retry answer cannot be empty.")
        if len(retry_text) > MAX_PRACTICE_TURN_CHARS:
            raise HTTPException(status_code=413, detail="Retry answer exceeds maximum allowed length.")
        _check_practice_not_debriefing(runtime, session_id, "submit_retry")

        async with runtime.practice_lock(session_id):
            session = await asyncio.to_thread(runtime.practice_store.get, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Practice session not found.")
            _require_practice_event(runtime, session, "submit_retry")
            rubric = session.config.rubric

            drill = session.debrief.get("retry_drill") if isinstance(session.debrief, dict) else {}
            if not isinstance(drill, dict):
                drill = {}

            if body.question_id is not None and body.question_id.strip():
                target_q = next((q for q in session.questions if q.id == body.question_id.strip()), None)
                if target_q is None:
                    raise HTTPException(status_code=400, detail="Unknown question_id for this practice session.")
            else:
                q_id = drill.get("question_id") or (session.questions[0].id if session.questions else None)
                target_q = next((q for q in session.questions if q.id == q_id), None) or (
                    session.questions[0] if session.questions else None
                )

            drill_focus = body.focus.strip() or str(drill.get("focus") or "Strengthen clarity & impact")
            drill_prompt = str(
                drill.get("prompt")
                or (target_q.prompt if target_q else "Re-deliver your sharpest 45-second answer.")
            )

            orig_answers = [
                t.text
                for t in session.turns
                if t.role == "candidate" and not t.is_retry and (not target_q or t.question_id == target_q.id)
            ]
            original_answer = "\n\n".join(orig_answers) or "No prior answer recorded."

            try:
                feedback = await _await_with_disconnect(
                    request,
                    runtime.llm.evaluate_practice_retry(
                        role_title=session.role_title,
                        target_level=session.config.target_level,
                        drill_focus=drill_focus,
                        drill_prompt=drill_prompt,
                        original_answer=original_answer,
                        retry_answer=retry_text,
                        rubric=rubric,
                    ),
                )
            except _ClientDisconnectedError:
                raise HTTPException(status_code=499, detail="Retry evaluation cancelled by client.")
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not evaluate retry right now: {exc}",
                ) from exc

            retry_turn = PracticeTurn(
                role="candidate",
                question_id=target_q.id if target_q else None,
                text=retry_text,
                is_retry=True,
                carl_signals=(
                    feedback.get("carl_after")
                    if rubric.coached and isinstance(feedback.get("carl_after"), dict)
                    else None
                ),
            )
            if not rubric.coached and isinstance(feedback, dict):
                feedback.pop("carl_after", None)
            session.turns.append(retry_turn)
            session.retry_history.append(
                {
                    "id": short_id("retry"),
                    "question_id": target_q.id if target_q else None,
                    "focus": drill_focus,
                    "prompt": drill_prompt,
                    "candidate_answer": retry_text,
                    "feedback": feedback,
                    "timestamp_ms": int(time.time() * 1000),
                }
            )
            await asyncio.to_thread(runtime.practice_store.save, session)
            return {
                "session": serialize_practice_session(runtime, session),
                "feedback": feedback,
            }

    @router.get("/api/practice/{session_id}/markdown")
    async def export_practice_markdown(session_id: str) -> PlainTextResponse:
        if not PRACTICE_SESSION_ID_RE.match(session_id or ""):
            raise HTTPException(status_code=404, detail="Practice session not found.")
        session = await asyncio.to_thread(runtime.practice_store.get, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Practice session not found.")
        safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", session.id)
        return PlainTextResponse(
            content=session.to_markdown(),
            headers={"Content-Disposition": f'attachment; filename="{safe_id}_practice.md"'},
        )

    return router
