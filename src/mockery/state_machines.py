from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import InterviewSession, QAGroup
    from .practice import PracticeSession
    from .profiles import CandidateProfileState


class InvalidStateTransitionError(ValueError):
    """Structured state-machine rejection designed for clear UX, DX logs, and AX recovery."""

    def __init__(
        self,
        *,
        machine: str,
        current_state: str,
        event: str,
        allowed_events: list[str],
        remediation: str,
    ) -> None:
        super().__init__(f"[{machine}] Cannot apply '{event}' in state '{current_state}'. {remediation}")
        self.machine = machine
        self.current_state = current_state
        self.event = event
        self.allowed_events = allowed_events
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "transition_rejected",
            "machine": self.machine,
            "state": self.current_state,
            "event": self.event,
            "allowed_events": self.allowed_events,
            "remediation": self.remediation,
            "message": self.remediation,
        }


# ---------------------------------------------------------------------------
# 1. VAD Speech-Turn State Machine (per audio stream)
# ---------------------------------------------------------------------------


class VADTurnState(StrEnum):
    SILENCE = "silence"  # Filling pre-roll ring buffer; waiting for speech >= speech_threshold
    ONSET = "onset"  # Tentative speech (< min_speech_ms); filters clicks/coughs
    SPEAKING = "speaking"  # Confirmed speech (>= min_speech_ms); emits periodic draft partials
    TRAILING_PAUSE = "trailing_pause"  # Below silence_threshold; counting toward utterance finalization


VAD_TURN_TRANSITIONS: dict[VADTurnState, set[VADTurnState]] = {
    VADTurnState.SILENCE: {VADTurnState.ONSET},
    VADTurnState.ONSET: {VADTurnState.SPEAKING, VADTurnState.TRAILING_PAUSE, VADTurnState.SILENCE},
    VADTurnState.SPEAKING: {VADTurnState.TRAILING_PAUSE, VADTurnState.SILENCE},
    VADTurnState.TRAILING_PAUSE: {VADTurnState.SPEAKING, VADTurnState.SILENCE},
}


# ---------------------------------------------------------------------------
# 2. Local ASR Engine Lifecycle State Machine
# ---------------------------------------------------------------------------


class ASRModelState(StrEnum):
    IDLE = "idle"
    LOADING = "loading"
    SWITCHING = "switching"
    READY = "ready"
    ERROR = "error"


ASR_MODEL_TRANSITIONS: dict[ASRModelState, set[ASRModelState]] = {
    ASRModelState.IDLE: {ASRModelState.LOADING, ASRModelState.SWITCHING},
    ASRModelState.LOADING: {ASRModelState.READY, ASRModelState.ERROR, ASRModelState.SWITCHING},
    ASRModelState.SWITCHING: {ASRModelState.READY, ASRModelState.ERROR, ASRModelState.SWITCHING},
    ASRModelState.READY: {ASRModelState.SWITCHING},
    ASRModelState.ERROR: {ASRModelState.LOADING, ASRModelState.SWITCHING},
}


# ---------------------------------------------------------------------------
# 3. Utterance Lifecycle State
# ---------------------------------------------------------------------------


class UtteranceState(StrEnum):
    DRAFT = "draft"
    FINAL = "final"
    SUPPRESSED_ECHO = "suppressed_echo"


# ---------------------------------------------------------------------------
# 4. Per-Card "Dive Deeper" Probe Lifecycle State Machine
# ---------------------------------------------------------------------------


class CardProbeState(StrEnum):
    EMPTY = "empty"  # 0 utterances in card
    IDLE = "idle"  # Utterances exist; no speculative cache or revealed probes yet
    WARMING = "warming"  # Speculative background pre-computation in flight
    READY_CACHED = "ready_cached"  # Fresh speculative cache matches current utterance count (0ms hit!)
    STALE_CACHED = "stale_cached"  # Speculative cache exists, but newer utterances arrived since
    LOADING = "loading"  # User-requested Dive Deeper actively running for THIS card
    REVEALED = "revealed"  # Probes displayed and up-to-date with current card utterances
    REVEALED_STALE = "revealed_stale"  # Probes displayed, but newer utterances arrived since


def derive_card_probe_state(
    session: InterviewSession,
    qa: QAGroup,
    *,
    warming_ids: set[str],
    loading_ids: set[str],
) -> CardProbeState:
    utts = session.utterances_for_card(qa, include_drafts=True)
    utt_count = len(utts)
    if utt_count == 0:
        return CardProbeState.EMPTY
    if qa.id in loading_ids:
        return CardProbeState.LOADING
    rev_sum = sum(u.rev for u in utts)

    def _is_fresh(resp: Any) -> bool:
        if resp.based_on_utt_count != utt_count:
            return False
        if resp.based_on_rev_sum > 0 and resp.based_on_rev_sum != rev_sum:
            return False
        return True

    if qa.speculative_cache is not None:
        if _is_fresh(qa.speculative_cache):
            return CardProbeState.READY_CACHED
        if qa.id in warming_ids:
            return CardProbeState.WARMING
        return CardProbeState.STALE_CACHED
    if qa.id in warming_ids:
        return CardProbeState.WARMING
    if qa.dive_deeper_history:
        latest = qa.dive_deeper_history[-1]
        if _is_fresh(latest):
            return CardProbeState.REVEALED
        return CardProbeState.REVEALED_STALE
    return CardProbeState.IDLE


# ---------------------------------------------------------------------------
# 5. Regroup & Calibration Background Job States
# ---------------------------------------------------------------------------


class AsyncJobState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"


# ---------------------------------------------------------------------------
# 6. Interview Session Lifecycle State Machine
# ---------------------------------------------------------------------------


class SessionPhase(StrEnum):
    READY = "ready"  # Fresh session with no transcript turns; live capture can start immediately
    CAPTURING = "capturing"  # Live audio streaming; historical session loads blocked (409)
    AWAITING_SELF_RATING = "awaiting_self_rating"  # Turns recorded & paused/stopped; lock in blind rating next
    RATED = "rated"  # Blind self-rating submitted; ready to reveal AI calibration
    CALIBRATING = "calibrating"  # Regroup + calibration report actively generating
    CALIBRATED = "calibrated"  # Calibration report complete (can re-run or resume capture)


SESSION_PHASE_TRANSITIONS: dict[SessionPhase, set[SessionPhase]] = {
    SessionPhase.READY: {
        SessionPhase.CAPTURING,
        SessionPhase.AWAITING_SELF_RATING,
        SessionPhase.RATED,
        SessionPhase.CALIBRATING,
    },
    SessionPhase.CAPTURING: {
        SessionPhase.READY,
        SessionPhase.AWAITING_SELF_RATING,
        SessionPhase.RATED,
        SessionPhase.CALIBRATED,
    },
    SessionPhase.AWAITING_SELF_RATING: {
        SessionPhase.CAPTURING,
        SessionPhase.RATED,
        SessionPhase.CALIBRATING,
    },
    SessionPhase.RATED: {
        SessionPhase.CAPTURING,
        SessionPhase.CALIBRATING,
        SessionPhase.RATED,
    },
    SessionPhase.CALIBRATING: {
        SessionPhase.CALIBRATED,
        SessionPhase.RATED,
        SessionPhase.AWAITING_SELF_RATING,
    },
    SessionPhase.CALIBRATED: {
        SessionPhase.CAPTURING,
        SessionPhase.RATED,
        SessionPhase.CALIBRATING,
    },
}


def derive_session_phase(
    session: InterviewSession,
    *,
    is_capturing: bool,
    calibration_job: AsyncJobState,
) -> SessionPhase:
    """
    Deterministically computes the canonical `SessionPhase` from explicit domain state,
    eliminating split-brain boolean drift across browser reloads, REST calls, and agents.
    """
    if is_capturing:
        return SessionPhase.CAPTURING
    if calibration_job == AsyncJobState.RUNNING:
        return SessionPhase.CALIBRATING
    if session.calibration_markdown:
        return SessionPhase.CALIBRATED
    if session.self_rating is not None:
        return SessionPhase.RATED
    if session.utterances or session.ended_epoch_ms is not None:
        return SessionPhase.AWAITING_SELF_RATING
    return SessionPhase.READY


def compute_allowed_session_events(
    session: InterviewSession,
    *,
    phase: SessionPhase,
    regroup_job: AsyncJobState,
) -> tuple[list[str], dict[str, str]]:
    """
    Returns `(allowed_events, blocked_reasons)` so UI controls and AI agents know
    exactly what actions are legal right now and *why* any blocked action is unavailable.
    """
    allowed: list[str] = ["configure_session", "new_session", "inject_utterance"]
    blocked: dict[str, str] = {}

    if phase == SessionPhase.CAPTURING:
        allowed.extend(["flush_and_end", "set_manual_speaker"])
        blocked["start_capture"] = "Live capture is already active."
        blocked["load_historical_session"] = "Stop live capture before opening a saved session."
    elif phase == SessionPhase.CALIBRATING:
        blocked["start_capture"] = "Wait for calibration to finish before starting capture."
        blocked["load_historical_session"] = "Wait for calibration to finish before opening a saved session."
    else:
        # Any idle phase (fresh, paused, rated, calibrated) may start or resume live capture.
        allowed.extend(["load_historical_session", "start_capture"])

    if regroup_job == AsyncJobState.RUNNING or phase == SessionPhase.CALIBRATING:
        blocked["force_new_card"] = "Card boundaries are locked while regrouping/calibration is running."
        blocked["split_card"] = "Card boundaries are locked while regrouping/calibration is running."
        blocked["merge_card"] = "Card boundaries are locked while regrouping/calibration is running."
        blocked["regroup_cards"] = "Regrouping is already running."
    else:
        allowed.append("force_new_card")
        if len(session.qa_groups) >= 2:
            allowed.append("merge_card")
        if any(len(qa.utterance_ids) >= 2 for qa in session.qa_groups):
            allowed.append("split_card")
        if len(session.utterances) >= 3:
            allowed.append("regroup_cards")
        else:
            blocked["regroup_cards"] = "Need at least 3 utterances to regroup cards."

    if any(len(qa.utterance_ids) > 0 for qa in session.qa_groups):
        allowed.append("dive_deeper")
    else:
        blocked["dive_deeper"] = "No active card with utterances yet."

    allowed.append("submit_self_rating")

    if phase == SessionPhase.CALIBRATING:
        blocked["generate_calibration"] = "Calibration is already running."
    elif len(session.utterances) == 0:
        blocked["generate_calibration"] = "Record or inject at least one utterance before calibrating."
    else:
        allowed.append("generate_calibration")

    return allowed, blocked


# ---------------------------------------------------------------------------
# 7. Canonical Structural Invariants (DX / AX Verification)
# ---------------------------------------------------------------------------


def validate_session_invariants(session: InterviewSession) -> list[str]:
    """
    Verifies all canonical log & derived-card invariants. Returns a list of violations (empty == valid).
    1. Every `Utterance.id` in `session.utterances` is unique and has `rev >= 1`.
    2. `warmup_utterance_ids` and `qa_groups[*].utterance_ids` form a strict, exact partition of
       `session.utterances` (no dangling IDs, no duplicate ownership across cards, no orphaned utterances).
    3. Card indices are contiguous `1..N`.
    """
    violations: list[str] = []
    seen_ids: set[str] = set()
    for u in session.utterances:
        if u.rev < 1:
            violations.append(f"Utterance {u.id} has invalid rev={u.rev} (< 1)")
        if u.id in seen_ids:
            violations.append(f"Duplicate utterance ID in canonical log: {u.id}")
        seen_ids.add(u.id)

    assigned: list[str] = list(session.warmup_utterance_ids)
    for idx, qa in enumerate(session.qa_groups, start=1):
        if qa.index != idx:
            violations.append(f"Card {qa.id} has non-contiguous index {qa.index} (expected {idx})")
        assigned.extend(qa.utterance_ids)

    assigned_set: set[str] = set()
    for uid in assigned:
        if uid not in seen_ids:
            violations.append(f"Dangling utterance reference '{uid}' not present in canonical log")
        if uid in assigned_set:
            violations.append(f"Utterance '{uid}' is owned by multiple cards/warmup")
        assigned_set.add(uid)

    orphaned = seen_ids - assigned_set
    if orphaned:
        violations.append(f"Orphaned utterances missing from both warmup and cards: {sorted(orphaned)}")

    return violations


# ---------------------------------------------------------------------------
# 8. AI Practice Session Lifecycle State Machine
# ---------------------------------------------------------------------------

VALID_CARL_STATUSES: frozenset[str] = frozenset({"missing", "partial", "strong"})


class PracticePhase(StrEnum):
    ACTIVE = "active"  # Interview in progress; candidate answering current question
    AWAITING_DEBRIEF = "awaiting_debrief"  # All questions finished; waiting to generate CARL debrief
    DEBRIEFING = "debriefing"  # LLM actively generating structured CARL coaching debrief
    COMPLETED = "completed"  # Debrief complete; read-only workspace + 45s retry drills enabled


PRACTICE_PHASE_TRANSITIONS: dict[PracticePhase, set[PracticePhase]] = {
    PracticePhase.ACTIVE: {
        PracticePhase.ACTIVE,
        PracticePhase.AWAITING_DEBRIEF,
        PracticePhase.DEBRIEFING,
    },
    PracticePhase.AWAITING_DEBRIEF: {
        PracticePhase.DEBRIEFING,
    },
    PracticePhase.DEBRIEFING: {
        PracticePhase.COMPLETED,
        PracticePhase.AWAITING_DEBRIEF,
        PracticePhase.ACTIVE,
    },
    PracticePhase.COMPLETED: {
        PracticePhase.COMPLETED,
        PracticePhase.DEBRIEFING,
    },
}


def derive_practice_phase(
    session: PracticeSession,
    *,
    debrief_job: AsyncJobState = AsyncJobState.IDLE,
) -> PracticePhase:
    """Deterministically derives canonical `PracticePhase` from domain state + live debrief job."""
    if debrief_job == AsyncJobState.RUNNING:
        return PracticePhase.DEBRIEFING
    if session.debrief is not None or session.phase == PracticePhase.COMPLETED.value:
        return PracticePhase.COMPLETED
    if (
        session.phase == PracticePhase.AWAITING_DEBRIEF.value
        or not session.questions
        or session.current_question_index >= len(session.questions)
        or all(q.status == "completed" for q in session.questions)
    ):
        return PracticePhase.AWAITING_DEBRIEF
    return PracticePhase.ACTIVE


def compute_allowed_practice_events(
    session: PracticeSession,
    *,
    phase: PracticePhase,
) -> tuple[list[str], dict[str, str]]:
    """Returns `(allowed_events, blocked_reasons)` for Practice Mode UX controls and AX callers."""
    allowed: list[str] = []
    blocked: dict[str, str] = {}

    if phase == PracticePhase.DEBRIEFING:
        msg = "Wait for the debrief to finish generating."
        for ev in (
            "submit_turn",
            "request_hint",
            "advance_question",
            "update_workspace",
            "generate_debrief",
            "submit_retry",
        ):
            blocked[ev] = msg
        return allowed, blocked

    if phase == PracticePhase.ACTIVE:
        if session.current_question() is not None:
            allowed.extend(["submit_turn", "request_hint", "advance_question"])
        else:
            blocked["submit_turn"] = "No active question in this practice session."
            blocked["request_hint"] = "No active question in this practice session."
            blocked["advance_question"] = "No remaining active questions in this session."
        allowed.extend(["update_workspace", "generate_debrief"])
        blocked["submit_retry"] = "Complete the debrief before submitting a retry drill."
        return allowed, blocked

    if phase == PracticePhase.AWAITING_DEBRIEF:
        allowed.extend(["update_workspace", "generate_debrief"])
        blocked["submit_turn"] = "All questions are complete — generate your debrief to continue."
        blocked["request_hint"] = "Hints are only available while a question is active."
        blocked["advance_question"] = "No remaining active questions in this session."
        blocked["submit_retry"] = "Complete the debrief before submitting a retry drill."
        return allowed, blocked

    # PracticePhase.COMPLETED
    allowed.append("generate_debrief")  # idempotent cached debrief read (or recovery if debrief missing)
    if session.debrief is not None:
        allowed.append("submit_retry")
    else:
        blocked["submit_retry"] = "Complete the debrief before submitting a retry drill."
    blocked["submit_turn"] = "This practice session is already completed."
    blocked["request_hint"] = "Hints are only available while a question is active."
    blocked["advance_question"] = "No remaining active questions in this session."
    blocked["update_workspace"] = "Cannot edit workspace after the practice session is completed."
    return allowed, blocked


def validate_practice_invariants(
    session: PracticeSession,
    *,
    debrief_job: AsyncJobState = AsyncJobState.IDLE,
) -> list[str]:
    """Verifies canonical PracticeSession structural invariants (empty list == valid)."""
    violations: list[str] = []
    phase = derive_practice_phase(session, debrief_job=debrief_job)

    q_ids: set[str] = set()
    active_q_indices: list[int] = []
    for idx, q in enumerate(session.questions):
        if not q.id:
            violations.append(f"Question at index {idx} has empty id")
        elif q.id in q_ids:
            violations.append(f"Duplicate question ID in practice session: {q.id}")
        q_ids.add(q.id)
        if q.status == "active":
            active_q_indices.append(idx)
        for dim, val in (q.carl_signals or {}).items():
            if val not in VALID_CARL_STATUSES:
                violations.append(f"Question {q.id} has invalid CARL status '{val}' for '{dim}'")

    seen_turns: set[str] = set()
    for t in session.turns:
        if not t.id:
            violations.append("Practice turn has empty id")
        elif t.id in seen_turns:
            violations.append(f"Duplicate turn ID in practice session: {t.id}")
        seen_turns.add(t.id)
        if t.question_id is not None and t.question_id not in q_ids:
            violations.append(f"Turn {t.id} references unknown question_id '{t.question_id}'")

    if session.questions and not (0 <= session.current_question_index < len(session.questions)):
        violations.append(
            f"current_question_index={session.current_question_index} out of bounds for {len(session.questions)} questions"
        )

    if len(active_q_indices) > 1:
        violations.append(f"Multiple questions marked active simultaneously: {active_q_indices}")

    if phase == PracticePhase.ACTIVE:
        if session.questions and 0 <= session.current_question_index < len(session.questions):
            if session.questions[session.current_question_index].status != "active":
                violations.append(
                    f"Active session question[{session.current_question_index}] has status "
                    f"'{session.questions[session.current_question_index].status}' (expected 'active')"
                )
    elif phase in (PracticePhase.AWAITING_DEBRIEF, PracticePhase.COMPLETED):
        if active_q_indices:
            violations.append(
                f"Session in phase '{phase.value}' still has active question(s): {active_q_indices}"
            )
    if phase == PracticePhase.COMPLETED and session.debrief is None:
        violations.append("Session in phase 'completed' is missing debrief payload")

    return violations


def serialize_practice_state_machines(
    session: PracticeSession,
    *,
    debrief_job: AsyncJobState = AsyncJobState.IDLE,
) -> dict[str, Any]:
    """Canonical `state_machines` payload attached to Practice Mode REST responses."""
    phase = derive_practice_phase(session, debrief_job=debrief_job)
    allowed, blocked = compute_allowed_practice_events(session, phase=phase)
    violations = validate_practice_invariants(session, debrief_job=debrief_job)
    return {
        "practice": {
            "phase": phase.value,
            "allowed_events": allowed,
            "blocked_reasons": blocked,
            "current_question_index": session.current_question_index,
            "question_count": len(session.questions),
        },
        "jobs": {
            "debrief": debrief_job.value,
        },
        "invariants_ok": len(violations) == 0,
        "invariant_violations": violations,
    }


# ---------------------------------------------------------------------------
# 9. Add-On Pack Lifecycle State Machine
# ---------------------------------------------------------------------------


class AddonPackState(StrEnum):
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UNINSTALLING = "uninstalling"
    ERROR = "error"
    COMING_SOON = "coming_soon"


ADDON_PACK_TRANSITIONS: dict[AddonPackState, set[AddonPackState]] = {
    AddonPackState.NOT_INSTALLED: {AddonPackState.INSTALLING},
    AddonPackState.INSTALLING: {AddonPackState.INSTALLED, AddonPackState.ERROR, AddonPackState.NOT_INSTALLED},
    AddonPackState.INSTALLED: {AddonPackState.UNINSTALLING},
    AddonPackState.UNINSTALLING: {AddonPackState.NOT_INSTALLED, AddonPackState.ERROR, AddonPackState.INSTALLED},
    AddonPackState.ERROR: {AddonPackState.INSTALLING, AddonPackState.UNINSTALLING},
    AddonPackState.COMING_SOON: set(),
}


def derive_addon_pack_state(
    *,
    installed: bool,
    available: bool,
    installing: bool = False,
    uninstalling: bool = False,
    error: str | None = None,
) -> AddonPackState:
    if installing:
        return AddonPackState.INSTALLING
    if uninstalling:
        return AddonPackState.UNINSTALLING
    if not available and not installed:
        return AddonPackState.COMING_SOON
    if error:
        return AddonPackState.ERROR
    if installed:
        return AddonPackState.INSTALLED
    return AddonPackState.NOT_INSTALLED


def compute_allowed_addon_pack_events(
    state: AddonPackState,
    *,
    installed: bool,
    available: bool = True,
) -> tuple[list[str], dict[str, str]]:
    allowed: list[str] = []
    blocked: dict[str, str] = {}
    if state == AddonPackState.NOT_INSTALLED:
        allowed.append("install")
        blocked["uninstall"] = "Pack is not installed."
    elif state == AddonPackState.INSTALLED:
        allowed.append("uninstall")
        blocked["install"] = "Pack is already installed."
    elif state == AddonPackState.INSTALLING:
        blocked["install"] = "Installation is already in progress."
        blocked["uninstall"] = "Wait for installation to finish before removing."
    elif state == AddonPackState.UNINSTALLING:
        blocked["install"] = "Wait for removal to finish before installing."
        blocked["uninstall"] = "Removal is already in progress."
    elif state == AddonPackState.ERROR:
        if available:
            allowed.append("install")
        else:
            blocked["install"] = "Not enabled in this build."
        if installed:
            allowed.append("uninstall")
        else:
            blocked["uninstall"] = "Pack is not installed."
    else:  # COMING_SOON
        blocked["install"] = "Not enabled in this build."
        blocked["uninstall"] = "Pack is not installed."
    return allowed, blocked


# ---------------------------------------------------------------------------
# 10. Online Two-Speaker Diarizer Enrollment State Machine
# ---------------------------------------------------------------------------


class DiarizerEnrollmentState(StrEnum):
    UNLOADED = "unloaded"
    COLD_START = "cold_start"
    SINGLE_SPEAKER = "single_speaker"
    TWO_SPEAKERS = "two_speakers"
    MANUAL_CALIBRATED = "manual_calibrated"


DIARIZER_ENROLLMENT_TRANSITIONS: dict[DiarizerEnrollmentState, set[DiarizerEnrollmentState]] = {
    DiarizerEnrollmentState.UNLOADED: {DiarizerEnrollmentState.COLD_START},
    DiarizerEnrollmentState.COLD_START: {
        DiarizerEnrollmentState.SINGLE_SPEAKER,
        DiarizerEnrollmentState.MANUAL_CALIBRATED,
    },
    DiarizerEnrollmentState.SINGLE_SPEAKER: {
        DiarizerEnrollmentState.TWO_SPEAKERS,
        DiarizerEnrollmentState.MANUAL_CALIBRATED,
        DiarizerEnrollmentState.COLD_START,
    },
    DiarizerEnrollmentState.TWO_SPEAKERS: {
        DiarizerEnrollmentState.MANUAL_CALIBRATED,
        DiarizerEnrollmentState.COLD_START,
    },
    DiarizerEnrollmentState.MANUAL_CALIBRATED: {
        DiarizerEnrollmentState.MANUAL_CALIBRATED,
        DiarizerEnrollmentState.COLD_START,
    },
}


def derive_diarizer_enrollment_state(
    *,
    embedder_ready: bool,
    enrolled_count: int,
    manual_override: bool,
) -> DiarizerEnrollmentState:
    if not embedder_ready:
        return DiarizerEnrollmentState.UNLOADED
    if manual_override and enrolled_count > 0:
        return DiarizerEnrollmentState.MANUAL_CALIBRATED
    if enrolled_count >= 2:
        return DiarizerEnrollmentState.TWO_SPEAKERS
    if enrolled_count == 1:
        return DiarizerEnrollmentState.SINGLE_SPEAKER
    return DiarizerEnrollmentState.COLD_START


def compute_allowed_diarizer_events(
    state: DiarizerEnrollmentState,
    *,
    has_utterance_history: bool = False,
) -> tuple[list[str], dict[str, str]]:
    # swap_roles & reset work at any point (including cold-start transcript label flip + manual lock)
    allowed: list[str] = ["reset", "swap_roles"]
    blocked: dict[str, str] = {}
    if state == DiarizerEnrollmentState.UNLOADED:
        blocked["manual_correction"] = "Speaker diarizer model is not loaded yet."
    elif has_utterance_history:
        allowed.append("manual_correction")
    else:
        blocked["manual_correction"] = "No voiced utterance embeddings recorded yet for acoustic adaptation."
    return allowed, blocked


# ---------------------------------------------------------------------------
# 11. Candidate Resume Redaction & Local TTS Backend States
# ---------------------------------------------------------------------------


class ResumeRedactionState(StrEnum):
    NO_RESUME = "no_resume"
    REDACTED_RULES_ONLY = "redacted_rules_only"
    REDACTED_RAMPART = "redacted_rampart"


def derive_resume_redaction_state(*, has_resume: bool, engine: str = "") -> ResumeRedactionState:
    if not has_resume:
        return ResumeRedactionState.NO_RESUME
    if "rampart" in (engine or "").lower():
        return ResumeRedactionState.REDACTED_RAMPART
    return ResumeRedactionState.REDACTED_RULES_ONLY


def validate_profile_invariants(profile_state: CandidateProfileState) -> list[str]:
    """Verifies candidate profile invariants (unique JD IDs, valid active_jd_id, non-empty redaction)."""
    violations: list[str] = []
    jd_ids: set[str] = set()
    for jd in profile_state.job_listings:
        if not jd.id:
            violations.append("Job listing has empty id")
        elif jd.id in jd_ids:
            violations.append(f"Duplicate job listing ID: {jd.id}")
        jd_ids.add(jd.id)
    if profile_state.active_jd_id is not None and profile_state.active_jd_id not in jd_ids:
        violations.append(
            f"active_jd_id '{profile_state.active_jd_id}' does not match any stored job listing"
        )
    if profile_state.resume is not None:
        if profile_state.resume.raw_text.strip() and not profile_state.resume.redacted_text.strip():
            violations.append("Stored resume has raw_text but empty redacted_text")
    return violations


class TTSBackendState(StrEnum):
    KOKORO_READY = "kokoro_ready"
    PHONON_READY = "phonon_ready"
    KOKORO_FALLBACK_MACOS = "kokoro_fallback_macos"
    PHONON_FALLBACK_MACOS = "phonon_fallback_macos"
    MACOS_SAY = "macos_say"
    BROWSER_FALLBACK = "browser_fallback"


def derive_tts_backend_state(
    *,
    kokoro_installed: bool,
    kokoro_runtime: bool,
    kokoro_error: str | None,
    macos_say_available: bool,
    phonon_installed: bool = False,
    phonon_runtime: bool = False,
    phonon_error: str | None = None,
    preferred_engine: str | None = None,
) -> TTSBackendState:
    kokoro_ready = kokoro_installed and kokoro_runtime and not kokoro_error
    phonon_ready = phonon_installed and phonon_runtime and not phonon_error
    if preferred_engine == "phonon" and phonon_ready:
        return TTSBackendState.PHONON_READY
    if kokoro_ready:
        return TTSBackendState.KOKORO_READY
    if phonon_ready:
        return TTSBackendState.PHONON_READY
    if kokoro_installed and kokoro_runtime and kokoro_error and macos_say_available:
        return TTSBackendState.KOKORO_FALLBACK_MACOS
    if phonon_installed and phonon_runtime and phonon_error and macos_say_available:
        return TTSBackendState.PHONON_FALLBACK_MACOS
    if macos_say_available:
        return TTSBackendState.MACOS_SAY
    return TTSBackendState.BROWSER_FALLBACK


def export_machine_specs() -> dict[str, Any]:
    """Machine-readable & Mermaid specification of all state machines for DX and AX introspection."""
    return {
        "machines": {
            "session_phase": {
                "states": [s.value for s in SessionPhase],
                "transitions": {k.value: sorted(v.value for v in vs) for k, vs in SESSION_PHASE_TRANSITIONS.items()},
                "mermaid": (
                    "stateDiagram-v2\n"
                    "  [*] --> ready\n"
                    "  ready --> capturing : start_capture\n"
                    "  capturing --> awaiting_self_rating : flush_and_end\n"
                    "  awaiting_self_rating --> capturing : start_capture (resume)\n"
                    "  awaiting_self_rating --> rated : submit_self_rating\n"
                    "  rated --> calibrating : generate_calibration\n"
                    "  calibrating --> calibrated : calibration_ready\n"
                    "  calibrated --> capturing : start_capture (resume)"
                ),
            },
            "card_probe": {
                "states": [s.value for s in CardProbeState],
                "description": "Per-card Dive Deeper probe lifecycle distinguishing instant fresh cache vs stale cache vs active loading.",
            },
            "vad_turn": {
                "states": [s.value for s in VADTurnState],
                "transitions": {k.value: sorted(v.value for v in vs) for k, vs in VAD_TURN_TRANSITIONS.items()},
            },
            "asr_model": {
                "states": [s.value for s in ASRModelState],
                "transitions": {k.value: sorted(v.value for v in vs) for k, vs in ASR_MODEL_TRANSITIONS.items()},
            },
            "utterance": {
                "states": [s.value for s in UtteranceState],
                "invariants": [
                    "Monotonic revision (`rev >= 1`) per `Utterance.id`.",
                    "Strict partition: every stored utterance belongs to warmup OR exactly one QAGroup.",
                ],
            },
            "practice_phase": {
                "states": [s.value for s in PracticePhase],
                "transitions": {k.value: sorted(v.value for v in vs) for k, vs in PRACTICE_PHASE_TRANSITIONS.items()},
                "invariants": [
                    "Unique question and turn IDs; every turn.question_id references an existing question.",
                    "At most one question has status='active'; zero active questions in awaiting_debrief/completed.",
                    "Completed practice sessions must have a debrief payload.",
                ],
                "mermaid": (
                    "stateDiagram-v2\n"
                    "  [*] --> active : start_practice\n"
                    "  active --> active : submit_turn / advance_question (has_more)\n"
                    "  active --> awaiting_debrief : advance_question (last)\n"
                    "  active --> debriefing : generate_debrief (early)\n"
                    "  awaiting_debrief --> debriefing : generate_debrief\n"
                    "  debriefing --> completed : debrief_ready\n"
                    "  debriefing --> active : debrief_error (early)\n"
                    "  debriefing --> awaiting_debrief : debrief_error\n"
                    "  completed --> debriefing : recover_debrief\n"
                    "  completed --> completed : submit_retry"
                ),
            },
            "addon_pack": {
                "states": [s.value for s in AddonPackState],
                "transitions": {k.value: sorted(v.value for v in vs) for k, vs in ADDON_PACK_TRANSITIONS.items()},
                "mermaid": (
                    "stateDiagram-v2\n"
                    "  [*] --> not_installed\n"
                    "  not_installed --> installing : install\n"
                    "  installing --> installed : ok\n"
                    "  installing --> error : failed\n"
                    "  error --> installing : retry_install\n"
                    "  installed --> uninstalling : uninstall\n"
                    "  uninstalling --> not_installed : ok\n"
                    "  uninstalling --> error : failed"
                ),
            },
            "diarizer_enrollment": {
                "states": [s.value for s in DiarizerEnrollmentState],
                "transitions": {
                    k.value: sorted(v.value for v in vs) for k, vs in DIARIZER_ENROLLMENT_TRANSITIONS.items()
                },
            },
            "resume_redaction": {
                "states": [s.value for s in ResumeRedactionState],
                "invariants": [
                    "Actively redacted resume PII spans are masked over HTTP and stripped before LLM prompts; only spans explicitly unhidden by the user are restored.",
                    "active_jd_id is either None or references an existing job listing.",
                ],
            },
            "tts_backend": {
                "states": [s.value for s in TTSBackendState],
                "description": "Local-first TTS engine cascade: Kokoro-82M neural voice -> macOS native say -> browser speech fallback.",
            },
        }
    }
