from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import InterviewSession, QAGroup


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
    SETUP = "setup"  # Fresh session; consent not yet confirmed and no transcript turns
    READY = "ready"  # Consent confirmed; ready to start live capture
    CAPTURING = "capturing"  # Live audio streaming; historical session loads blocked (409)
    AWAITING_SELF_RATING = "awaiting_self_rating"  # Turns recorded & paused/stopped; lock in blind rating next
    RATED = "rated"  # Blind self-rating submitted; ready to reveal AI calibration
    CALIBRATING = "calibrating"  # Regroup + calibration report actively generating
    CALIBRATED = "calibrated"  # Calibration report complete (can re-run or resume capture)


SESSION_PHASE_TRANSITIONS: dict[SessionPhase, set[SessionPhase]] = {
    SessionPhase.SETUP: {
        SessionPhase.READY,
        SessionPhase.CAPTURING,
        SessionPhase.AWAITING_SELF_RATING,
        SessionPhase.RATED,
        SessionPhase.CALIBRATING,
    },
    SessionPhase.READY: {
        SessionPhase.SETUP,
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
    if session.config.consent_confirmed:
        return SessionPhase.READY
    return SessionPhase.SETUP


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
        allowed.append("load_historical_session")
        if session.config.consent_confirmed:
            allowed.append("start_capture")
        else:
            blocked["start_capture"] = "Confirm candidate consent (`consent_confirmed=true`) before starting mic/tab capture."

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


def export_machine_specs() -> dict[str, Any]:
    """Machine-readable & Mermaid specification of all state machines for DX and AX introspection."""
    return {
        "machines": {
            "session_phase": {
                "states": [s.value for s in SessionPhase],
                "transitions": {k.value: sorted(v.value for v in vs) for k, vs in SESSION_PHASE_TRANSITIONS.items()},
                "mermaid": (
                    "stateDiagram-v2\n"
                    "  [*] --> setup\n"
                    "  setup --> ready : confirm_consent\n"
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
        }
    }
