from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .levels import DEFAULT_LEVEL, normalize_level
from .models import DEFAULT_PRACTICE_ROLE_TITLE, short_id
from .practice_conversation import PracticeConversation
from .rubrics import (
    CARL,
    CODING,
    GENERAL,
    SIGNAL_SLOTS,
    STAR,
    SYSTEM_DESIGN,
    AnswerFramework,
    Rubric,
    rubric_for,
)

PracticeInterviewType = Literal["behavioral", "system_design", "coding", "general"]
PracticeQuestionSource = Literal["bank_only", "resume_jd_tailored", "mixed"]
PracticePhase = Literal["active", "awaiting_debrief", "debriefing", "completed"]
CarlStatus = Literal["strong", "partial", "missing"]
MAX_CACHED_PRACTICE_SESSIONS = 32


class PracticeConfig(BaseModel):
    interview_type: PracticeInterviewType = "behavioral"
    question_source: PracticeQuestionSource = "mixed"
    target_level: str = DEFAULT_LEVEL
    coach_style: Literal["supportive", "balanced"] = "supportive"
    live_coach_nudges: bool = True
    voice_enabled: bool = True
    question_count: int = Field(default=4, ge=1, le=8)
    selected_question_id: str | None = None
    answer_framework: AnswerFramework = "carl"
    """Sessions saved before frameworks existed were all graded on CARL, which is the default."""

    @property
    def rubric(self) -> Rubric:
        return rubric_for(self.interview_type, self.answer_framework)

    @field_validator("target_level", mode="before")
    @classmethod
    def _fold_ladder_codes(cls, value: object) -> str:
        return normalize_level(value) or DEFAULT_LEVEL


class PracticeQuestionItem(BaseModel):
    id: str = Field(default_factory=lambda: short_id("pq"))
    bank_question_id: str | None = None
    source: Literal["bank", "tailored", "custom"] = "bank"
    category: str = "behavioral"
    title: str = ""
    competency: str = ""
    prompt: str
    carl_focus: str = ""
    follow_ups: list[str] = Field(default_factory=list)
    core_requirements: list[str] = Field(default_factory=list)
    deep_dives: list[str] = Field(default_factory=list)
    starter_code: dict[str, str] = Field(default_factory=dict)
    tests: dict[str, str] = Field(default_factory=dict)
    difficulty: str = ""
    level_bar: str = ""
    status: Literal["pending", "active", "completed"] = "pending"
    follow_up_count: int = 0
    carl_signals: dict[str, str] = Field(
        default_factory=lambda: {
            "context": "missing",
            "action": "missing",
            "result": "missing",
            "learning": "missing",
        }
    )


class PracticeTurn(BaseModel):
    id: str = Field(default_factory=lambda: short_id("pt"))
    role: Literal["interviewer", "candidate", "coach"]
    question_id: str | None = None
    text: str
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    is_retry: bool = False
    carl_signals: dict[str, str] | None = None
    coach_tip: str = ""
    fallback_reason: str = ""
    """Set on an interviewer or coach turn that is built-in text because the connected model failed."""


class PracticeSession(BaseModel):
    id: str = Field(default_factory=lambda: short_id("prac"))
    created_at_iso: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    role_title: str = DEFAULT_PRACTICE_ROLE_TITLE
    company: str = ""
    config: PracticeConfig = Field(default_factory=PracticeConfig)
    phase: PracticePhase = "active"
    questions: list[PracticeQuestionItem] = Field(default_factory=list)
    current_question_index: int = 0
    turns: list[PracticeTurn] = Field(default_factory=list)
    live_nudge: dict[str, Any] | None = None
    debrief: dict[str, Any] | None = None
    retry_history: list[dict[str, Any]] = Field(default_factory=list)
    workspace_context: dict[str, Any] = Field(default_factory=dict)
    llm_conversation: PracticeConversation | None = None

    def current_question(self) -> PracticeQuestionItem | None:
        if 0 <= self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    def to_markdown(self) -> str:
        lines: list[str] = []
        header = f"{self.role_title}" + (f" · {self.company}" if self.company else "")
        lines.append(f"# Mockery AI Practice Coaching Report — {header}")
        lines.append("")
        lines.append(
            f"- **Format:** {self.config.interview_type.replace('_', ' ').title()} "
            f"| **Level:** {self.config.target_level} "
            f"| **Source:** {self.config.question_source.replace('_', ' ')} "
            f"| **Date:** {self.created_at_iso}"
        )
        lines.append("")

        if isinstance(self.debrief, dict):
            d = self.debrief
            if d.get("overall_encouragement"):
                lines.append("## Coach's Note")
                lines.append("")
                lines.append(str(d["overall_encouragement"]))
                lines.append("")
            strengths = [str(s) for s in (d.get("top_strengths") or []) if s]
            if strengths:
                lines.append("## What you did well")
                lines.append("")
                for s in strengths:
                    lines.append(f"- {s}")
                lines.append("")
            q_reviews = [qr for qr in (d.get("question_reviews") or []) if isinstance(qr, dict)]
            if q_reviews:
                rubric = self.config.rubric
                framework = f"{rubric.name} " if rubric.named_framework else ""
                lines.append(f"## Question-by-question {framework}coaching")
                lines.append("")
                for idx, qr in enumerate(q_reviews, 1):
                    lines.append(f"### Q{idx}: {qr.get('title') or str(qr.get('prompt', ''))[:80]}")
                    graded = qr.get("carl") if isinstance(qr.get("carl"), dict) else {}
                    if rubric.coached:
                        marks = " · ".join(
                            f"{signal.label} `{graded.get(signal.slot, '—')}`" for signal in rubric.signals
                        )
                        lines.append(f"- **{rubric.name}:** {marks}")
                    if qr.get("strength"):
                        lines.append(f"- **Strength:** {qr['strength']}")
                    if qr.get("coaching_tip"):
                        lines.append(f"- **Next-Level Tip:** {qr['coaching_tip']}")
                    if qr.get("carl_reframe"):
                        lines.append(f"- **Suggested {framework}Reframe:** {qr['carl_reframe']}")
                    lines.append("")
            drill = d.get("retry_drill") if isinstance(d.get("retry_drill"), dict) else {}
            if drill.get("prompt"):
                lines.append("## Recommended 45-second retry drill")
                lines.append("")
                lines.append(f"- **Focus:** {drill.get('focus', 'Strengthen your answer')}")
                lines.append(f"- **Prompt:** {drill['prompt']}")
                lines.append("")

        if self.retry_history:
            lines.append("## Retry drill progress")
            lines.append("")
            for idx, r in enumerate(self.retry_history, 1):
                lines.append(f"### Retry #{idx}")
                lines.append(f"> {r.get('candidate_answer', '')}")
                lines.append("")
                fb = r.get("feedback") or {}
                if fb.get("encouragement"):
                    lines.append(f"- **Coach Feedback:** {fb['encouragement']}")
                if fb.get("what_improved"):
                    lines.append(f"- **What Improved:** {fb['what_improved']}")
                if fb.get("one_more_polish"):
                    lines.append(f"- **One Final Polish:** {fb['one_more_polish']}")
                lines.append("")

        def _fenced_block(content: str, lang: str = "") -> list[str]:
            runs = re.findall(r"`+", content)
            fence = "`" * max(3, max((len(r) for r in runs), default=0) + 1)
            return [f"{fence}{lang}", content, fence]

        if isinstance(self.workspace_context, dict):
            diag = str(self.workspace_context.get("diagram_summary") or "").strip()
            if diag:
                lines.append("## System Design Whiteboard Summary")
                lines.append("")
                lines.extend(_fenced_block(diag[:6000], "text"))
                lines.append("")
            code = str(self.workspace_context.get("code") or "").strip()
            if code:
                lang = re.sub(r"[^A-Za-z0-9_+-]", "", str(self.workspace_context.get("language") or "text")) or "text"
                lines.append(f"## Coding Studio Solution ({lang})")
                lines.append("")
                lines.extend(_fenced_block(code[:12000], lang))
                test_out = str(self.workspace_context.get("test_output") or "").strip()
                if test_out:
                    lines.append("")
                    lines.append("**Latest Test Output:**")
                    lines.extend(_fenced_block(test_out[:4000], "text"))
                lines.append("")

        lines.append("## Full practice transcript")
        lines.append("")
        for t in self.turns:
            if t.role == "coach":
                lines.append(f"- **Coach tip:** {t.text}")
            elif t.role == "interviewer":
                lines.append(f"- **AI Interviewer:** {t.text}")
            else:
                tag = " (Retry)" if t.is_retry else ""
                lines.append(f"- **You{tag}:** {t.text}")
        lines.append("")
        return "\n".join(lines)


_FILLER_RE = re.compile(
    r"\b(?:u+h+m*|u+m+|h+m+|e+r+m*|a+h+|o+h+|yeah|yep|yes|no|nope|well|so|like|okay|ok|right|sure|"
    r"you know|i mean|i guess|let me think|let me see|not sure|no idea|i don't know|i dunno|"
    r"(?:great|good|nice|tough|hard|interesting) question)\b",
    re.IGNORECASE,
)
MIN_ANSWER_WORDS = 3
"""Fewer real words than this, once the filler is set aside, is not an answer."""


def said_too_little(answer_text: str | None) -> bool:
    """True when nothing was really said: only filler, or a word or two. A number always counts
    ("$130k to $150k base", "Yes, in 2021"), so a short factual answer is still an answer."""
    text = answer_text or ""
    if re.search(r"\d", text):
        return False
    return len(re.findall(r"[A-Za-z']+", _FILLER_RE.sub(" ", text))) < MIN_ANSWER_WORDS


_RANK = {"missing": 0, "partial": 1, "strong": 2}
_LEVEL = {0: "missing", 1: "partial", 2: "strong"}
_METRIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%(?!\w)|(?:ms|s|x|k|m|million|thousand|hours|days|weeks|users|qps|rps)\b)"
)
_SETTING_CUES = (
    "when i", "at my", "our team", "project", "problem", "incident", "outage",
    "context", "situation", "legacy", "migrat", "quarter", "stakeholder", "deadline",
)
_OWN_ACTION_CUES = (
    "i designed", "i built", "i led", "i proposed", "i refactored", "i implemented",
    "i wrote", "i decided", "i aligned", "i benchmarked", "i set up", "i profiled",
    "i debugged", "my approach", "i chose", "i worked with", "i convinced",
)
_OUTCOME_CUES = (
    "reduced", "improved", "increased", "shipped", "launched", "saved",
    "result", "outcome", "impact", "resolved", "adopted", "unblocked",
)
_REFLECTION_CUES = (
    "learned", "takeaway", "next time", "retrospective", "in hindsight",
    "changed how", "now i always", "lesson", "would do differently", "taught me",
)
_OWNED_TASK_CUES = (
    "my job was", "my role was", "i was responsible", "my responsibility", "i was asked to",
    "i was tasked", "it fell to me", "i owned", "my goal was", "i was brought in to",
)
_GOAL_CUES = ("needed to", "had to", "the goal", "our goal", "objective", "responsible for", "supposed to")
_EXAMPLE_CUES = ("for example", "for instance", "specifically", "such as", "one time", "last year", "recently")
_ROLE_FIT_CUES = (
    "this role", "the role", "this position", "this job", "your team", "this team",
    "your company", "the company", "your product", "your mission", "your customers", "working here",
)


def _setting_level(low: str, wc: int) -> str:
    has_setting = any(k in low for k in _SETTING_CUES)
    # Length alone earns partial credit only for a long answer; a dozen words of filler set no scene.
    return "strong" if (has_setting and wc >= 20) else ("partial" if has_setting or wc >= 40 else "missing")


def _own_action_level(low: str, wc: int) -> str:
    has_action = any(k in low for k in _OWN_ACTION_CUES) or (" i " in f" {low} " and wc >= 25)
    return "strong" if (has_action and wc >= 35) else ("partial" if has_action else "missing")


def _outcome_level(low: str, wc: int) -> str:
    if _METRIC_RE.search(low):
        return "strong"
    return "partial" if any(k in low for k in _OUTCOME_CUES) else "missing"


def _reflection_level(low: str, wc: int) -> str:
    return "strong" if any(k in low for k in _REFLECTION_CUES) else "missing"


def _owned_task_level(low: str, wc: int) -> str:
    if any(k in low for k in _OWNED_TASK_CUES):
        return "strong"
    return "partial" if any(k in low for k in _GOAL_CUES) else "missing"


def _direct_level(low: str, wc: int) -> str:
    first_sentence = re.split(r"[.!?]", low, maxsplit=1)[0].split()
    # A short factual answer ("$130k to $150k base") is as direct as an answer gets.
    if len(first_sentence) <= 30 and (wc >= 12 or re.search(r"\d", low)):
        return "strong"
    return "partial"


def _specific_level(low: str, wc: int) -> str:
    if _METRIC_RE.search(low) or any(k in low for k in _EXAMPLE_CUES):
        return "strong"
    return "partial" if ("because" in low or wc >= 40) else "missing"


def _role_fit_level(low: str, wc: int) -> str:
    hits = sum(1 for k in _ROLE_FIT_CUES if k in low)
    return "strong" if hits >= 2 else ("partial" if hits == 1 else "missing")


def _concise_level(low: str, wc: int) -> str:
    if wc <= 220:
        return "strong"
    return "partial" if wc <= 350 else "missing"


def _cue_level(cues: tuple[str, ...]) -> Any:
    """Two distinct cues is a clear signal, one is a mention."""

    def level(low: str, wc: int) -> str:
        hits = sum(1 for cue in cues if cue in low)
        return "strong" if hits >= 2 else ("partial" if hits == 1 else "missing")

    return level


def _scale_level(low: str, wc: int) -> str:
    if _METRIC_RE.search(low):
        return "strong"
    return _cue_level(("bottleneck", "throughput", "partition", "hot key", "capacity", "scale", "peak", "load"))(low, wc)


def _complexity_level(low: str, wc: int) -> str:
    if re.search(r"\bo\(|o of \w", low):
        return "strong"
    return _cue_level(("complexity", "linear", "constant time", "quadratic", "logarithmic", "log n", "amortized"))(low, wc)


_CARL_SCORERS = {
    "context": _setting_level,
    "action": _own_action_level,
    "result": _outcome_level,
    "learning": _reflection_level,
}
_SCORERS_BY_RUBRIC = {
    SYSTEM_DESIGN.id: {
        "context": _cue_level(
            ("requirement", "constraint", "users", "latency", "read-heavy", "write-heavy", "sla", "assum", "scope")
        ),
        "action": _cue_level(
            ("service", "database", "cache", "queue", "load balancer", "api", "shard", "replica", "storage", "gateway")
        ),
        "result": _scale_level,
        "learning": _cue_level(
            ("tradeoff", "trade-off", "instead of", "versus", "at the cost", "consistency", "availability", "downside")
        ),
    },
    CODING.id: {
        "context": _cue_level(
            ("approach", "brute force", "plan", "idea", "i would", "i'd", "hash", "pointer", "sort", "stack", "recurs")
        ),
        "action": _cue_level(("works", "passes", "correct", "handles", "returns", "invariant", "walk through", "trace")),
        "result": _complexity_level,
        "learning": _cue_level(
            ("edge case", "empty", "null", "duplicate", "overflow", "boundary", "negative", "single element", "zero")
        ),
    },
    STAR.id: {
        "context": _setting_level,
        "action": _owned_task_level,
        "result": _own_action_level,
        "learning": _outcome_level,
    },
    GENERAL.id: {
        "context": _direct_level,
        "action": _specific_level,
        "result": _role_fit_level,
        "learning": _concise_level,
    },
}


def estimate_carl_heuristic(
    answer_text: str,
    existing: dict[str, str] | None = None,
    rubric: Rubric = CARL,
) -> dict[str, str]:
    """
    Fast, deterministic estimate of the four coaching signals, used as the instant baseline
    and the offline fallback. Each slot is scored by what it means in ``rubric``.
    Never downgrades a signal across follow-up turns for the same question.
    """
    low = (answer_text or "").lower()
    wc = len(low.split())
    prev = existing or {}
    scorers = _SCORERS_BY_RUBRIC.get(rubric.id, _CARL_SCORERS)
    answered = not said_too_little(answer_text)
    return {
        slot: _LEVEL[
            max(_RANK.get(prev.get(slot, "missing"), 0), _RANK[scorers[slot](low, wc)] if answered else 0)
        ]
        for slot in SIGNAL_SLOTS
    }


def build_heuristic_nudge(
    carl: dict[str, str],
    interview_type: str,
    rubric: Rubric = CARL,
    answer_text: str | None = None,
) -> dict[str, str]:
    """Deterministic coach nudge when no model is connected. It praises only what the answer earned."""
    too_little = answer_text is not None and said_too_little(answer_text)
    # Short remarks are normal while drawing or coding, so a studio round keeps its tip and drops the praise.
    if interview_type == "system_design":
        tip = "call out one concrete bottleneck (traffic, storage, or failure mode) and how you'd mitigate it."
        return {"tone": "tip", "message": tip.capitalize() if too_little else f"Good momentum — {tip}"}
    if interview_type == "coding":
        tip = "state the time/space complexity and check an edge case before finalizing."
        return {"tone": "tip", "message": tip.capitalize() if too_little else f"Nice thinking out loud — {tip}"}
    if too_little:
        return {"tone": "tip", "message": rubric.nothing_yet}
    for slot in rubric.nudge_order:
        if carl.get(slot) == "missing":
            signal = rubric.signal(slot)
            before = rubric.preceding(slot)
            earned = before is not None and carl.get(before.slot, "missing") != "missing"
            message = f"{signal.earned_praise} {signal.nudge}" if earned and signal.earned_praise else signal.nudge
            return {"tone": signal.nudge_tone if earned else "tip", "message": message}
    if all(carl.get(slot, "missing") == "missing" for slot in SIGNAL_SLOTS):
        return {"tone": "tip", "message": rubric.reframe_tip}
    return {"tone": "encourage", "message": rubric.all_landed}


class PracticeSessionStore:
    """
    Thread-safe JSON persistence for AI Practice sessions under `<data_root>/practice/<id>.json`
    with exclusive 0600 atomic writes, snapshot-isolated bounded LRU caching, and
    `(mtime_ns, size)`-keyed summary caching for fast `list_recent()`.
    """

    def __init__(self, practice_dir: Path) -> None:
        self.practice_dir = practice_dir
        self.practice_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.practice_dir, 0o700)
        except OSError:
            pass
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, PracticeSession] = OrderedDict()
        self._summary_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}

    @staticmethod
    def _summary_from_raw(raw: dict[str, Any], stem: str) -> dict[str, Any]:
        cfg = raw.get("config") or {}
        interview_type = cfg.get("interview_type", "behavioral")
        rubric = rubric_for(interview_type, str(cfg.get("answer_framework") or "carl"))
        return {
            "id": raw.get("id", stem),
            "created_at_iso": raw.get("created_at_iso", ""),
            "role_title": raw.get("role_title", "Practice Interview"),
            "company": raw.get("company", ""),
            "interview_type": interview_type,
            "target_level": normalize_level(cfg.get("target_level")) or DEFAULT_LEVEL,
            "phase": raw.get("phase", "active"),
            "question_count": len(raw.get("questions") or []),
            "turn_count": len(raw.get("turns") or []),
            "has_debrief": bool(raw.get("debrief")),
            **PracticeSessionStore._history_extras(
                raw.get("questions") or [], raw.get("retry_history"), raw.get("debrief"), scored=rubric.coached
            ),
        }

    @staticmethod
    def _history_extras(
        questions: list[Any],
        retry_history: list[Any] | None = None,
        debrief: Any = None,
        *,
        scored: bool = True,
    ) -> dict[str, Any]:
        """Small, privacy-safe facts that let the history rail tell sessions apart.

        A session with no answer structure (``scored=False``) has no score.

        ``score`` is the same number the debrief headline shows (see ``aggregateSignals`` /
        ``effectiveQuestionSignals`` in practice_studio.js): each signal is averaged across
        the scored questions, and once a retry has been scored its ``carl_after`` stands in
        for the drilled question — so Home and the Debrief can never disagree.
        """
        dims = ("context", "action", "result", "learning")
        weight = {"strong": 1.0, "partial": 0.5}
        items = [q if isinstance(q, dict) else q.model_dump() for q in questions]

        focus_title = ""
        answered = 0
        for item in items:
            if not focus_title:
                focus_title = str(item.get("title") or item.get("competency") or "").strip()[:80]
            if item.get("status") == "completed":
                answered += 1

        def points(signals: Any) -> float:
            if not isinstance(signals, dict):
                return 0.0
            return sum(weight.get(signals.get(dim), 0.0) for dim in dims)

        review_carl_by_qid: dict[str, dict[str, Any]] = {}
        if isinstance(debrief, dict):
            for rev in debrief.get("question_reviews") or []:
                if isinstance(rev, dict) and rev.get("question_id") and isinstance(rev.get("carl"), dict):
                    qid = str(rev["question_id"])
                    if qid not in review_carl_by_qid:
                        review_carl_by_qid[qid] = rev["carl"]

        rows: list[Any] = [
            review_carl_by_qid.get(str(item.get("id") or ""), item.get("carl_signals"))
            for item in items
        ]

        retry_after = None
        retry_qid: str | None = None
        for attempt in reversed(retry_history or []):
            feedback = attempt.get("feedback") if isinstance(attempt, dict) else None
            after = feedback.get("carl_after") if isinstance(feedback, dict) else None
            if isinstance(after, dict):
                retry_after = after
                raw_rqid = attempt.get("question_id") if isinstance(attempt, dict) else None
                retry_qid = str(raw_rqid).strip() if raw_rqid else None
                break
        if retry_after is not None and rows:
            drill = debrief.get("retry_drill") if isinstance(debrief, dict) else None
            drill_qid = drill.get("question_id") if isinstance(drill, dict) else None
            target_qid = retry_qid or (str(drill_qid).strip() if drill_qid else None)
            drill_idx = next(
                (i for i, item in enumerate(items) if target_qid and item.get("id") == target_qid),
                # No explicit target: the drill reworks the weakest answer (first one on ties).
                min(range(len(rows)), key=lambda i: points(rows[i])),
            )
            rows[drill_idx] = retry_after

        graded = [row for row in rows if isinstance(row, dict)]
        score: float | None = None
        if scored and graded:
            score = 0.0
            for dim in dims:
                avg = sum(weight.get(row.get(dim), 0.0) for row in graded) / len(graded)
                score += 1.0 if avg >= 0.75 else 0.5 if avg >= 0.25 else 0.0
        return {"focus_title": focus_title, "answered_count": answered, "score": score}

    @staticmethod
    def _summary_from_session(session: PracticeSession) -> dict[str, Any]:
        return {
            "id": session.id,
            "created_at_iso": session.created_at_iso,
            "role_title": session.role_title,
            "company": session.company,
            "interview_type": session.config.interview_type,
            "target_level": session.config.target_level,
            "phase": session.phase,
            "question_count": len(session.questions),
            "turn_count": len(session.turns),
            "has_debrief": bool(session.debrief),
            **PracticeSessionStore._history_extras(
                list(session.questions),
                session.retry_history,
                session.debrief,
                scored=session.config.rubric.coached,
            ),
        }

    def _safe_path(self, session_id: str) -> Path:
        if not re.match(r"^[A-Za-z0-9_\-]{4,64}$", session_id or ""):
            raise ValueError("Invalid practice session ID.")
        return self.practice_dir / f"{session_id}.json"

    def _put_cache_locked(self, session: PracticeSession) -> None:
        self._cache[session.id] = session.model_copy(deep=True)
        self._cache.move_to_end(session.id)
        while len(self._cache) > MAX_CACHED_PRACTICE_SESSIONS:
            self._cache.popitem(last=False)

    def save(self, session: PracticeSession) -> PracticeSession:
        snapshot = session.model_copy(deep=True)
        snapshot.updated_at_ms = int(time.time() * 1000)
        path = self._safe_path(snapshot.id)
        payload = snapshot.model_dump_json(indent=2).encode("utf-8")
        with self._lock:
            self.practice_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.practice_dir / f".{snapshot.id}.{secrets.token_hex(6)}.tmp"
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp_path), str(path))
            except Exception:
                try:
                    os.unlink(str(tmp_path))
                except OSError:
                    pass
                raise
            self._put_cache_locked(snapshot)
            try:
                st = os.stat(path)
                self._summary_cache[path.name] = (
                    st.st_mtime_ns,
                    st.st_size,
                    self._summary_from_session(snapshot),
                )
            except OSError:
                self._summary_cache.pop(path.name, None)
        session.updated_at_ms = snapshot.updated_at_ms
        return snapshot.model_copy(deep=True)

    def get(self, session_id: str) -> PracticeSession | None:
        try:
            path = self._safe_path(session_id)
        except ValueError:
            return None
        with self._lock:
            if session_id in self._cache:
                self._cache.move_to_end(session_id)
                return self._cache[session_id].model_copy(deep=True)
            if not path.is_file():
                return None
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                sess = PracticeSession.model_validate(raw)
                self._put_cache_locked(sess)
                return sess.model_copy(deep=True)
            except Exception:
                return None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with self._lock:
            if not self.practice_dir.is_dir():
                self._summary_cache.clear()
                return []
            entries: list[tuple[int, int, str, str]] = []
            live_names: set[str] = set()
            try:
                with os.scandir(self.practice_dir) as it:
                    for entry in it:
                        name = entry.name
                        if not name.endswith(".json") or name.startswith("."):
                            continue
                        try:
                            st = entry.stat()
                        except OSError:
                            continue
                        if not entry.is_file():
                            continue
                        live_names.add(name)
                        entries.append((st.st_mtime_ns, st.st_size, name, entry.path))
            except OSError:
                return []

            if len(self._summary_cache) > len(live_names):
                for stale in list(self._summary_cache.keys()):
                    if stale not in live_names:
                        self._summary_cache.pop(stale, None)

            entries.sort(key=lambda row: row[0], reverse=True)
            for mtime_ns, size, name, full_path in entries[:limit]:
                cached = self._summary_cache.get(name)
                if cached is not None and cached[0] == mtime_ns and cached[1] == size:
                    items.append(dict(cached[2]))
                    continue
                try:
                    raw = json.loads(Path(full_path).read_text(encoding="utf-8"))
                    summary = self._summary_from_raw(raw, name[:-5])
                    self._summary_cache[name] = (mtime_ns, size, summary)
                    items.append(dict(summary))
                except Exception:
                    self._summary_cache.pop(name, None)
                    continue
        return items
