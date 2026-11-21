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

from pydantic import BaseModel, Field

from .models import short_id

PracticeInterviewType = Literal["behavioral", "system_design", "coding", "general"]
PracticeQuestionSource = Literal["bank_only", "resume_jd_tailored", "mixed"]
PracticePhase = Literal["active", "awaiting_debrief", "debriefing", "completed"]
CarlStatus = Literal["strong", "partial", "missing"]
MAX_CACHED_PRACTICE_SESSIONS = 32


class PracticeConfig(BaseModel):
    interview_type: PracticeInterviewType = "behavioral"
    question_source: PracticeQuestionSource = "mixed"
    target_level: str = "L5"
    coach_style: Literal["supportive", "balanced"] = "supportive"
    live_coach_nudges: bool = True
    voice_enabled: bool = True
    question_count: int = Field(default=4, ge=1, le=8)
    selected_question_id: str | None = None


class PracticeQuestionItem(BaseModel):
    id: str = Field(default_factory=lambda: short_id("pq"))
    bank_question_id: str | None = None
    source: Literal["bank", "tailored"] = "bank"
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


class PracticeSession(BaseModel):
    id: str = Field(default_factory=lambda: short_id("prac"))
    created_at_iso: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    role_title: str = "Senior Software Engineer"
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
                lines.append("## What You Did Well")
                lines.append("")
                for s in strengths:
                    lines.append(f"- {s}")
                lines.append("")
            q_reviews = [qr for qr in (d.get("question_reviews") or []) if isinstance(qr, dict)]
            if q_reviews:
                lines.append("## Question-by-Question CARL Coaching")
                lines.append("")
                for idx, qr in enumerate(q_reviews, 1):
                    lines.append(f"### Q{idx}: {qr.get('title') or str(qr.get('prompt', ''))[:80]}")
                    carl = qr.get("carl") if isinstance(qr.get("carl"), dict) else {}
                    lines.append(
                        f"- **CARL:** Context `{carl.get('context', '—')}` · "
                        f"Action `{carl.get('action', '—')}` · "
                        f"Result `{carl.get('result', '—')}` · "
                        f"Learning `{carl.get('learning', '—')}`"
                    )
                    if qr.get("strength"):
                        lines.append(f"- **Strength:** {qr['strength']}")
                    if qr.get("coaching_tip"):
                        lines.append(f"- **Next-Level Tip:** {qr['coaching_tip']}")
                    if qr.get("carl_reframe"):
                        lines.append(f"- **Suggested CARL Reframe:** {qr['carl_reframe']}")
                    lines.append("")
            drill = d.get("retry_drill") if isinstance(d.get("retry_drill"), dict) else {}
            if drill.get("prompt"):
                lines.append("## Recommended 45-Second Retry Drill")
                lines.append("")
                lines.append(f"- **Focus:** {drill.get('focus', 'Strengthen CARL signal')}")
                lines.append(f"- **Prompt:** {drill['prompt']}")
                lines.append("")

        if self.retry_history:
            lines.append("## Retry Drill Progress")
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

        lines.append("## Full Practice Transcript")
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


def estimate_carl_heuristic(answer_text: str, existing: dict[str, str] | None = None) -> dict[str, str]:
    """
    Fast, deterministic CARL signal estimator used as instant baseline and offline fallback.
    Never downgrades an already-strong dimension across follow-up turns for the same question.
    """
    low = (answer_text or "").lower()
    words = low.split()
    wc = len(words)

    rank = {"missing": 0, "partial": 1, "strong": 2}
    inv_rank = {0: "missing", 1: "partial", 2: "strong"}

    prev = existing or {"context": "missing", "action": "missing", "result": "missing", "learning": "missing"}

    has_context = any(
        k in low
        for k in (
            "when i", "at my", "our team", "project", "problem", "incident", "outage",
            "context", "situation", "legacy", "migrat", "quarter", "stakeholder", "deadline",
        )
    )
    has_action = any(
        k in low
        for k in (
            "i designed", "i built", "i led", "i proposed", "i refactored", "i implemented",
            "i wrote", "i decided", "i aligned", "i benchmarked", "i set up", "i profiled",
            "i debugged", "my approach", "i chose", "i worked with", "i convinced",
        )
    ) or (" i " in f" {low} " and wc >= 25)
    has_metric = bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:%(?!\w)|(?:ms|s|x|k|m|million|thousand|hours|days|weeks|users|qps|rps)\b)",
            low,
        )
    )
    has_result = has_metric or any(
        k in low
        for k in (
            "reduced", "improved", "increased", "shipped", "launched", "saved",
            "result", "outcome", "impact", "resolved", "adopted", "unblocked",
        )
    )
    has_learning = any(
        k in low
        for k in (
            "learned", "takeaway", "next time", "retrospective", "in hindsight",
            "changed how", "now i always", "lesson", "would do differently", "taught me",
        )
    )

    cur = {
        "context": "strong" if (has_context and wc >= 20) else ("partial" if has_context or wc >= 12 else "missing"),
        "action": "strong" if (has_action and wc >= 35) else ("partial" if has_action else "missing"),
        "result": "strong" if has_metric else ("partial" if has_result else "missing"),
        "learning": "strong" if has_learning else "missing",
    }
    return {
        dim: inv_rank[max(rank.get(prev.get(dim, "missing"), 0), rank.get(cur[dim], 0))]
        for dim in ("context", "action", "result", "learning")
    }


def build_heuristic_nudge(carl: dict[str, str], interview_type: str) -> dict[str, str]:
    """Warm, encouraging deterministic coach nudge when LLM is unavailable."""
    if interview_type == "system_design":
        return {
            "tone": "tip",
            "message": "Good momentum — call out one concrete bottleneck (traffic, storage, or failure mode) and how you'd mitigate it.",
        }
    if interview_type == "coding":
        return {
            "tone": "tip",
            "message": "Nice thinking out loud — state the time/space complexity and check an edge case before finalizing.",
        }
    if carl.get("action") == "missing":
        return {
            "tone": "tip",
            "message": "Good setup on the context! Now zoom into what YOU specifically decided and built.",
        }
    if carl.get("result") == "missing":
        return {
            "tone": "tip",
            "message": "Strong ownership on the actions — next, anchor it with a concrete result or metric (latency, adoption, time saved).",
        }
    if carl.get("learning") == "missing":
        return {
            "tone": "encourage",
            "message": "Solid C-A-R foundation! Adding a 1-sentence Learning ('what I'd do differently or kept doing since') makes senior answers shine.",
        }
    return {
        "tone": "encourage",
        "message": "Great CARL structure — clear context, personal ownership, measurable impact, and genuine reflection!",
    }


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
        return {
            "id": raw.get("id", stem),
            "created_at_iso": raw.get("created_at_iso", ""),
            "role_title": raw.get("role_title", "Practice Interview"),
            "company": raw.get("company", ""),
            "interview_type": cfg.get("interview_type", "behavioral"),
            "target_level": cfg.get("target_level", "L5"),
            "phase": raw.get("phase", "active"),
            "question_count": len(raw.get("questions") or []),
            "turn_count": len(raw.get("turns") or []),
            "has_debrief": bool(raw.get("debrief")),
            **PracticeSessionStore._history_extras(raw.get("questions") or []),
        }

    @staticmethod
    def _history_extras(questions: list[Any]) -> dict[str, Any]:
        """Small, privacy-safe facts that let the history rail tell sessions apart."""
        focus_title = ""
        answered = 0
        total = 0.0
        scored = 0
        for q in questions:
            item = q if isinstance(q, dict) else q.model_dump()
            if not focus_title:
                focus_title = str(item.get("title") or item.get("competency") or "").strip()[:80]
            if item.get("status") == "completed":
                answered += 1
            signals = item.get("carl_signals") or {}
            if isinstance(signals, dict) and signals:
                scored += 1
                for dim in ("context", "action", "result", "learning"):
                    val = signals.get(dim)
                    total += 1.0 if val == "strong" else 0.5 if val == "partial" else 0.0
        score = round((total / scored) * 2) / 2 if scored else None
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
            **PracticeSessionStore._history_extras(list(session.questions)),
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
