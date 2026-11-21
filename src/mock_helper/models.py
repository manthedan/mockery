from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

SpeakerRole = Literal["interviewer", "candidate"]
CaptureMode = Literal["remote_meet", "in_person"]
InterviewType = Literal["behavioral", "system_design", "coding", "general"]
HireSignal = Literal["strong_no_hire", "lean_no_hire", "lean_hire", "strong_hire"]
Confidence = Literal["high", "medium", "low"]


LLM_PRESETS: dict[str, dict[str, str]] = {
    "agy": {
        "label": "Local agy CLI (Zero API Key)",
        "base_url": "cli://agy",
        "fast_model": "gemini-4-hydrogen:low",
        "calibration_model": "gemini-4-hydrogen:high",
    },
    "pi": {
        "label": "Local pi CLI (Zero API Key)",
        "base_url": "cli://pi",
        "fast_model": "openai-codex/gpt-6-luna:low",
        "calibration_model": "openai-codex/gpt-6.2-sol:high",
    },
    "deepseek": {
        "label": "DeepSeek (V4.1 Flash + Reasoner)",
        "base_url": "https://api.deepseek.com/v1",
        "fast_model": "deepseek-flash",
        "calibration_model": "deepseek-reasoner",
    },
    "gemini": {
        "label": "Google Gemini (Flash + Pro)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "fast_model": "gemini-2.5-flash",
        "calibration_model": "gemini-2.5-pro",
    },
    "openrouter": {
        "label": "OpenRouter (Any Model)",
        "base_url": "https://openrouter.ai/api/v1",
        "fast_model": "deepseek/deepseek-flash",
        "calibration_model": "deepseek/deepseek-r1",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "fast_model": "gpt-4.1-mini",
        "calibration_model": "gpt-4.1",
    },
    "ollama": {
        "label": "Local Ollama (Offline)",
        "base_url": "http://localhost:11434/v1",
        "fast_model": "qwen2.5:7b",
        "calibration_model": "qwen2.5:14b",
    },
}

PARAKEET_CHECKPOINTS: dict[str, str] = {
    "mlx-community/parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3 (Multilingual default)",
    "mlx-community/parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B v2 (English-focused — A/B test)",
}


def short_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{uid}" if prefix else uid


class Utterance(BaseModel):
    """
    Canonical upsertable transcript event.
    - `id` stays stable from VAD speech onset through draft partials (`final=False`) to `final=True`.
    - `rev` increments monotonically on every draft update AND on any delayed Phase 2 diarization
      speaker-label correction (`speaker` flip at `rev + 1`).
    """

    id: str = Field(default_factory=lambda: short_id("utt"))
    rev: int = 1
    final: bool = True
    speaker: SpeakerRole
    text: str
    start_ms: int
    end_ms: int


class ProbeSuggestion(BaseModel):
    angle: str
    question: str
    why: str


class DiveDeeperResponse(BaseModel):
    qa_id: str
    quick_read: str
    probes: list[ProbeSuggestion]
    generated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    model_used: str = ""
    based_on_utt_count: int = 0
    based_on_rev_sum: int = 0


class QAGroup(BaseModel):
    """
    Derived topic/question card pointing into `InterviewSession.utterances` by ID.
    Splitting, merging, or full-context regrouping never mutates the underlying utterance log.
    """

    id: str = Field(default_factory=lambda: short_id("qa"))
    index: int
    card_kind: Literal["question", "phase"] = "question"
    title: str
    main_question: str = ""
    utterance_ids: list[str] = Field(default_factory=list)
    dive_deeper_history: list[DiveDeeperResponse] = Field(default_factory=list)
    speculative_cache: DiveDeeperResponse | None = None
    notes: str = ""
    started_ms: int = 0
    updated_ms: int = 0

    def badge_prefix(self) -> str:
        return "P" if self.card_kind == "phase" else "Q"


class CardTimingStat(BaseModel):
    qa_id: str
    badge: str
    title: str
    span_seconds: float
    interviewer_speak_seconds: float
    candidate_speak_seconds: float
    interviewer_words: int
    candidate_words: int


class SessionMetrics(BaseModel):
    duration_seconds: float = 0.0
    interviewer_speak_seconds: float = 0.0
    candidate_speak_seconds: float = 0.0
    total_utterances: int = 0
    qa_count: int = 0
    interviewer_words: int = 0
    candidate_words: int = 0
    interviewer_talk_pct: float = 0.0
    candidate_talk_pct: float = 0.0
    interviewer_time_pct: float = 0.0
    candidate_time_pct: float = 0.0
    avg_candidate_words_per_qa: float = 0.0
    probes_requested: int = 0
    echo_bleed_dropped: int = 0
    per_card: list[CardTimingStat] = Field(default_factory=list)


class InterviewerSelfRating(BaseModel):
    """
    Recorded by the interviewer BEFORE revealing the AI calibration report,
    so the tool trains your calibration muscle instead of replacing it.
    """

    hire_signal: HireSignal
    perceived_level: str = ""
    gut_rationale: str = ""
    biggest_concern: str = ""
    submitted_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class TurnSegment(BaseModel):
    """Single-speaker sub-turn emitted by LLM turn triage (supports splitting rapid single-mic handoffs)."""

    speaker: SpeakerRole
    text: str


class TurnTriagePlan(BaseModel):
    """
    Typed operation plan emitted by fast LLM triage (inspired by scratch/triage-lab).
    High-confidence ops apply automatically; medium/low confidence ops queue in `pending_reviews`.
    """

    utterance_id: str
    segments: list[TurnSegment] = Field(default_factory=list)
    card_action: Literal["keep_in_current", "open_new_card", "ignore_backchannel"] = "keep_in_current"
    suggested_title: str = ""
    confidence: Confidence = "high"
    rationale: str = ""


class PendingReviewItem(BaseModel):
    """Confidence-gated review chip surfaced inline when turn triage is medium/low confidence."""

    id: str = Field(default_factory=lambda: short_id("rev"))
    utterance_id: str
    qa_id: str | None = None
    kind: Literal["flip_speaker", "split_card", "split_mixed_turn"]
    suggested_speaker: SpeakerRole | None = None
    suggested_segments: list[TurnSegment] = Field(default_factory=list)
    suggested_title: str = ""
    rationale: str = ""
    confidence: Confidence = "medium"


class SessionConfig(BaseModel):
    role_title: str = "Senior Software Engineer"
    interview_type: InterviewType = "behavioral"
    mode: CaptureMode = "remote_meet"
    rubric_notes: str = ""
    # Legacy field: consent is handled verbally by the interviewer and no longer gates anything.
    # Kept (defaulting to True) only so older saved `sessions/*.json` files still deserialize.
    consent_confirmed: bool = True
    save_raw_audio: bool = False
    auto_diarize: bool = True


class InterviewSession(BaseModel):
    id: str = Field(
        default_factory=lambda: time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    )
    created_at_iso: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    started_epoch_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    ended_epoch_ms: int | None = None
    elapsed_capture_ms: int = 0
    config: SessionConfig = Field(default_factory=SessionConfig)

    # Canonical source-of-truth log (upsertable by `Utterance.id` + `Utterance.rev`)
    utterances: list[Utterance] = Field(default_factory=list)

    # Derived views over `utterances`
    warmup_utterance_ids: list[str] = Field(default_factory=list)
    qa_groups: list[QAGroup] = Field(default_factory=list)
    pending_reviews: list[PendingReviewItem] = Field(default_factory=list)

    echo_bleed_dropped: int = 0
    self_rating: InterviewerSelfRating | None = None
    calibration_markdown: str | None = None

    def utterance_map(self) -> dict[str, Utterance]:
        return {u.id: u for u in self.utterances}

    def upsert_utterance_with_status(self, incoming: Utterance) -> tuple[Utterance, bool, bool]:
        """
        Upserts an utterance by `id`. Returns `(stored_utterance, is_brand_new, applied)`.
        Ignores stale out-of-order revisions (`incoming.rev < existing.rev`) with `applied=False`.
        """
        for idx, existing in enumerate(self.utterances):
            if existing.id == incoming.id:
                if incoming.rev >= existing.rev:
                    self.utterances[idx] = incoming
                    return incoming, False, True
                return existing, False, False

        self.utterances.append(incoming)
        self.utterances.sort(key=lambda u: (u.start_ms, u.id))
        return incoming, True, True

    def upsert_utterance(self, incoming: Utterance) -> tuple[Utterance, bool]:
        """
        Upserts an utterance by `id`. Returns `(stored_utterance, is_brand_new)`.
        Ignores stale out-of-order revisions (`incoming.rev < existing.rev`).
        """
        stored, is_new, _applied = self.upsert_utterance_with_status(incoming)
        return stored, is_new

    def remove_utterance(self, utt_id: str) -> bool:
        """
        Removes an utterance and its derived card/warmup references (used when an earlier
        mic draft revision is subsequently identified as cross-channel echo bleed or empty final).
        """
        before = len(self.utterances)
        self.utterances = [u for u in self.utterances if u.id != utt_id]
        removed = len(self.utterances) < before
        if utt_id in self.warmup_utterance_ids:
            self.warmup_utterance_ids = [uid for uid in self.warmup_utterance_ids if uid != utt_id]
            removed = True
        self.pending_reviews = [r for r in self.pending_reviews if r.utterance_id != utt_id]
        umap = self.utterance_map()
        non_empty_cards: list[QAGroup] = []
        for qa in self.qa_groups:
            if utt_id in qa.utterance_ids:
                qa.utterance_ids = [uid for uid in qa.utterance_ids if uid != utt_id]
                qa.speculative_cache = None
                removed = True
                remaining = [umap[uid] for uid in qa.utterance_ids if uid in umap]
                if remaining:
                    qa.started_ms = min(u.start_ms for u in remaining)
                    qa.updated_ms = max(u.end_ms for u in remaining)
                    first_int = next((u for u in remaining if u.speaker == "interviewer"), None)
                    qa.main_question = first_int.text if first_int else ""
            # Drop auto-created card if its sole utterance was invalidated echo bleed and it has no notes
            if not qa.utterance_ids and not qa.notes.strip() and not qa.dive_deeper_history:
                continue
            non_empty_cards.append(qa)
        for idx, qa in enumerate(non_empty_cards, start=1):
            qa.index = idx
        self.qa_groups = non_empty_cards
        return removed

    def utterances_for_card(
        self,
        qa: QAGroup,
        *,
        include_drafts: bool = True,
        umap: dict[str, Utterance] | None = None,
    ) -> list[Utterance]:
        lookup = umap if umap is not None else self.utterance_map()
        out: list[Utterance] = []
        for uid in qa.utterance_ids:
            u = lookup.get(uid)
            if u is not None and (include_drafts or u.final):
                out.append(u)
        return out

    def card_revision_sum(self, qa: QAGroup) -> int:
        return sum(u.rev for u in self.utterances_for_card(qa, include_drafts=True))

    def warmup_utterances(self, *, umap: dict[str, Utterance] | None = None) -> list[Utterance]:
        lookup = umap if umap is not None else self.utterance_map()
        return [lookup[uid] for uid in self.warmup_utterance_ids if uid in lookup]

    def card_word_counts(self, qa: QAGroup) -> dict[SpeakerRole, int]:
        counts: dict[SpeakerRole, int] = {"interviewer": 0, "candidate": 0}
        for u in self.utterances_for_card(qa):
            counts[u.speaker] += len(u.text.split())
        return counts

    def compute_metrics(self) -> SessionMetrics:
        iw = 0
        cw = 0
        i_ms = 0
        c_ms = 0
        min_start: int | None = None
        max_end: int | None = None
        umap: dict[str, Utterance] = {}

        for u in self.utterances:
            umap[u.id] = u
            words = len(u.text.split())
            dur = max(0, u.end_ms - u.start_ms)
            if u.speaker == "interviewer":
                iw += words
                i_ms += dur
            else:
                cw += words
                c_ms += dur
            if min_start is None or u.start_ms < min_start:
                min_start = u.start_ms
            if max_end is None or u.end_ms > max_end:
                max_end = u.end_ms

        total_w = iw + cw
        total_speak_ms = i_ms + c_ms
        dur_s = max(0.0, (max_end - min_start) / 1000.0) if (min_start is not None and max_end is not None) else 0.0

        per_card: list[CardTimingStat] = []
        probes_requested = 0
        for qa in self.qa_groups:
            probes_requested += len(qa.dive_deeper_history)
            utts = self.utterances_for_card(qa, umap=umap)
            if utts:
                c_min = min(u.start_ms for u in utts)
                c_max = max(u.end_ms for u in utts)
                span_s = max(0.0, (c_max - c_min) / 1000.0)
            else:
                span_s = 0.0
            card_i_ms = 0
            card_c_ms = 0
            card_iw = 0
            card_cw = 0
            for u in utts:
                w = len(u.text.split())
                d = max(0, u.end_ms - u.start_ms)
                if u.speaker == "interviewer":
                    card_i_ms += d
                    card_iw += w
                else:
                    card_c_ms += d
                    card_cw += w
            per_card.append(
                CardTimingStat(
                    qa_id=qa.id,
                    badge=f"{qa.badge_prefix()}{qa.index}",
                    title=qa.title,
                    span_seconds=round(span_s, 1),
                    interviewer_speak_seconds=round(card_i_ms / 1000.0, 1),
                    candidate_speak_seconds=round(card_c_ms / 1000.0, 1),
                    interviewer_words=card_iw,
                    candidate_words=card_cw,
                )
            )

        qa_n = len(self.qa_groups)
        return SessionMetrics(
            duration_seconds=round(dur_s, 1),
            interviewer_speak_seconds=round(i_ms / 1000.0, 1),
            candidate_speak_seconds=round(c_ms / 1000.0, 1),
            total_utterances=len(self.utterances),
            qa_count=qa_n,
            interviewer_words=iw,
            candidate_words=cw,
            interviewer_talk_pct=round(iw / total_w * 100.0, 1) if total_w else 0.0,
            candidate_talk_pct=round(cw / total_w * 100.0, 1) if total_w else 0.0,
            interviewer_time_pct=round(i_ms / total_speak_ms * 100.0, 1) if total_speak_ms else 0.0,
            candidate_time_pct=round(c_ms / total_speak_ms * 100.0, 1) if total_speak_ms else 0.0,
            avg_candidate_words_per_qa=round(cw / qa_n, 1) if qa_n else 0.0,
            probes_requested=probes_requested,
            echo_bleed_dropped=self.echo_bleed_dropped,
            per_card=per_card,
        )

    def to_markdown(
        self,
        *,
        audience: Literal["interviewer", "candidate"] = "interviewer",
        metrics: SessionMetrics | None = None,
    ) -> str:
        metrics = metrics if metrics is not None else self.compute_metrics()
        umap = self.utterance_map()
        mins = int(metrics.duration_seconds // 60)
        secs = int(metrics.duration_seconds % 60)
        heading = (
            f"# Mock Interview Debrief (Interviewer Copy) — {self.config.role_title}"
            if audience == "interviewer"
            else f"# Mock Interview Transcript & Notes (Candidate Copy) — {self.config.role_title}"
        )
        lines = [
            heading,
            "",
            f"- **Session ID:** `{self.id}`",
            f"- **Date:** {self.created_at_iso}",
            f"- **Focus:** `{self.config.interview_type}` | **Span:** {mins}m {secs:02d}s",
            f"- **Speaking Time Ratio:** Interviewer {metrics.interviewer_time_pct}% ({metrics.interviewer_speak_seconds}s) vs. Candidate {metrics.candidate_time_pct}% ({metrics.candidate_speak_seconds}s)",
            f"- **Word Ratio:** Interviewer {metrics.interviewer_talk_pct}% ({metrics.interviewer_words}w) vs. Candidate {metrics.candidate_talk_pct}% ({metrics.candidate_words}w)",
            "",
        ]

        if audience == "interviewer" and self.self_rating is not None:
            lines.extend(
                [
                    "## Interviewer Blind Pre-Calibration",
                    f"- **Initial Signal:** `{self.self_rating.hire_signal}`",
                    f"- **Perceived Level:** {self.self_rating.perceived_level or '—'}",
                    f"- **Initial Rationale:** {self.self_rating.gut_rationale or '—'}",
                    f"- **Key Concern:** {self.self_rating.biggest_concern or '—'}",
                    "",
                ]
            )

        warm = self.warmup_utterances(umap=umap)
        if warm:
            lines.append("## Warm-up / Intro")
            for u in warm:
                lines.append(f"- **[{fmt_ms(u.start_ms)} · `{u.id}`] {u.speaker.title()}:** {u.text}")
            lines.append("")

        lines.append("## Organized Interview Transcript")
        lines.append("")
        for qa, stat in zip(self.qa_groups, metrics.per_card):
            lines.append(
                f"### {stat.badge}: {qa.title} *(Span: {stat.span_seconds}s | Candidate: {stat.candidate_words}w)*"
            )
            for u in self.utterances_for_card(qa, umap=umap):
                role = "Interviewer" if u.speaker == "interviewer" else "Candidate"
                draft_tag = " *(in-progress)*" if not u.final else ""
                lines.append(f"- **[{fmt_ms(u.start_ms)} · `{u.id}`] {role}:** {u.text}{draft_tag}")
            if audience == "interviewer":
                if qa.notes.strip():
                    lines.append(f"\n> **Interviewer Note:** {qa.notes.strip()}")
                if qa.dive_deeper_history:
                    latest = qa.dive_deeper_history[-1]
                    lines.append("\n**Suggested Dive-Deeper Probes:**")
                    for p in latest.probes:
                        lines.append(f"  - *[{p.angle}]* {p.question} *(Why: {p.why})*")
            lines.append("")

        if audience == "interviewer" and self.calibration_markdown:
            lines.extend(
                ["---", "", "## Calibration & Interviewer Coaching Report", "", self.calibration_markdown, ""]
            )

        return "\n".join(lines)


def fmt_ms(ms: int) -> str:
    total_s = max(0, ms // 1000)
    m, s = divmod(total_s, 60)
    return f"{m:02d}:{s:02d}"
