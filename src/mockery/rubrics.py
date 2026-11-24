"""The four coaching signals a Practice answer is graded on, per interview format and answer framework.

Every rubric fills the same four storage slots (the keys of ``carl_signals``), in display order.
What each slot means, and what a person reads, belongs to the rubric: CARL and STAR for
behavioral rounds, a plain-answer rubric for general questions, and the studio rubrics for
system design and coding. ``NO_STRUCTURE`` grades nothing and shows no meter: the candidate
is coached on substance alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AnswerFramework = Literal["carl", "star", "general", "none"]
SIGNAL_SLOTS: tuple[str, ...] = ("context", "action", "result", "learning")
SIGNAL_LEVELS: tuple[str, ...] = ("missing", "partial", "strong")


@dataclass(frozen=True)
class Signal:
    slot: str
    key: str
    """The name the grader uses for this signal in its JSON reply."""
    letter: str
    label: str
    nudge: str = ""
    """Offline coach tip when this signal has not landed yet; empty when it is never the tip."""
    earned_praise: str = ""
    """Said before the tip only when the signal just before this one did land."""
    nudge_tone: Literal["tip", "encourage"] = "tip"


@dataclass(frozen=True)
class Rubric:
    id: str
    name: str
    meter_title: str
    summary: str
    signals: tuple[Signal, Signal, Signal, Signal]
    grader: Literal["carl", "star", "general", "system_design", "coding"]
    """Which grading vocabulary in ``prompts.py`` describes these signals to the model."""
    nudge_order: tuple[str, ...]
    """Slots checked, in turn, for the live coach tip."""
    drill_order: tuple[str, ...]
    """Slots checked, in turn, for the one thing most worth a retry."""
    all_landed: str
    nothing_yet: str
    """Offline coach line when the answer was too short to coach."""
    ready_tip: str
    reframe_tip: str
    primer: str
    """One line on how the rubric works, shown at the top of an offline debrief."""
    default_focus: str
    """Coaching focus for a question that arrives without one."""
    focus_label: str
    """How a question's coaching focus is introduced to the interviewer model."""
    debrief_name: str
    offline_hint: str = ""
    """Hint used when no model is connected; ``{focus}`` is the question's coaching focus.
    The studio rounds build theirs from the problem instead."""
    coached: bool = True
    """False for ``NO_STRUCTURE``: signals are neither stored nor shown, and no framework is taught."""
    named_framework: bool = False
    """True when the rubric is a method people know by name (CARL, STAR) and coaching may say so."""
    legacy_slot_fallback: bool = False
    """True for studio rubrics whose older sessions or models may still emit legacy slot names."""

    def signal(self, slot: str) -> Signal:
        return next(s for s in self.signals if s.slot == slot)

    @property
    def signal_keys(self) -> tuple[str, ...]:
        return tuple(s.key for s in self.signals)

    def public_dict(self) -> dict[str, Any]:
        return {
            "coached": self.coached,
            "id": self.id,
            "name": self.name,
            "meter_title": self.meter_title,
            "summary": self.summary,
            "ready_tip": self.ready_tip,
            "signals": [{"slot": s.slot, "letter": s.letter, "label": s.label} for s in self.signals],
        }

    def signals_from_grader(self, raw: Any) -> dict[str, str]:
        """Map the grader's reply onto storage slots, keeping only recognised levels."""
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for signal in self.signals:
            raw_val = raw.get(signal.key)
            if raw_val is None and self.legacy_slot_fallback:
                raw_val = raw.get(signal.slot)
            level = str(raw_val if raw_val is not None else "").strip().lower()
            if level in SIGNAL_LEVELS:
                out[signal.slot] = level
        return out

    def preceding(self, slot: str) -> Signal | None:
        index = SIGNAL_SLOTS.index(slot)
        return self.signals[index - 1] if index > 0 else None

    def drill_slot(self, signals: dict[str, str]) -> str:
        """The first signal that is not yet strong, in drill order; the first of all when every one is."""
        for slot in self.drill_order:
            if signals.get(slot) != "strong":
                return slot
        return self.drill_order[0]


CARL = Rubric(
    id="carl",
    name="CARL",
    meter_title="Live CARL Meter",
    summary="Context → Action (what you drove) → Result (metric or outcome) → Learning.",
    signals=(
        Signal("context", "context", "C", "Context"),
        Signal(
            "action",
            "action",
            "A",
            "Action",
            nudge="Now zoom into what YOU specifically decided and built.",
            earned_praise="Good setup on the context!",
        ),
        Signal(
            "result",
            "result",
            "R",
            "Result",
            nudge="Next, anchor it with a concrete result or number (time saved, revenue, adoption, quality).",
            earned_praise="Strong ownership on the actions.",
        ),
        Signal(
            "learning",
            "learning",
            "L",
            "Learning",
            nudge=(
                "Adding a 1-sentence Learning ('what I'd do differently or kept doing since') "
                "makes strong answers shine."
            ),
            earned_praise="Solid C-A-R foundation!",
            nudge_tone="encourage",
        ),
    ),
    grader="carl",
    nudge_order=("action", "result", "learning"),
    drill_order=("result", "learning", "action", "context"),
    all_landed="Great CARL structure — clear context, personal ownership, measurable impact, and genuine reflection!",
    nothing_yet="There isn't an answer to coach yet. Pick one real example and start with where you were and what was at stake.",
    ready_tip=(
        "Think out loud. Set the context, show what you drove, and close with a measurable result "
        "and what you learned."
    ),
    reframe_tip=(
        "Try closing with: 'As a result, [measurable outcome], and what I carried forward to future "
        "projects was [1-sentence learning].'"
    ),
    primer="CARL = Context (brief setup) → Action (what YOU decided & drove) → Result (measurable outcome) → Learning (what changed afterward).",
    default_focus="Clear Context, your personal Actions, measurable Result, and Learning.",
    offline_hint="Start with a 2-sentence Context, then walk through 2 specific Actions YOU drove (focus on: {focus}).",
    focus_label="CARL / Rubric Focus",
    debrief_name="CARL Coaching Debrief",
    named_framework=True,
)

STAR = Rubric(
    id="star",
    name="STAR",
    meter_title="Live STAR Meter",
    summary="Situation → Task (what was yours to do) → Action (what you did) → Result (metric or outcome).",
    signals=(
        Signal("context", "situation", "S", "Situation"),
        Signal(
            "action",
            "task",
            "T",
            "Task",
            nudge="Now say what was yours to do — the goal or responsibility you personally owned.",
            earned_praise="Good scene-setting!",
        ),
        Signal(
            "result",
            "action",
            "A",
            "Action",
            nudge="Now walk through what YOU specifically did, step by step.",
            earned_praise="The task is clear.",
        ),
        Signal(
            "learning",
            "result",
            "R",
            "Result",
            nudge="Close with a concrete result or number (time saved, revenue, adoption, quality).",
            earned_praise="Strong ownership on the actions.",
        ),
    ),
    grader="star",
    nudge_order=("action", "result", "learning"),
    drill_order=("learning", "result", "action", "context"),
    all_landed="Great STAR structure — a clear situation, the task you owned, your own actions, and a measurable result!",
    nothing_yet="There isn't an answer to coach yet. Pick one real example and start with the situation and what was at stake.",
    ready_tip=(
        "Think out loud. Set the situation, name the task that was yours, walk through what you did, "
        "and close with the result."
    ),
    reframe_tip=(
        "Try this shape: 'My job was to [task]. So I [two specific actions]. As a result, [measurable outcome].'"
    ),
    primer="STAR = Situation (brief setup) → Task (what was yours to do) → Action (what YOU did) → Result (measurable outcome).",
    default_focus="A clear Situation, the Task that was yours, your own Actions, and a measurable Result.",
    offline_hint="Set the Situation in two sentences, say what was yours to do, then walk through 2 specific Actions YOU took (focus on: {focus}).",
    focus_label="STAR / Rubric Focus",
    debrief_name="STAR Coaching Debrief",
    named_framework=True,
)

GENERAL = Rubric(
    id="general",
    name="General",
    meter_title="Live Answer Signals",
    summary=(
        "Direct (answer the question first) → Specific (a concrete reason or example) → "
        "Fit (tie it to this role) → Concise (stop when you have answered)."
    ),
    signals=(
        Signal(
            "context",
            "direct",
            "D",
            "Direct",
            nudge="Lead with your answer in one sentence, then explain it.",
        ),
        Signal(
            "action",
            "specific",
            "S",
            "Specific",
            nudge="Now back it with one concrete reason or a short example.",
            earned_praise="Good, clear answer.",
        ),
        Signal(
            "result",
            "fit",
            "F",
            "Fit",
            nudge="Tie it to this role or team so it is clearly about them, not any job.",
            earned_praise="Nice and specific.",
        ),
        Signal(
            "learning",
            "concise",
            "C",
            "Concise",
            nudge="You have made the point — a shorter version would land harder.",
        ),
    ),
    grader="general",
    nudge_order=("context", "action", "result", "learning"),
    drill_order=("context", "action", "result", "learning"),
    all_landed="That was direct, specific, clearly about this role, and tight. Exactly what these questions call for.",
    nothing_yet="There isn't an answer to coach yet. Say your answer in one plain sentence, then give one reason for it.",
    ready_tip="Answer the question in your first sentence, give one concrete reason, and tie it to this role.",
    reframe_tip="Try this shape: '[Your answer in one sentence]. For example, [one concrete detail]. That matters here because [link to the role].'",
    primer="Answer the question first, back it with one concrete reason or example, and tie it to this role.",
    default_focus="A direct answer, one concrete reason or example, and a clear link to this role.",
    offline_hint="Answer in one sentence first, then give one concrete reason (focus on: {focus}).",
    focus_label="Answer Focus",
    debrief_name="Coaching Debrief",
    legacy_slot_fallback=True,
)

SYSTEM_DESIGN = Rubric(
    id="system_design",
    name="Design",
    meter_title="Live Design Signals",
    summary=(
        "Requirements (scope & constraints) → Architecture (components & data flow) → "
        "Scale (bottlenecks & numbers) → Tradeoffs (what you chose and why)."
    ),
    signals=(
        Signal("context", "requirements", "R", "Requirements"),
        Signal("action", "architecture", "A", "Architecture"),
        Signal("result", "scale", "S", "Scale"),
        Signal("learning", "tradeoffs", "T", "Tradeoffs"),
    ),
    grader="system_design",
    nudge_order=(),
    drill_order=("result", "learning", "action", "context"),
    all_landed="Requirements, architecture, scale and tradeoffs all came through clearly.",
    nothing_yet="There isn't much to go on yet. Start by saying the requirements and the scale you are designing for.",
    ready_tip=(
        "Think out loud. Pin down requirements first, then sketch the architecture and name your tradeoffs."
    ),
    reframe_tip="Try closing with: 'The main tradeoff I accepted was [X] in exchange for [Y], because [constraint].'",
    primer="Requirements → Architecture → Scale → Tradeoffs: scope it, sketch it, stress it, then defend your choices.",
    default_focus="Clear requirements, a sound architecture, scaling limits, and the tradeoffs you chose.",
    focus_label="Design Focus",
    debrief_name="System Design Coaching Debrief",
    legacy_slot_fallback=True,
)

CODING = Rubric(
    id="coding",
    name="Coding",
    meter_title="Live Coding Signals",
    summary=(
        "Approach (plan before code) → Correctness (working solution) → "
        "Complexity (time & space) → Edge cases (tested boundaries)."
    ),
    signals=(
        Signal("context", "approach", "A", "Approach"),
        Signal("action", "correctness", "C", "Correctness"),
        Signal("result", "complexity", "O", "Complexity"),
        Signal("learning", "edge_cases", "E", "Edge cases"),
    ),
    grader="coding",
    nudge_order=(),
    drill_order=("result", "learning", "action", "context"),
    all_landed="Approach, correctness, complexity and edge cases all came through clearly.",
    nothing_yet="There isn't much to go on yet. Say your approach out loud before you write any code.",
    ready_tip="Think out loud. State your approach and its complexity before you write the first line.",
    reframe_tip="Try closing with: 'This runs in [time] and [space], and the edge case I checked was [X].'",
    primer="Approach → Correctness → Complexity → Edge cases: plan it, make it work, size it, then try to break it.",
    default_focus="A clear approach, a working solution, its complexity, and the edge cases you tested.",
    focus_label="Coding Focus",
    debrief_name="Coding Coaching Debrief",
    legacy_slot_fallback=True,
)

NO_STRUCTURE = Rubric(
    id="none",
    name="",
    meter_title="",
    summary="",
    signals=CARL.signals,
    grader="carl",
    nudge_order=(),
    drill_order=CARL.drill_order,
    all_landed="Good — keep going, and add one concrete detail to whatever you say next.",
    nothing_yet="There isn't an answer to coach yet. Take a moment, then say the most concrete thing you can about it.",
    ready_tip="Think out loud, and answer the way you would in the room.",
    reframe_tip="Try saying the same point again in two sentences, with one concrete detail.",
    primer="Clear, specific answers in your own words are what land.",
    default_focus="A clear, specific answer in your own words.",
    offline_hint="Say the most concrete version of your answer first (focus on: {focus}).",
    focus_label="Answer Focus",
    debrief_name="Coaching Debrief",
    coached=False,
)

_BY_FRAMEWORK: dict[str, Rubric] = {"carl": CARL, "star": STAR, "general": GENERAL, "none": NO_STRUCTURE}


def framework_for(interview_type: str, requested: str) -> AnswerFramework:
    """The framework a new session records: behavioral rounds choose, general rounds use their own."""
    if interview_type == "behavioral":
        return "star" if requested == "star" else "none" if requested == "none" else "carl"
    if interview_type == "general":
        return "general"
    return "carl"


def rubric_for(interview_type: str, answer_framework: str) -> Rubric:
    """The rubric a session is graded on."""
    if interview_type == "system_design":
        return SYSTEM_DESIGN
    if interview_type == "coding":
        return CODING
    return _BY_FRAMEWORK.get(answer_framework, CARL)
