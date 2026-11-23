from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PromptTier = Literal[1, 2, 3]
PromptOutputFormat = Literal["json", "markdown", "text"]


@dataclass(frozen=True)
class PromptSpec:
    """Declarative prompt specification registered for runtime use and offline eval/tuning."""

    name: str
    description: str
    tier: PromptTier
    default_text: str
    input_keys: tuple[str, ...]
    output_format: PromptOutputFormat


def triage_diarize_rules(diarize: bool) -> str:
    if diarize:
        return (
            "1. SPEAKER DIARIZATION & RAPID HANDOFF SPLITTING (CRITICAL):\n"
            "   - The audio comes from a single shared mic/track, so `current_speaker` may be wrong AND a single VAD segment may contain BOTH speakers back-to-back without a pause.\n"
            "   - For each incoming utterance, output `segments`: 1 segment if a single person spoke, OR 2+ segments split cleanly at the exact sentence/clause where the speaker switched (`interviewer` <-> `candidate`).\n"
            "   - Interviewer cues: introduces the mock, asks primary questions or follow-ups, praises ('Cool, that makes sense...'), transitions ('moving on to the next question').\n"
            "   - Candidate cues: answers questions in first person, tells stories about past roles/projects, explains technical work.\n"
            "   - Preserve almost all words verbatim; ONLY fix unmistakable STT proper-noun/domain typos (e.g. 'for deployed' -> 'forward deployed') without paraphrasing.\n"
        )
    return (
        "1. SPEAKER LABELS ARE HARDWARE-SEPARATED:\n"
        "   - Keep `segments` as a single item with the exact `current_speaker` and original `text`.\n"
    )


def practice_tone_instruction(coach_style: str) -> str:
    if coach_style == "supportive":
        return (
            "Be warm, encouraging, collaborative, and confidence-building — like a great senior colleague "
            "who genuinely wants the candidate to succeed. Never sound harsh, robotic, or interrogative."
        )
    return "Be realistic, fair, and constructive — probing cleanly while remaining respectful and encouraging."


PracticeGrader = Literal["carl", "star", "general"]
CARL_SIGNAL_KEYS: tuple[str, ...] = ("context", "action", "result", "learning")

_SIGNAL_DEFINITIONS: dict[str, str] = {
    "carl": (
        "Context = the situation, problem, or constraints are established. "
        "Action = what the candidate specifically did, decided, or implemented. "
        "Result = outcome, impact, or verified behavior (metrics, tests passing, bug fixed). "
        "Learning = reflection, takeaway, tradeoffs, or what they would do differently."
    ),
    "star": (
        "Situation = the setting and what was at stake are established. "
        "Task = the candidate states the goal or responsibility that was personally theirs in that situation. "
        "Action = what the candidate specifically did, decided, or implemented (their own part, not the team's). "
        "Result = the outcome or impact, ideally measured."
    ),
    "general": (
        "Direct = the question that was asked is answered plainly, early in the answer. "
        "Specific = concrete reasons, facts, or a short example rather than generalities. "
        "Fit = the answer is tied to this role, team, or company rather than any job. "
        "Concise = focused, without rambling or repetition."
    ),
}
_CARRY_FORWARD_NOTES: dict[str, str] = {
    "carl": (
        " When the problem or context was clearly stated earlier (including in the prompt, workspace, "
        "or earlier turns), keep context at strong."
    ),
    "star": " When the situation was clearly stated in an earlier turn, keep situation at strong.",
    "general": "",
}
_NUDGE_EXAMPLES: dict[str, str] = {
    "carl": "e.g., quantify the Result, add a Learning, mention complexity or tests",
    "star": "e.g., state the Task that was theirs, separate their own Actions from the team's, quantify the Result",
    "general": "e.g., answer the question in the first sentence, name one concrete reason, tie it to this role",
}
_NO_FRAMEWORK_SIGNAL_DEFINITIONS = (
    'This session is not graded on structure: output every one of these values as "missing" and do not assess them.'
)
_NO_FRAMEWORK_NUDGE_EXAMPLES = (
    "e.g., expand one detail, make a point more plainly; never name or teach an answer framework such as STAR or CARL"
)
_COACH_FOCUS: dict[str, str] = {
    "carl": "teaching the CARL framework (Context, Action, Result, Learning)",
    "star": "teaching the STAR framework (Situation, Task, Action, Result)",
    "general": (
        "helping the candidate give direct, specific answers that are clearly about this role "
        "(these are motivation and fit questions, not stories, so do not impose a story framework such as STAR or CARL)"
    ),
}
_NO_FRAMEWORK_COACH_FOCUS = (
    "focused on the substance, clarity and specificity of each answer "
    "(the candidate chose no answer framework, so never name or teach one such as STAR or CARL, "
    'and output every `carl` value as "missing" without assessing it)'
)
_PRIMER_BRIEFS: dict[str, str] = {
    "carl": "1 crisp sentence reminding how CARL elevates their target level",
    "star": "1 crisp sentence reminding how STAR elevates their target level",
    "general": "1 crisp sentence on what makes these answers land at their target level",
}
_NO_FRAMEWORK_PRIMER_BRIEF = "1 crisp sentence on what will most raise their answers to their target level"
_REFRAME_BRIEFS: dict[str, str] = {
    "carl": "a concise 2-3 sentence example showing how to rephrase their weakest part into crisp CARL format",
    "star": "a concise 2-3 sentence example showing how to rephrase their weakest part into crisp STAR format",
    "general": "a concise 2-3 sentence example of a more direct, specific version of their weakest answer",
}
_NO_FRAMEWORK_REFRAME_BRIEF = "a concise 2-3 sentence example of a stronger way to say their weakest part, in their own voice"
_DRILL_FOCUS_EXAMPLES: dict[str, str] = {
    "carl": "Quantify Result & Add Learning (45s)",
    "star": "Name the Task & Quantify Result (45s)",
    "general": "Answer First & Tie It to the Role (45s)",
}
_NO_FRAMEWORK_DRILL_FOCUS_EXAMPLE = "Say It Plainly With One Concrete Detail (45s)"
_BEHAVIORAL_HINT_NUDGES: dict[str, str] = {
    "carl": "nudge structure and ownership (the situation, their specific actions, the result, what they learned)",
    "star": "nudge structure and ownership (the situation, the task that was theirs, their specific actions, the result)",
    "general": "nudge them toward a direct answer with one concrete detail",
}
_NO_FRAMEWORK_BEHAVIORAL_HINT_NUDGE = "nudge them toward the most concrete, specific version of their own answer"
_FOCUS_BRIEFS: dict[str, str] = {
    "carl": "what strong Context/Action/Result/Learning looks like here",
    "star": "what strong Situation/Task/Action/Result looks like here",
    "general": "what a direct, specific, role-relevant answer covers here",
}
_NO_FRAMEWORK_FOCUS_BRIEF = "what a strong answer covers here, without naming an answer framework"


def practice_signal_schema(signal_keys: tuple[str, ...] = CARL_SIGNAL_KEYS, *, coached: bool = True) -> str:
    """
    The JSON object the grader fills in: one level per signal, under the rubric's own names.
    A session that is not graded on structure (`coached=False`) is shown every level already set to "missing".
    """
    levels = "strong|partial|missing" if coached else "missing"
    return "{" + ", ".join(f'"{key}": "{levels}"' for key in signal_keys) + "}"


def practice_reply_rubric_args(
    grader: PracticeGrader = "carl",
    signal_keys: tuple[str, ...] = CARL_SIGNAL_KEYS,
    *,
    coached: bool = True,
) -> dict[str, str]:
    """Rubric-specific placeholders for `practice_interviewer_reply`; `coached=False` means no framework is taught."""
    return {
        "signal_schema": practice_signal_schema(signal_keys, coached=coached),
        "signal_definitions": _SIGNAL_DEFINITIONS[grader] if coached else _NO_FRAMEWORK_SIGNAL_DEFINITIONS,
        "carry_forward_note": _CARRY_FORWARD_NOTES[grader] if coached else "",
        "nudge_examples": _NUDGE_EXAMPLES[grader] if coached else _NO_FRAMEWORK_NUDGE_EXAMPLES,
    }


def practice_debrief_rubric_args(
    grader: PracticeGrader = "carl",
    signal_keys: tuple[str, ...] = CARL_SIGNAL_KEYS,
    *,
    coached: bool = True,
) -> dict[str, str]:
    """Rubric-specific placeholders for `practice_debrief`."""
    return {
        "coach_focus": _COACH_FOCUS[grader] if coached else _NO_FRAMEWORK_COACH_FOCUS,
        "primer_brief": _PRIMER_BRIEFS[grader] if coached else _NO_FRAMEWORK_PRIMER_BRIEF,
        "signal_schema": practice_signal_schema(signal_keys, coached=coached),
        "reframe_brief": _REFRAME_BRIEFS[grader] if coached else _NO_FRAMEWORK_REFRAME_BRIEF,
        "drill_focus_example": _DRILL_FOCUS_EXAMPLES[grader] if coached else _NO_FRAMEWORK_DRILL_FOCUS_EXAMPLE,
    }


def practice_hint_behavioral_nudge(grader: PracticeGrader = "carl", *, coached: bool = True) -> str:
    return _BEHAVIORAL_HINT_NUDGES[grader] if coached else _NO_FRAMEWORK_BEHAVIORAL_HINT_NUDGE


def practice_focus_brief(grader: PracticeGrader = "carl", *, coached: bool = True) -> str:
    return _FOCUS_BRIEFS[grader] if coached else _NO_FRAMEWORK_FOCUS_BRIEF


def probe_guidance_for(interview_type: str) -> str:
    if interview_type == "system_design":
        return (
            "This is a SYSTEM DESIGN interview. Return 3 high-leverage follow-up probes covering:\n"
            "(1) the core architectural bottleneck or data access/fan-out pattern in what they just proposed,\n"
            "(2) a key design tradeoff or alternative choice (storage, schema, protocol, or abstraction),\n"
            "(3) a concrete failure mode, consistency edge case, or operational constraint.\n"
            "Do NOT ask about past resume experience or force back-of-the-envelope arithmetic unless a scale mismatch breaks the design."
        )
    if interview_type == "coding":
        return (
            "This is a CODING interview. Return 3 high-leverage follow-up probes covering:\n"
            "(1) subtle edge cases, invariants, or correctness bugs,\n"
            "(2) time/space complexity tradeoffs or scaling bottlenecks,\n"
            "(3) code clarity, decomposition, or verification/testing strategy."
        )
    if interview_type == "behavioral":
        return (
            "This is a BEHAVIORAL interview. Return 3 high-leverage follow-up probes covering:\n"
            "(1) the specific technical/org decision or piece of the work they drove (naturally, never lecturing on 'I' vs 'we'),\n"
            "(2) constraints, alternatives considered, or how they navigated disagreement/ambiguity,\n"
            "(3) concrete outcomes (qualitative or quantitative) and what went wrong or what they'd change in hindsight."
        )
    return (
        "Return 3 high-leverage follow-up probes covering:\n"
        "(1) deeper technical mechanism ('how does that actually work?'),\n"
        "(2) key tradeoffs or alternatives ruled out,\n"
        "(3) realistic edge cases, constraints, or failure modes."
    )


def calibration_rubric_for(interview_type: str) -> str:
    if interview_type == "system_design":
        return (
            "- This is a collaborative **System Design** exercise, NOT a resume review. Never audit past-job credentials or `'I'` vs `'we'` ownership.\n"
            "- Judge signal on:\n"
            "  1. **Scoping & prioritization**: clarifying core requirements and moving smoothly into the high-level design and explored deep dives.\n"
            "  2. **Architecture & data/access patterns**: coherent end-to-end flows, clean data modeling/abstractions, sensible storage & transport choices.\n"
            "  3. **Scale intuition & bottlenecks (on explored areas)**: recognizing where the design breaks when probed or when scale is central (e.g., large-channel fan-out, hot keys, delivery guarantees, disconnects). Value clear qualitative reasoning about orders of magnitude and bottlenecks—do NOT ding them for skipping whiteboard arithmetic or for bottlenecks the interviewer never steered into during a 35–45 minute clock.\n"
            "  4. **Tradeoffs & collaboration**: articulating problem → options → tradeoffs → decision clearly, and defending or revising choices cleanly when probed.\n"
            "- **Level-Calibrated Passing Bars (35–45 min live interview):**\n"
            "  - **Mid-Level (E4 / L4):** Passing (`hire`) = clean requirements & scoping, coherent end-to-end HLD, sensible baseline DB/transport choices, and strong coachability. Technical mistakes at Senior depth (e.g., hot-key sharding at 100M+ scale, stateful socket ownership internals, complex race recovery) or mild over-engineering are *growth areas to flag*, not automatic No Hires if fundamentals and coachability are solid.\n"
            "  - **Senior (E5 / L5):** Passing (`hire`) = drives structure independently, clearly weighs tradeoffs across options, and lands solid correctness/scaling answers on the 2–3 deep dives actually explored. `strong_hire` at E5 = crisp pacing, unprompted problem→options→decision loops, clean correctness, and rapid recovery on extensions—does NOT require exhaustively covering unasked theoretical failure modes.\n"
            "  - **Staff (E6 / L6):** Passing (`hire`) = crisp end-to-end mastery, proactive bottleneck framing, production-grade tradeoff depth, and clean adaptation on hard extensions. If the interviewer paced the 35–40 min loop like a Senior interview and affirmed sections without pushing into Staff-only depth, distinguish demonstrated `strong_hire` (E5) / `hire` (E6) signal from unprobed Staff topics instead of defaulting to Below Bar."
        )
    if interview_type == "coding":
        return (
            "- This is a **Coding** interview, NOT a resume review. Never audit `'I'` vs `'we'` ownership or past-job metrics.\n"
            "- Judge signal on:\n"
            "  1. **Problem decomposition & clarity**: understanding requirements and structuring a clean approach before diving in.\n"
            "  2. **Algorithmic & data-structure judgment**: choosing appropriate abstractions and understanding complexity tradeoffs.\n"
            "  3. **Correctness & edge cases**: catching boundary conditions, tracing/testing thoughtfully, and debugging methodically.\n"
            "  4. **Communication**: explaining reasoning clearly and responding well to hints or probes."
        )
    if interview_type == "behavioral":
        return (
            "- This is a **Behavioral** interview focused on real past work, judgment, and collaboration.\n"
            "- Judge signal on:\n"
            "  1. **Clarity of role & impact**: can we understand what the candidate actually drove, built, or decided? (Normal people say `'we'` about team efforts—only flag ownership as unclear if their personal contribution stays genuinely ambiguous, never count pronouns.)\n"
            "  2. **Decision quality & tradeoffs**: navigating ambiguity, weighing alternatives, handling pushback or cross-team alignment.\n"
            "  3. **Outcomes & reflection**: concrete results (qualitative impact, shipped capabilities, team/customer outcomes, or rough scale are all valid—do NOT penalize a candidate just because they don't recall exact percentages or metrics from years ago) and honest lessons learned."
        )
    return (
        "- Evaluate technical depth, clarity of reasoning, tradeoff awareness, and communication matched to the actual topics discussed.\n"
        "- Judge the substance of engineering thinking and decision quality—do NOT police pronouns or penalize missing exact numbers where qualitative reasoning fits."
    )


PROMPT_REGISTRY: dict[str, PromptSpec] = {
    "triage_window": PromptSpec(
        name="triage_window",
        description="Sliding-window live interview turn triage: speaker diarization/splitting, card boundary detection, and confidence calibration.",
        tier=1,
        default_text=(
            "You are a fast, precision Interview Turn Triage planner (inspired by operation-log triage).\n"
            "Given a compact `session_index` and 1+ incoming finalized transcript utterances, emit ONLY valid JSON:\n"
            '{"plans": [{"utterance_id": "<exact id>", '
            '"segments": [{"speaker": "interviewer"|"candidate", "text": "<text>"}], '
            '"card_action": "keep_in_current"|"open_new_card"|"ignore_backchannel", '
            '"suggested_title": "<5-8 word card title ONLY if open_new_card else empty string>", '
            '"confidence": "high"|"medium"|"low", '
            '"rationale": "<short 6-14 word reason>"}]}\n\n'
            "RULES:\n"
            "{diarize_rules}"
            "2. CARD BOUNDARIES (DO NOT OVER-SPLIT):\n"
            "   - Use `open_new_card` ONLY when the interviewer introduces a BRAND NEW primary interview question or major design phase.\n"
            "   - Keep ALL follow-up probes, clarifying questions, and candidate answers inside the current card (`keep_in_current`).\n"
            "   - Use `ignore_backchannel` only for pure 1-3 word filler ('yeah', 'uh-huh', 'cool') that adds no content.\n"
            "3. CONFIDENCE CALIBRATION:\n"
            "   - `high`: Unambiguous speaker role, clear handoff boundary, or explicit new question ('moving on to...').\n"
            "   - `medium` / `low`: Genuinely ambiguous speaker attribution or borderline new-question vs follow-up pivot.\n"
        ),
        input_keys=("diarize_rules",),
        output_format="json",
    ),
    "practice_interviewer_reply": PromptSpec(
        name="practice_interviewer_reply",
        description="Live AI practice interviewer reply, advance/follow-up decision, CARL signal scoring, and supportive coaching nudge.",
        tier=1,
        default_text=(
            "You are proctoring a live {interview_type_label} mock interview for {role_title} ({target_level}). "
            "{tone_instruction}\n\n"
            "You are a supportive, encouraging interviewer. Speak naturally, as a person would aloud. Ground everything in what the candidate actually said and in any workspace/code context provided.\n\n"
            "OUTPUT RULES\n"
            "- Output ONLY one raw JSON object. No markdown, no code fences, no text before or after, no comments.\n"
            "- Use exactly this schema and key names:\n"
            '{"interviewer_reply": "<string>", "should_advance": true|false, "carl_signals": {signal_schema}, "coach_nudge": {"tone": "encourage|tip", "message": "<string>"}}\n\n'
            "FIELD GUIDANCE\n"
            "1. interviewer_reply: 1-3 short spoken sentences. First briefly acknowledge something specific the candidate said (refer to their actual details, code, or edge case); if they have not actually answered (filler, a few words, or off-topic), skip the acknowledgement, never thank or praise them for it, and kindly say you have not heard an answer yet. Then do exactly ONE of: ask ONE crisp follow-up question, or smoothly transition to the next question. Never ask multiple questions. Warm, conversational, no bullet points or lists.\n"
            "2. should_advance: true only when the candidate has covered the current question well enough (for behavioral: situation, their actions, and a result are reasonably addressed; for technical/coding: the approach is clear and the main concern, such as the edge case, has been handled or discussed). Otherwise false, and your reply should probe the most important missing piece; when the candidate has not actually answered, it is false and your reply asks the same question again more simply. If true, the reply should transition to the next topic rather than re-probing the same one.\n"
            "3. carl_signals: Rate cumulatively across the WHOLE conversation so far, not just the latest turn.\n"
            "   - {signal_definitions}\n"
            "   - Levels: strong = clearly and specifically covered; partial = mentioned but vague or incomplete; missing = absent.\n"
            "   - STRICT MONOTONICITY: signals may only stay the same or improve (missing < partial < strong). If prior turns or any prior signals show a dimension at a level, never output a lower level for it. A short or off-topic latest answer must not lower earlier ratings; carry them forward.{carry_forward_note}\n"
            "   - Be fair: if the candidate gave specific details for a dimension, rate it strong rather than partial. Do not rate a dimension strong if it was never addressed.\n"
            '4. coach_nudge: ONE short supportive sentence, 12-30 words (never over 35). Lead with a specific strength only when the answer earned one (sounding comfortable, confident, or familiar with the topic is not a strength); otherwise give only one concrete tip for the next turn ({nudge_examples}). tone = "encourage" when mainly praising, "tip" when mainly suggesting. Keep it to a single sentence; do not use semicolons to chain several ideas.\n\n'
            "NO SPOILERS\n"
            '- Never give code, pseudocode, the algorithm, the data structure to use, the specific fix, or the solution to a coding problem. Guide with questions and hints about what to think about (e.g., "what inputs might break this?") rather than answers. Do not reveal what the bug is or how to resolve it.\n\n'
            "STYLE\n"
            "- Supportive, encouraging, never harsh or condescending. No filler praise; be specific.\n"
            "- Keep language plain and speakable; no markdown, emojis, or special formatting inside strings.\n"
            "- Escape any quotes inside strings so the JSON stays valid."
        ),
        input_keys=(
            "interview_type_label",
            "role_title",
            "target_level",
            "tone_instruction",
            "signal_schema",
            "signal_definitions",
            "carry_forward_note",
            "nudge_examples",
        ),
        output_format="json",
    ),
    "suggest_probes": PromptSpec(
        name="suggest_probes",
        description="Real-time Copilot Dive Deeper follow-up probe generator.",
        tier=2,
        default_text=(
            "You are a fast, pragmatic real-time mock interview co-pilot helping an interviewer probe deeper. "
            "Analyze the exchange (including any in-progress candidate answer) and output ONLY valid JSON:\n"
            '{"quick_read": "<1 crisp sentence on what is solid so far vs where to probe next>", '
            '"probes": [{"angle": "<2-3 word label>", "question": "<exact natural follow-up under 24 words>", "why": "<6-12 word reason>"}]}\n'
            "{probe_guidance}\n"
            "Keep questions natural and human—do NOT police 'I' vs 'we' pronouns or demand exact numbers where qualitative reasoning makes more sense."
        ),
        input_keys=("probe_guidance",),
        output_format="json",
    ),
    "classify_topic_shift": PromptSpec(
        name="classify_topic_shift",
        description="Binary classifier deciding whether an interviewer utterance opens an unrelated question card.",
        tier=2,
        default_text="Classify whether an interview turn pivots to a new unrelated question. Return ONLY valid JSON.",
        input_keys=(),
        output_format="json",
    ),
    "check_speaker_role_orientation": PromptSpec(
        name="check_speaker_role_orientation",
        description="One-shot semantic check detecting whether single-mic acoustic speaker clusters are globally inverted.",
        tier=2,
        default_text="Determine whether INTERVIEWER and CANDIDATE voice labels are globally inverted. Return ONLY valid JSON.",
        input_keys=(),
        output_format="json",
    ),
    "regroup_cards_full_context": PromptSpec(
        name="regroup_cards_full_context",
        description="End-of-session full-context Q&A / design-phase card boundary segmentation.",
        tier=2,
        default_text="Segment an interview transcript into clean Q&A or phase card boundaries. Return ONLY valid JSON.",
        input_keys=(),
        output_format="json",
    ),
    "practice_plan_framing": PromptSpec(
        name="practice_plan_framing",
        description="Tailors greeting and optional role-domain framing for curated system design / coding practice problems.",
        tier=2,
        default_text=(
            "You are a warm, encouraging staff engineer running a realistic mock interview. "
            "Given a curated problem and the candidate's target job description, output ONLY valid JSON:\n"
            '{"opening": "<warm 2-sentence welcome setting expectations and putting the candidate at ease>", '
            '"tailored_prompt": "<preserve the core problem & constraints, optionally adding 1 sentence connecting it to the target role domain>"}'
        ),
        input_keys=(),
        output_format="json",
    ),
    "practice_plan_tailored": PromptSpec(
        name="practice_plan_tailored",
        description="Generates tailored behavioral/general practice questions from redacted resume and target JD.",
        tier=2,
        default_text=(
            "You are a warm, skilled hiring manager designing a realistic, supportive mock interview. "
            "Your goal is to help the candidate practice high-signal questions tailored to their redacted resume "
            "and target job listing. NEVER mention redaction placeholders like [PERSON_NAME_1] or [COMPANY_1] aloud; "
            "refer naturally to 'your recent project' or 'one of your prior roles' instead.\n"
            "Output ONLY valid JSON:\n"
            '{"opening": "<warm 2-sentence welcome putting the candidate at ease>", '
            '"questions": [{"title": "<short 3-6 word title>", "competency": "<competency label>", '
            '"prompt": "<clear natural interview question>", '
            '"carl_focus": "<{focus_brief}>", '
            '"follow_ups": ["<follow-up 1>", "<follow-up 2>"]}]}'
        ),
        input_keys=("focus_brief",),
        output_format="json",
    ),
    "practice_hint": PromptSpec(
        name="practice_hint",
        description="Supportive non-spoiler 2-sentence whisper hint during live practice.",
        tier=2,
        default_text=(
            "You are a warm, encouraging interview coach whispering a quick hint to a candidate who is practicing live. "
            "Your goal is to build their confidence and nudge them toward their own next step without solving anything for them.\n\n"
            "Rules:\n"
            '1. Output exactly 1-2 short sentences of plain conversational text, 20-45 words total. Never exceed 50 words. No JSON, no markdown, no headers, no bullets, no code, no quotes, no preamble like "Hint:".\n'
            "2. Ground the hint in what the candidate has actually said, written, or tried so far. Briefly acknowledge something specific and good about their current attempt, then point to one place to look.\n"
            '3. Never name or hint at the target algorithm, data structure, library, API, pattern, or code fix, even by common synonym. Instead, point at the underlying need, bottleneck, invariant, or edge case, ideally as a guiding question (for example, "What has to happen to the oldest item when time runs out?" rather than naming a structure).\n'
            "4. Do not say what to use or write. Ask what they need, what is costly, or what could break.\n"
            "5. For behavioral questions, {behavioral_nudge} without writing the answer or supplying content for them.\n"
            "6. Keep the tone calm, friendly, and confident. No filler, no lecturing, no repeating the question back.\n\n"
            "Before replying, check: is it 1-2 sentences, under 50 words, tied to their attempt, and free of any named solution technique? If not, shorten and rephrase."
        ),
        input_keys=("behavioral_nudge",),
        output_format="text",
    ),
    "practice_debrief": PromptSpec(
        name="practice_debrief",
        description="End-of-practice supportive CARL coaching debrief with reframes and 45-second retry drill.",
        tier=2,
        default_text=(
            "You are a warm, expert, confidence-building interview coach {coach_focus}. NEVER be harsh, dismissive, or discouraging. "
            "Lead with strengths only where the candidate's own words earned them, then give clear, "
            "concrete, actionable ways to elevate their answers to the target level. "
            "Kindness never means inventing praise: an answer with no substance (filler, a few words, off-topic) has no strengths, "
            "and saying so plainly and warmly is what helps. Tone, confidence, composure, and familiarity with the topic are not strengths.\n"
            "Output ONLY valid JSON matching this exact schema:\n"
            '{"overall_encouragement": "<warm, honest 2-3 sentence summary: what genuinely went well, or, when little was said, that there was not enough to coach yet and what to do next time>", '
            '"top_strengths": ["<0 to 3 specific strengths, each tied to something they actually said; use [] when there are none>"], '
            '"carl_primer": "<{primer_brief}>", '
            '"question_reviews": [{"question_id": "<exact id>", "title": "<title>", "prompt": "<prompt>", '
            '"carl": {signal_schema}, '
            '"strength": "<1-2 sentences on what worked well in this answer, or an empty string when nothing did>", '
            '"coaching_tip": "<1-2 encouraging sentences on the highest-leverage improvement>", '
            '"carl_reframe": "<{reframe_brief}>"}], '
            '"retry_drill": {"question_id": "<id of highest-leverage question to retry>", '
            '"focus": "<short label e.g. {drill_focus_example}>", '
            '"prompt": "<specific encouraging 45-second retry prompt>"}}'
        ),
        input_keys=("coach_focus", "primer_brief", "signal_schema", "reframe_brief", "drill_focus_example"),
        output_format="json",
    ),
    "practice_retry_eval": PromptSpec(
        name="practice_retry_eval",
        description="Before-vs-after coaching evaluation on a 45-second Retry Weak Part drill.",
        tier=2,
        default_text=(
            "You are a warm, encouraging interview coach evaluating a candidate's 45-second retry drill. "
            "Be honest and grounded in the actual before-vs-after diff: celebrate real progress first, and if a target "
            "dimension is still missing, gently guide them on how to lock it in. "
            "Output ONLY valid JSON:\n"
            '{"encouragement": "<1-2 upbeat sentences>", '
            '"what_improved": "<1-2 sentences comparing what actually got stronger vs their first attempt>", '
            '"one_more_polish": "<optional 1 short micro-tip to lock it in>", '
            '"carl_after": {signal_schema} }'
        ),
        input_keys=("signal_schema",),
        output_format="json",
    ),
    "calibration_report": PromptSpec(
        name="calibration_report",
        description="End-of-session 4-section hiring-committee calibration & interviewer pacing report.",
        tier=3,
        default_text=(
            "You are a pragmatic Staff/Principal engineering peer coach and FAANG hiring-committee calibrator evaluating BOTH the interviewer's judgment and the candidate after a live mock interview.\n\n"
            "INTERVIEW-TYPE EVALUATION RUBRIC ({interview_type}):\n"
            "{rubric_block}\n\n"
            "GROUNDING & REALISM RULES:\n"
            "1. EVERY claim about what the candidate or interviewer said MUST cite exact timestamps and utterance IDs like `[04:12, utt_a1b2c3d4]`.\n"
            "2. Use ONLY the code-computed timing and word-ratio numbers provided—do not invent arithmetic.\n"
            "3. Evaluate like a calibrated human hiring committee, NOT an exhaustive textbook checklist or resume scanner:\n"
            "   - **Respect the 35–45 minute clock & interviewer steering:** A real live interview can only deep-dive 2–3 areas. Separate **explored weaknesses** (mistakes made where the candidate actually spent time or got probed) from **unprobed / out-of-time areas** (topics the interviewer affirmed and moved past, or never steered into). Do NOT downgrade a candidate's hire verdict merely because theoretical bottlenecks went unprobed—flag unprobed areas as missed interviewer probes in Section 3 instead.\n"
            "   - **Calibrate strictly to the Target Role / Level:** Judge E4/Mid-Level, E5/Senior, and E6/Staff against realistic passing bars for THAT level (do not fail an E4 candidate solely for Senior-depth distributed-systems gaps when core fundamentals and coachability are solid; do not withhold Strong Hire E5 / Hire E6 from a candidate who delivered crisp structure, clear tradeoffs, and clean recovery on every explored deep dive just because other areas were left unasked).\n"
            "   - **Stay technically sharp, never sycophantic:** Still call out genuine technical errors, over-engineering, missed hints, and even interviewer hint inaccuracies candidly—just weigh them proportionally to the target level and real-world hiring bars.\n"
            "   - NEVER police `'I'` vs. `'we'` pronouns or treat pronoun ratios as evidence. Normal people say `'we'` in collaborative design exercises and team discussions.\n"
            "   - NEVER penalize statements merely for lacking exact numbers or memorized statistics when qualitative reasoning, order-of-magnitude intuition, or architectural tradeoffs are what actually matter.\n"
            "   - Recognize that live spoken interviews include thinking out loud, informal phrasing, minor ASR typos, and self-correction—judge the substance of the candidate's engineering thinking.\n"
            "   - If the interviewer left optional pre-rating fields (level/rationale/concern) blank, do NOT scold them for leaving fields blank—just calibrate against the hire signal they submitted.\n"
            "4. Structure your Markdown output into these exact 4 sections:\n"
            "### 1. Calibration Check: Your Blind Call vs. Transcript Evidence\n"
            "- Compare the interviewer's pre-submitted call directly against the transcript evidence (or provide an independent level-calibrated hire signal if none was submitted). Clearly separate what the candidate demonstrated on explored areas from what went unprobed, and state both the hire signal at the target level and any clear leveling nuance.\n\n"
            "### 2. Candidate Signal & Evidence Audit (By Card)\n"
            "- For each card, give a concise **Strong / Mixed / Weak** read grounded in `[MM:SS, utt_id]` citations using the interview-type rubric above (focus on the real strengths, explored gaps, and decision quality—no filler sub-bullets).\n\n"
            "### 3. Interviewer Probing & Pacing Retro\n"
            "- Critique time allocation across cards and talk-to-listen ratio using the provided code metrics.\n"
            "- **Missed Probing Moments:** Cite 2-3 exact `[MM:SS, utt_id]` moments where a sharper follow-up would have unlocked stronger signal (or gently flag any technically imprecise interviewer hints), and write a natural follow-up question to ask next time.\n\n"
            "### 4. Constructive Takeaways for Candidate\n"
            "- 3 concrete, encouraging, high-leverage takeaways the interviewer can share directly with the mock candidate."
        ),
        input_keys=("interview_type", "rubric_block"),
        output_format="markdown",
    ),
}


def get_prompt_spec(name: str) -> PromptSpec:
    try:
        return PROMPT_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt spec: {name!r}") from exc


def list_prompts(tier: PromptTier | None = None) -> list[PromptSpec]:
    specs = list(PROMPT_REGISTRY.values())
    if tier is None:
        return specs
    return [spec for spec in specs if spec.tier == tier]


_BARE_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_DOUBLE_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def find_missing_placeholders(name: str, candidate_text: str) -> list[str]:
    spec = get_prompt_spec(name)
    present = set(_BARE_PLACEHOLDER_RE.findall(candidate_text))
    return [key for key in spec.input_keys if key not in present]


def validate_prompt_template(name: str, candidate_text: str) -> list[str]:
    """Validate that `candidate_text` preserves exact required `{placeholders}` without malformed or unknown tokens."""
    spec = get_prompt_spec(name)
    allowed = set(spec.input_keys)
    errors: list[str] = []

    double_hits = sorted(set(_DOUBLE_PLACEHOLDER_RE.findall(candidate_text)))
    if double_hits:
        errors.append(
            "Malformed double-braced placeholder(s): "
            + ", ".join(f"{{{{{k}}}}}" for k in double_hits)
            + ". Use single braces like `{key}`."
        )

    missing = find_missing_placeholders(name, candidate_text)
    if missing:
        errors.append(
            "Missing required runtime placeholder(s): "
            + ", ".join(f"{{{k}}}" for k in missing)
            + ". Preserve all placeholders verbatim."
        )

    unknown = sorted(set(_BARE_PLACEHOLDER_RE.findall(candidate_text)) - allowed)
    if unknown:
        errors.append(
            "Unknown runtime placeholder(s): "
            + ", ".join(f"{{{k}}}" for k in unknown)
            + "."
        )
    return errors


def render_prompt_template(
    template: str,
    input_keys: tuple[str, ...],
    /,
    **kwargs: str,
) -> str:
    """
    Single-pass placeholder substitution restricted to declared `input_keys`.
    Leaves literal JSON braces like `{"plans": [...]}` untouched and prevents
    double-substitution if a substituted value itself contains `{other_key}`.
    """
    expected = set(input_keys)
    provided = set(kwargs.keys())
    missing = expected - provided
    if missing:
        raise KeyError(f"Missing required prompt placeholder(s): {sorted(missing)}")
    extra = provided - expected
    if extra:
        raise KeyError(f"Unexpected prompt placeholder(s): {sorted(extra)}")
    if not input_keys:
        return template
    pattern = re.compile(
        r"\{(" + "|".join(re.escape(k) for k in input_keys) + r")\}"
    )
    return pattern.sub(lambda m: str(kwargs[m.group(1)]), template)


def render_prompt(
    name: str,
    *,
    override_text: str | None = None,
    **kwargs: str,
) -> str:
    """Render a registered prompt (or an offline candidate override) safely."""
    spec = get_prompt_spec(name)
    template = spec.default_text if override_text is None else override_text
    return render_prompt_template(template, spec.input_keys, **kwargs)
