"""Pure canonical multi-turn conversation builder for Practice Studio.

Builds one append-only `[system, user_1, assistant_1, ..., user_N]` message sequence
from persisted `PracticeSession` state so every transport — persistent `agy`/`claude`
CLI workers, HTTP APIs (Anthropic/OpenRouter/OpenAI/Gemini/oMLX), and stateless CLIs
(`codex`/`gemini`/`pi`) — shares ONE prompt shape and achieves optimal prefix caching.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .prompts import practice_reply_rubric_args, practice_tone_instruction, render_prompt

if TYPE_CHECKING:
    from .practice import PracticeSession

PRACTICE_CONVERSATION_VERSION = 2
MAX_PRACTICE_CONVERSATION_CHARS = 120_000
MAX_DESIGN_NOTES_CHARS = 16_000
MAX_DESIGN_NOTES_PROMPT_CHARS = 3_000
MAX_QUESTION_PLAN_CHARS = 24_000
_SEED_HISTORY_TARGET_CHARS = MAX_PRACTICE_CONVERSATION_CHARS // 2


def practice_max_followups(interview_type: str) -> int:
    """Maximum follow-up turns allowed on one question before advancing."""
    if interview_type == "system_design":
        return 12
    if interview_type == "coding":
        return 3
    return 2


def bounded_head_tail(text: str, max_chars: int) -> str:
    """Keep opening context (35%) and latest trailing edits (65%) strictly within `max_chars`."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n[... middle omitted ...]\n"
    if max_chars <= len(marker) + 4:
        return text[:max_chars]
    usable = max_chars - len(marker)
    head_len = max(1, int(usable * 0.35))
    tail_len = max(1, usable - head_len)
    return text[:head_len].rstrip() + marker + text[-tail_len:].lstrip()


def format_bounded_list(
    items: list[Any],
    *,
    max_items: int = 8,
    max_item_chars: int = 240,
    max_total_chars: int = 1_600,
) -> str:
    """Render a compact, budget-safe list representation for prompts (safe for arbitrary custom banks)."""
    cleaned: list[str] = []
    used = 0
    for raw in items[:max_items]:
        item = " ".join(str(raw or "").split()).strip()
        if not item:
            continue
        if len(item) > max_item_chars:
            item = item[: max(1, max_item_chars - 1)].rstrip() + "…"
        cost = len(item) + (3 if cleaned else 0)
        if cleaned and used + cost > max_total_chars:
            break
        if not cleaned and len(item) > max_total_chars:
            item = item[: max(1, max_total_chars - 1)].rstrip() + "…"
        cleaned.append(item)
        used += cost
    if not cleaned:
        return ""
    omitted = max(0, len([x for x in items if str(x or "").strip()]) - len(cleaned))
    rendered = "; ".join(cleaned)
    if omitted > 0:
        rendered += f" (+{omitted} more)"
    return rendered


def _format_notes_delta(prev_notes: str, curr_notes: str, max_chars: int = MAX_DESIGN_NOTES_PROMPT_CHARS) -> str:
    """Return a bounded view of design notes that always includes the edited region when notes exceed `max_chars`."""
    if len(curr_notes) <= max_chars:
        return curr_notes
    if not prev_notes:
        return bounded_head_tail(curr_notes, max_chars)

    # Locate the changed slice via common prefix and suffix so edits anywhere in a 16k pad are visible.
    min_len = min(len(prev_notes), len(curr_notes))
    prefix = 0
    while prefix < min_len and prev_notes[prefix] == curr_notes[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < (min_len - prefix)
        and prev_notes[len(prev_notes) - 1 - suffix] == curr_notes[len(curr_notes) - 1 - suffix]
    ):
        suffix += 1
    edit_start = prefix
    edit_end = len(curr_notes) - suffix

    # Expand a window around the changed span within `max_chars`.
    marker_head = "[... earlier notes omitted ...]\n"
    marker_tail = "\n[... later notes omitted ...]"
    budget = max_chars - len(marker_head) - len(marker_tail)
    edit_len = max(0, edit_end - edit_start)
    if edit_len >= budget:
        span = bounded_head_tail(curr_notes[edit_start:edit_end], budget)
        out = (marker_head if edit_start > 0 else "") + span + (marker_tail if edit_end < len(curr_notes) else "")
        return out[:max_chars]

    slack = budget - edit_len
    win_start = max(0, edit_start - slack // 2)
    win_end = min(len(curr_notes), win_start + budget)
    win_start = max(0, win_end - budget)
    body = curr_notes[win_start:win_end]
    out = (marker_head if win_start > 0 else "") + body + (marker_tail if win_end < len(curr_notes) else "")
    return out[:max_chars]


def _build_intervening_notes_budgeted(
    session: PracticeSession,
    start_idx: int,
    *,
    max_chars: int,
) -> list[str]:
    """Capture turns from `start_idx`, keeping newest turns within `max_chars` so overflow resets stay bounded."""
    raw_notes = _build_intervening_notes(session, start_idx)
    if not raw_notes:
        return []
    if max_chars <= 0:
        return ["- (... earlier interview turns omitted to stay within context budget ...)"]
    total = sum(len(n) + 1 for n in raw_notes)
    if total <= max_chars:
        return raw_notes

    kept: list[str] = []
    used = 0
    marker = "- (... earlier interview turns omitted to stay within context budget ...)"
    budget = max(0, max_chars - len(marker) - 2)
    for note in reversed(raw_notes):
        clipped = note[: max(200, budget)] if len(note) > budget and not kept else note
        cost = len(clipped) + 1
        if kept and used + cost > budget:
            break
        if not kept and cost > budget:
            kept.append(clipped[: max(80, budget)])
            break
        kept.append(clipped)
        used += cost
    kept.reverse()
    return [marker, *kept]


class PracticeConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PracticeConversation(BaseModel):
    version: int = 1
    system: str = ""
    messages: list[PracticeConversationMessage] = Field(default_factory=list)
    last_workspace: dict[str, str] = Field(default_factory=dict)
    last_synced_turn_count: int = 0


def build_workspace_snapshot(workspace_context: dict[str, Any] | None) -> dict[str, str]:
    """Extract the bounded model-visible workspace fields for deterministic change detection."""
    if not isinstance(workspace_context, dict):
        return {}
    out: dict[str, str] = {}
    diagram = str(workspace_context.get("diagram_summary") or "").strip()
    if diagram:
        out["diagram_summary"] = diagram[:2500]
    design_notes = str(workspace_context.get("design_notes") or "").strip()
    if design_notes:
        out["design_notes"] = design_notes[:MAX_DESIGN_NOTES_CHARS]
    code = str(workspace_context.get("code") or "").strip()
    if code:
        out["language"] = str(workspace_context.get("language") or "code").strip()[:40] or "code"
        out["code"] = code[:3000]
    test_out = str(workspace_context.get("test_output") or "").strip()
    if test_out:
        out["test_output"] = test_out[:1200]
    return out


def _format_workspace_delta(prev: dict[str, str], curr: dict[str, str]) -> str:
    """Format only fields that changed (or were explicitly cleared) since the last committed turn."""
    parts: list[str] = []
    if curr.get("diagram_summary") != prev.get("diagram_summary"):
        if curr.get("diagram_summary"):
            parts.append(f"LIVE SYSTEM DESIGN DIAGRAM STATE:\n{curr['diagram_summary']}")
        elif prev.get("diagram_summary"):
            parts.append("LIVE SYSTEM DESIGN DIAGRAM STATE:\n(cleared)")

    if curr.get("design_notes") != prev.get("design_notes"):
        if curr.get("design_notes"):
            excerpt = _format_notes_delta(prev.get("design_notes", ""), curr["design_notes"])
            parts.append(f"LIVE SYSTEM DESIGN NOTES (API / SCHEMA / MATH):\n{excerpt}")
        elif prev.get("design_notes"):
            parts.append("LIVE SYSTEM DESIGN NOTES (API / SCHEMA / MATH):\n(cleared)")

    if curr.get("code") != prev.get("code") or curr.get("language") != prev.get("language"):
        if curr.get("code"):
            lang = curr.get("language") or "code"
            parts.append(f"LIVE CODING WORKSPACE ({lang}):\n```\n{curr['code']}\n```")
        elif prev.get("code"):
            parts.append("LIVE CODING WORKSPACE:\n(cleared)")

    if curr.get("test_output") != prev.get("test_output"):
        if curr.get("test_output"):
            parts.append(f"LATEST TEST RUN OUTPUT:\n{curr['test_output']}")
        elif prev.get("test_output"):
            parts.append("LATEST TEST RUN OUTPUT:\n(cleared)")

    return "\n\n".join(parts)


def build_practice_session_system_prompt(session: PracticeSession) -> str:
    """Render the frozen session-wide Practice Studio system prompt."""
    rubric = session.config.rubric
    return render_prompt(
        "practice_interviewer_reply",
        interview_type_label=session.config.interview_type.replace("_", " "),
        role_title=session.role_title,
        target_level=session.config.target_level,
        tone_instruction=practice_tone_instruction(session.config.coach_style),
        **practice_reply_rubric_args(rubric.grader, rubric.signal_keys, coached=rubric.coached),
    )


def _format_question_plan(session: PracticeSession) -> str:
    """Format the full immutable question plan once on Turn 1 so Q advances never alter prefix."""
    lines = ["SESSION INTERVIEW PLAN (fixed for the full interview):"]
    if session.company:
        lines.append(f"Company: {session.company[:160]}")
    for idx, q in enumerate(session.questions, 1):
        focus = (q.carl_focus or format_bounded_list(q.core_requirements, max_items=8, max_item_chars=200, max_total_chars=1_000)).strip()[:1_000]
        title = (q.title or q.competency or "Question").strip()[:160]
        prompt = str(q.prompt or "").strip()[:2_000]
        lines.append(f"Q{idx} [{q.id}] {title}")
        lines.append(f"  Prompt: {prompt}")
        if focus:
            lines.append(f"  {session.config.rubric.focus_label}: {focus}")
        if q.level_bar:
            lines.append(f"  Depth expected at {session.config.target_level} level: {str(q.level_bar).strip()[:600]}")
        deep_dives_str = format_bounded_list(q.deep_dives, max_items=8, max_item_chars=240, max_total_chars=1_600)
        if deep_dives_str:
            lines.append(f"  Deep-dive bottlenecks & failure modes to probe: {deep_dives_str}")
        follow_ups_str = format_bounded_list(q.follow_ups, max_items=8, max_item_chars=240, max_total_chars=1_600)
        if follow_ups_str:
            lines.append(f"  Suggested follow-up angles: {follow_ups_str}")
    rendered = "\n".join(lines)
    return rendered[:MAX_QUESTION_PLAN_CHARS]


def _format_turn_directive(
    *,
    current_question: dict[str, Any],
    next_question: dict[str, Any] | None,
    follow_up_count: int,
    max_followups: int,
    intervening_notes: list[str],
    ws_delta: str,
    candidate_answer: str,
    debrief_name: str,
    interview_type: str = "",
) -> str:
    q_id = current_question.get("id") or "current"
    q_title = current_question.get("title") or ""
    next_note = (
        f"Next planned question (ONLY introduce if should_advance=true): "
        f"[{next_question.get('id', 'next')}] {next_question.get('prompt')}"
        if next_question
        else f"This is the FINAL question of the interview. If should_advance=true, wrap up warmly and invite them to view their {debrief_name}."
    )
    pacing_note = (
        f"(system design: keep should_advance=false while clarifying requirements, sketching the baseline "
        f"architecture, or exploring deep dives; advance only after requirements, baseline design, and "
        f"2-3 concrete bottlenecks/failure modes are covered OR after {max_followups} turns)."
        if interview_type == "system_design"
        else f"(advance once you've asked {max_followups} follow-ups OR they've already given strong complete signal)."
    )
    sections: list[str] = []
    if intervening_notes:
        sections.append("SESSION EVENTS SINCE LAST TURN:\n" + "\n".join(intervening_notes))
    sections.append(
        f"ACTIVE QUESTION: [{q_id}] {q_title}\n"
        f"Prompt: {current_question.get('prompt')}\n"
        f"Follow-ups already asked on this question: {follow_up_count} {pacing_note}\n"
        f"{next_note}"
    )
    if ws_delta:
        sections.append(ws_delta)
    sections.append(f"Candidate's latest answer:\n{candidate_answer}")
    return "\n\n".join(sections)


def _build_intervening_notes(session: PracticeSession, start_idx: int) -> list[str]:
    """Capture any manual `/next` transitions, opening prompt, or coach hints since `start_idx`."""
    notes: list[str] = []
    for t in session.turns[start_idx:]:
        if t.is_retry:
            continue
        if t.role == "interviewer":
            notes.append(f"- Interviewer message shown to candidate: {t.text}")
        elif t.role == "coach":
            notes.append(f"- Coach hint shown to candidate: {t.text}")
        elif t.role == "candidate":
            notes.append(f"- Prior candidate message: {t.text}")
    return notes


def build_practice_turn_messages(
    session: PracticeSession,
    *,
    current_question: dict[str, Any],
    next_question: dict[str, Any] | None,
    candidate_answer: str,
    follow_up_count: int,
    max_followups: int,
    workspace_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], str, dict[str, str], str, bool]:
    """
    Build the canonical `[system, user_1, assistant_1, ..., user_N]` message list without mutating `session`.

    Returns `(messages, pending_user_content, new_workspace_snapshot, system_prompt, reset_conversation)`.
    `reset_conversation` is True when seeding Turn 1, migrating a legacy session, or resetting after
    the conversation exceeds `MAX_PRACTICE_CONVERSATION_CHARS`.
    """
    curr_ws = build_workspace_snapshot(workspace_context)
    conv = session.llm_conversation
    expected_system = build_practice_session_system_prompt(session)

    total_chars = (
        len(conv.system) + sum(len(m.content) for m in conv.messages)
        if conv is not None
        else 0
    )
    stale_studio_conv = (
        conv is not None
        and conv.version < PRACTICE_CONVERSATION_VERSION
        and session.config.interview_type in ("system_design", "coding")
    )
    needs_seed = (
        conv is None
        or not conv.system
        or stale_studio_conv
        or conv.system != expected_system
        or total_chars > MAX_PRACTICE_CONVERSATION_CHARS
    )

    if not needs_seed:
        assert conv is not None
        system_prompt = conv.system
        intervening = _build_intervening_notes(session, conv.last_synced_turn_count)
        ws_delta = _format_workspace_delta(conv.last_workspace, curr_ws)
        user_content = _format_turn_directive(
            current_question=current_question,
            next_question=next_question,
            follow_up_count=follow_up_count,
            max_followups=max_followups,
            intervening_notes=intervening,
            ws_delta=ws_delta,
            candidate_answer=candidate_answer,
            debrief_name=session.config.rubric.debrief_name,
            interview_type=session.config.interview_type,
        )
        if total_chars + len(user_content) <= MAX_PRACTICE_CONVERSATION_CHARS:
            messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            for msg in conv.messages:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": user_content})
            return messages, user_content, curr_ws, system_prompt, False

    # Seed Turn 1, migrate legacy session, or cleanly re-budget after overflow so the prompt
    # ALWAYS stays well below MAX_PRACTICE_CONVERSATION_CHARS (preventing reset-every-turn loops).
    system_prompt = build_practice_session_system_prompt(session)
    plan_block = _format_question_plan(session)
    ws_delta = _format_workspace_delta({}, curr_ws)
    base_directive = _format_turn_directive(
        current_question=current_question,
        next_question=next_question,
        follow_up_count=follow_up_count,
        max_followups=max_followups,
        intervening_notes=[],
        ws_delta=ws_delta,
        candidate_answer=candidate_answer,
        debrief_name=session.config.rubric.debrief_name,
        interview_type=session.config.interview_type,
    )
    fixed_overhead = len(system_prompt) + len(plan_block) + len(base_directive) + 64
    history_budget = max(0, _SEED_HISTORY_TARGET_CHARS - fixed_overhead)
    prior_notes = _build_intervening_notes_budgeted(session, 0, max_chars=history_budget)
    directive = _format_turn_directive(
        current_question=current_question,
        next_question=next_question,
        follow_up_count=follow_up_count,
        max_followups=max_followups,
        intervening_notes=prior_notes,
        ws_delta=ws_delta,
        candidate_answer=candidate_answer,
        debrief_name=session.config.rubric.debrief_name,
        interview_type=session.config.interview_type,
    )
    user_content = f"{plan_block}\n\n{directive}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages, user_content, curr_ws, system_prompt, True


_SIGNAL_GRADES_RE = re.compile(r'"carl_signals"\s*:\s*\{[^{}]*\}')
_EARNED_GRADE_RE = re.compile(r'"(?:strong|partial)"')


def _without_signal_grades(assistant_raw: str) -> str:
    """The interviewer's reply with every structure grade set back to "missing"."""
    return _SIGNAL_GRADES_RE.sub(lambda grades: _EARNED_GRADE_RE.sub('"missing"', grades.group(0)), assistant_raw)


def commit_practice_conversation_turn(
    session: PracticeSession,
    *,
    system_prompt: str,
    user_content: str,
    assistant_raw: str,
    workspace_snapshot: dict[str, str],
    synced_turn_count: int,
    reset_conversation: bool,
) -> None:
    """
    Atomically append the committed `(user, assistant)` exchange onto `session.llm_conversation`.
    A session that is not graded on structure keeps no grades in the reply it stores and replays.
    """
    if not session.config.rubric.coached:
        assistant_raw = _without_signal_grades(assistant_raw)
    if reset_conversation or session.llm_conversation is None:
        session.llm_conversation = PracticeConversation(
            version=PRACTICE_CONVERSATION_VERSION,
            system=system_prompt,
            messages=[
                PracticeConversationMessage(role="user", content=user_content),
                PracticeConversationMessage(role="assistant", content=assistant_raw),
            ],
            last_workspace=dict(workspace_snapshot),
            last_synced_turn_count=synced_turn_count,
        )
        return

    conv = session.llm_conversation
    conv.messages.append(PracticeConversationMessage(role="user", content=user_content))
    conv.messages.append(PracticeConversationMessage(role="assistant", content=assistant_raw))
    conv.last_workspace = dict(workspace_snapshot)
    conv.last_synced_turn_count = synced_turn_count


def messages_prefix_digest(messages: list[dict[str, str]]) -> str:
    """Deterministic SHA-256 digest of a message sequence for CLI worker prefix-sync verification."""
    payload = json.dumps(
        [[str(m.get("role", "")), str(m.get("content", ""))] for m in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_flat_messages(
    messages: list[dict[str, str]],
    *,
    include_system: bool = True,
    format_rule: str = "",
) -> str:
    """
    Flatten a canonical `[system, user_1, assistant_1, ..., user_N]` conversation into one
    self-contained prompt for cold CLI bootstrap, crash recovery, and stateless CLIs (`codex`/`gemini`/`pi`).
    """
    sys_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system" and m.get("content")]
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]

    blocks: list[str] = []
    if include_system and sys_parts:
        blocks.append("<INSTRUCTIONS>\n" + "\n\n".join(sys_parts).strip() + "\n</INSTRUCTIONS>")
    if format_rule:
        blocks.append(format_rule)

    if len(turns) == 1 and turns[0].get("role") == "user":
        blocks.append("<INPUT>\n" + str(turns[0].get("content", "")) + "\n</INPUT>")
        return "\n\n".join(blocks)

    if len(turns) > 1:
        history_chunks: list[str] = []
        for idx, msg in enumerate(turns[:-1], 1):
            role_tag = "USER" if msg.get("role") == "user" else "ASSISTANT"
            history_chunks.append(f"--- TURN {idx} ({role_tag}) ---\n{msg.get('content', '')}")
        blocks.append("<CONVERSATION_HISTORY>\n" + "\n\n".join(history_chunks) + "\n</CONVERSATION_HISTORY>")
        latest = turns[-1]
        blocks.append("<CURRENT_TURN>\n" + str(latest.get("content", "")) + "\n</CURRENT_TURN>")
    return "\n\n".join(blocks)
