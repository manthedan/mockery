"""Pure canonical multi-turn conversation builder for Practice Studio.

Builds one append-only `[system, user_1, assistant_1, ..., user_N]` message sequence
from persisted `PracticeSession` state so every transport — persistent `agy`/`claude`
CLI workers, HTTP APIs (Anthropic/OpenRouter/OpenAI/Gemini/oMLX), and stateless CLIs
(`codex`/`gemini`/`pi`) — shares ONE prompt shape and achieves optimal prefix caching.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .prompts import practice_tone_instruction, render_prompt

if TYPE_CHECKING:
    from .practice import PracticeSession

MAX_PRACTICE_CONVERSATION_CHARS = 120_000
_SEED_HISTORY_TARGET_CHARS = MAX_PRACTICE_CONVERSATION_CHARS // 2


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
    return render_prompt(
        "practice_interviewer_reply",
        interview_type_label=session.config.interview_type.replace("_", " "),
        role_title=session.role_title,
        target_level=session.config.target_level,
        tone_instruction=practice_tone_instruction(session.config.coach_style),
    )


def _format_question_plan(session: PracticeSession) -> str:
    """Format the full immutable question plan once on Turn 1 so Q advances never alter prefix."""
    lines = ["SESSION INTERVIEW PLAN (fixed for the full interview):"]
    if session.company:
        lines.append(f"Company: {session.company}")
    for idx, q in enumerate(session.questions, 1):
        focus = q.carl_focus or ", ".join(q.core_requirements or [])
        lines.append(f"Q{idx} [{q.id}] {q.title or q.competency or 'Question'}")
        lines.append(f"  Prompt: {q.prompt}")
        if focus:
            lines.append(f"  CARL / Rubric Focus: {focus}")
        if q.level_bar:
            lines.append(f"  Depth expected at {session.config.target_level} level: {q.level_bar}")
        if q.follow_ups:
            lines.append(f"  Suggested follow-up angles: {q.follow_ups}")
    return "\n".join(lines)


def _format_turn_directive(
    *,
    current_question: dict[str, Any],
    next_question: dict[str, Any] | None,
    follow_up_count: int,
    max_followups: int,
    intervening_notes: list[str],
    ws_delta: str,
    candidate_answer: str,
) -> str:
    q_id = current_question.get("id") or "current"
    q_title = current_question.get("title") or ""
    next_note = (
        f"Next planned question (ONLY introduce if should_advance=true): "
        f"[{next_question.get('id', 'next')}] {next_question.get('prompt')}"
        if next_question
        else "This is the FINAL question of the interview. If should_advance=true, wrap up warmly and invite them to view their CARL Coaching Debrief."
    )
    sections: list[str] = []
    if intervening_notes:
        sections.append("SESSION EVENTS SINCE LAST TURN:\n" + "\n".join(intervening_notes))
    sections.append(
        f"ACTIVE QUESTION: [{q_id}] {q_title}\n"
        f"Prompt: {current_question.get('prompt')}\n"
        f"Follow-ups already asked on this question: {follow_up_count} "
        f"(advance once you've asked {max_followups} follow-ups OR they've already given strong complete signal).\n"
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

    total_chars = (
        len(conv.system) + sum(len(m.content) for m in conv.messages)
        if conv is not None
        else 0
    )
    needs_seed = conv is None or not conv.system or total_chars > MAX_PRACTICE_CONVERSATION_CHARS

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
    )
    user_content = f"{plan_block}\n\n{directive}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages, user_content, curr_ws, system_prompt, True


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
    """Atomically append the exact committed `(user, assistant)` exchange onto `session.llm_conversation`."""
    if reset_conversation or session.llm_conversation is None:
        session.llm_conversation = PracticeConversation(
            version=1,
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
