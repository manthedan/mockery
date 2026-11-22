from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mockery.llm import _extract_json_object, _head_tail_excerpt
from mockery.prompts import (
    practice_tone_instruction,
    probe_guidance_for,
    render_prompt,
    triage_diarize_rules,
    validate_prompt_template,
)

TARGET_TO_PROMPT_NAME: dict[str, str] = {
    "triage_window": "triage_window",
    "practice_interviewer": "practice_interviewer_reply",
    "suggest_probes": "suggest_probes",
    "practice_retry_eval": "practice_retry_eval",
    "practice_hint": "practice_hint",
    "practice_debrief": "practice_debrief",
}

CARL_DIMS: tuple[str, ...] = ("context", "action", "result", "learning")
CARL_RANK: dict[str, int] = {"missing": 0, "partial": 1, "strong": 2}
VALID_SPEAKERS = {"interviewer", "candidate"}
VALID_CARD_ACTIONS = {"keep_in_current", "open_new_card", "ignore_backchannel"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_NUDGE_TONES = {"encourage", "tip"}
VALID_SPLITS = {"train", "val"}

HARSH_TONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byou\s+failed\b", re.I),
    re.compile(r"\bterrible\b", re.I),
    re.compile(r"\bcompletely\s+wrong\b", re.I),
    re.compile(r"\bweak\s+answer\b", re.I),
    re.compile(r"\bpoor\s+answer\b", re.I),
    re.compile(r"\binadequate\b", re.I),
    re.compile(r"\bunacceptable\b", re.I),
    re.compile(r"\bnot\s+good\s+enough\b", re.I),
)

WORD_RE = re.compile(r"[a-z0-9']+", re.I)
SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")


def load_jsonl_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {path}")
            rows.append(item)
    return rows


def validate_dataset(target: str, rows: list[dict[str, Any]]) -> None:
    """Validate dataset schema, unique IDs, split labels, and target-specific fields before model calls."""
    if target not in TARGET_TO_PROMPT_NAME:
        raise ValueError(f"Unsupported eval target: {target!r}")
    if not rows:
        raise ValueError("Dataset is empty; expected at least one train and one val example.")

    seen_ids: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Row {idx} is missing a non-empty string `id`.")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate example `id` {row_id!r} in dataset.")
        seen_ids.add(row_id)

        split = row.get("split")
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Row {row_id!r} has invalid `split`={split!r}; expected one of {sorted(VALID_SPLITS)}."
            )

        expected = row.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"Row {row_id!r} is missing required object `expected`.")

        if target == "triage_window":
            if not isinstance(row.get("diarize"), bool):
                raise ValueError(f"Row {row_id!r} requires boolean `diarize`.")
            if not isinstance(row.get("session_index"), dict):
                raise ValueError(f"Row {row_id!r} requires object `session_index`.")
            incoming = row.get("incoming_utterances")
            if not isinstance(incoming, list) or not incoming:
                raise ValueError(f"Row {row_id!r} requires non-empty list `incoming_utterances`.")
            incoming_ids: set[str] = set()
            for u in incoming:
                if (
                    not isinstance(u, dict)
                    or not isinstance(u.get("id"), str)
                    or not u["id"].strip()
                    or u.get("current_speaker") not in VALID_SPEAKERS
                    or not isinstance(u.get("text"), str)
                ):
                    raise ValueError(f"Row {row_id!r} has invalid entry in `incoming_utterances`.")
                incoming_ids.add(u["id"])
            exp_plans = expected.get("plans")
            if not isinstance(exp_plans, list) or not exp_plans:
                raise ValueError(f"Row {row_id!r} requires non-empty `expected.plans`.")
            for ep in exp_plans:
                if (
                    not isinstance(ep, dict)
                    or ep.get("utterance_id") not in incoming_ids
                    or ep.get("card_action") not in VALID_CARD_ACTIONS
                    or not isinstance(ep.get("expected_segments"), list)
                    or not ep["expected_segments"]
                ):
                    raise ValueError(f"Row {row_id!r} has invalid entry in `expected.plans`.")
                for seg in ep["expected_segments"]:
                    if not isinstance(seg, dict) or seg.get("speaker") not in VALID_SPEAKERS:
                        raise ValueError(f"Row {row_id!r} has invalid `expected_segments` entry.")
        elif target == "practice_interviewer":
            for req_str in ("role_title", "interview_type", "target_level", "candidate_answer"):
                if not isinstance(row.get(req_str), str) or not row[req_str].strip():
                    raise ValueError(f"Row {row_id!r} requires non-empty string `{req_str}`.")
            cur_q = row.get("current_question")
            if not isinstance(cur_q, dict) or not isinstance(cur_q.get("prompt"), str) or not cur_q["prompt"].strip():
                raise ValueError(f"Row {row_id!r} requires `current_question.prompt`.")
            if not isinstance(row.get("follow_up_count"), int) or isinstance(row.get("follow_up_count"), bool):
                raise ValueError(f"Row {row_id!r} requires integer `follow_up_count`.")
            existing_carl = row.get("existing_carl")
            if not isinstance(existing_carl, dict) or any(
                existing_carl.get(d) not in CARL_RANK for d in CARL_DIMS
            ):
                raise ValueError(f"Row {row_id!r} requires valid `existing_carl` for all 4 CARL dims.")
            if not isinstance(expected.get("should_advance"), bool):
                raise ValueError(f"Row {row_id!r} requires boolean `expected.should_advance`.")
            exp_carl = expected.get("expected_carl")
            if not isinstance(exp_carl, dict) or any(
                exp_carl.get(d) not in CARL_RANK for d in CARL_DIMS
            ):
                raise ValueError(f"Row {row_id!r} requires valid `expected.expected_carl` for all 4 CARL dims.")
        elif target == "suggest_probes":
            for req_str in ("interview_type", "role_title", "card_title", "transcript"):
                if not isinstance(row.get(req_str), str) or not row[req_str].strip():
                    raise ValueError(f"Row {row_id!r} requires non-empty string `{req_str}`.")
        elif target == "practice_retry_eval":
            for req_str in ("role_title", "target_level", "drill_focus", "drill_prompt", "original_answer", "retry_answer"):
                if not isinstance(row.get(req_str), str) or not row[req_str].strip():
                    raise ValueError(f"Row {row_id!r} requires non-empty string `{req_str}`.")
            exp_ca = expected.get("expected_carl_after")
            if not isinstance(exp_ca, dict) or any(exp_ca.get(d) not in CARL_RANK for d in CARL_DIMS):
                raise ValueError(f"Row {row_id!r} requires valid `expected.expected_carl_after` for all 4 CARL dims.")
        elif target == "practice_hint":
            for req_str in ("interview_type", "target_level"):
                if not isinstance(row.get(req_str), str) or not row[req_str].strip():
                    raise ValueError(f"Row {row_id!r} requires non-empty string `{req_str}`.")
            cur_q = row.get("current_question")
            if not isinstance(cur_q, dict) or not isinstance(cur_q.get("prompt"), str) or not cur_q["prompt"].strip():
                raise ValueError(f"Row {row_id!r} requires `current_question.prompt`.")
        elif target == "practice_debrief":
            for req_str in ("role_title", "interview_type", "target_level"):
                if not isinstance(row.get(req_str), str) or not row[req_str].strip():
                    raise ValueError(f"Row {row_id!r} requires non-empty string `{req_str}`.")
            qs = row.get("questions")
            turns = row.get("turns")
            if not isinstance(qs, list) or not qs:
                raise ValueError(f"Row {row_id!r} requires non-empty `questions` list.")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"Row {row_id!r} requires non-empty `turns` list.")
            exp_qids = expected.get("question_ids")
            if not isinstance(exp_qids, list) or not exp_qids:
                raise ValueError(f"Row {row_id!r} requires non-empty `expected.question_ids`.")


def split_dataset(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition dataset into strict, disjoint, non-empty train and val splits."""
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in rows:
        split = row.get("split")
        if split == "train":
            train.append(row)
        elif split == "val":
            val.append(row)
        else:
            raise ValueError(
                f"Unrecognized dataset split {split!r} on row {row.get('id')!r}; expected 'train' or 'val'."
            )
    if not train or not val:
        raise ValueError(
            f"Dataset must contain non-empty disjoint train and val splits (got train={len(train)}, val={len(val)})."
        )
    return train, val


def validate_candidate_prompt(target: str, candidate_text: str) -> tuple[bool, list[str]]:
    prompt_name = TARGET_TO_PROMPT_NAME[target]
    if not isinstance(candidate_text, str) or len(candidate_text.strip()) < 40:
        return False, ["Candidate prompt must be a non-empty instruction string (>= 40 chars)."]
    errors = validate_prompt_template(prompt_name, candidate_text)
    return (len(errors) == 0), errors


def render_system_prompt_for_example(
    target: str,
    example: dict[str, Any],
    *,
    override_text: str | None = None,
) -> str:
    if target == "triage_window":
        diarize = bool(example.get("diarize", True))
        return render_prompt(
            "triage_window",
            override_text=override_text,
            diarize_rules=triage_diarize_rules(diarize),
        )
    if target == "practice_interviewer":
        interview_type = str(example.get("interview_type") or "behavioral")
        coach_style = str(example.get("coach_style") or "supportive")
        return render_prompt(
            "practice_interviewer_reply",
            override_text=override_text,
            interview_type_label=interview_type.replace("_", " "),
            role_title=str(example.get("role_title") or "Software Engineer"),
            target_level=str(example.get("target_level") or "Senior (L5)"),
            tone_instruction=practice_tone_instruction(coach_style),
        )
    if target == "suggest_probes":
        interview_type = str(example.get("interview_type") or "behavioral")
        return render_prompt(
            "suggest_probes",
            override_text=override_text,
            probe_guidance=probe_guidance_for(interview_type),
        )
    if target == "practice_retry_eval":
        return render_prompt("practice_retry_eval", override_text=override_text)
    if target == "practice_hint":
        return render_prompt("practice_hint", override_text=override_text)
    if target == "practice_debrief":
        return render_prompt("practice_debrief", override_text=override_text)
    raise ValueError(f"Unsupported eval target: {target!r}")


def build_triage_user_prompt(example: dict[str, Any]) -> str:
    """Matches the exact JSON payload built by InterviewCopilotLLM.triage_window."""
    return json.dumps(
        {
            "session_index": example["session_index"],
            "incoming_utterances": example["incoming_utterances"],
        },
        ensure_ascii=False,
    )


def build_practice_interviewer_user_prompt(example: dict[str, Any]) -> str:
    """Matches the exact user prompt built by InterviewCopilotLLM.generate_practice_interviewer_reply."""
    interview_type = str(example.get("interview_type") or "behavioral")
    max_followups = 3 if interview_type in ("system_design", "coding") else 2
    current_question = example.get("current_question") or {}
    next_question = example.get("next_question")
    recent_turns = example.get("recent_turns") or []
    follow_ups = list(current_question.get("follow_ups") or [])
    follow_up_count = int(example.get("follow_up_count") or 0)
    candidate_answer = str(example.get("candidate_answer") or "")
    workspace_context = example.get("workspace_context")

    history_lines = [
        f"{'INTERVIEWER' if t.get('role') == 'interviewer' else 'CANDIDATE'}: {t.get('text', '')}"
        for t in recent_turns[-8:]
        if t.get("role") in ("interviewer", "candidate")
    ]
    ws_block = ""
    if isinstance(workspace_context, dict):
        if workspace_context.get("diagram_summary"):
            ws_block += (
                "\nLIVE SYSTEM DESIGN DIAGRAM STATE:\n"
                f"{str(workspace_context['diagram_summary'])[:2500]}\n"
            )
        if workspace_context.get("code"):
            ws_block += (
                f"\nLIVE CODING WORKSPACE ({workspace_context.get('language', 'code')}):\n"
                f"```\n{str(workspace_context['code'])[:3000]}\n```\n"
            )
        if workspace_context.get("test_output"):
            ws_block += f"\nLATEST TEST RUN OUTPUT:\n{str(workspace_context['test_output'])[:1200]}\n"

    next_q_block = (
        f"Next planned question (ONLY introduce if should_advance=true): {next_question.get('prompt')}"
        if isinstance(next_question, dict) and next_question.get("prompt")
        else "This is the FINAL question of the interview. If should_advance=true, wrap up warmly and invite them to view their CARL Coaching Debrief."
    )

    return (
        f"Current Question: {current_question.get('prompt')}\n"
        f"CARL / Rubric Focus: {current_question.get('carl_focus') or ', '.join(current_question.get('core_requirements') or [])}\n"
        f"Follow-ups already asked on this question: {follow_up_count} (advance once you've asked {max_followups} follow-ups OR they've already given strong complete signal).\n"
        f"Suggested follow-up angles: {follow_ups}\n"
        f"{next_q_block}\n"
        f"{ws_block}\n"
        "Recent conversation:\n"
        + "\n".join(history_lines)
        + "\n\n"
        f"Candidate's latest answer:\n{candidate_answer}"
    )


def build_suggest_probes_user_prompt(example: dict[str, Any]) -> str:
    """Matches the exact user prompt built by InterviewCopilotLLM.suggest_probes."""
    already_asked = list(example.get("already_asked") or [])
    prior_suggested = list(example.get("prior_suggested") or [])
    do_not_repeat = (already_asked[-6:] + prior_suggested[-6:])[-10:]
    avoid_block = ""
    if do_not_repeat:
        avoid_block = (
            "\nQuestions already asked or suggested (DO NOT repeat or paraphrase these):\n- "
            + "\n- ".join(do_not_repeat)
        )
    return (
        f"Target Role: {example.get('role_title')} | Focus: {example.get('interview_type')}\n"
        f"Rubric Context: {example.get('rubric_notes') or 'Standard pragmatic engineering bar'}\n\n"
        f"Active Card ({example.get('card_badge', 'Q1')}: {example.get('card_title')}):\n{example.get('transcript')}"
        f"{avoid_block}"
    )


def build_practice_retry_eval_user_prompt(example: dict[str, Any]) -> str:
    """Matches the exact user prompt built by InterviewCopilotLLM.evaluate_practice_retry."""
    return (
        f"Role: {example.get('role_title')} ({example.get('target_level')})\n"
        f"Retry Drill Focus: {example.get('drill_focus')}\nDrill Prompt: {example.get('drill_prompt')}\n\n"
        f"Original Answer:\n{str(example.get('original_answer') or '')[:2500]}\n\n"
        f"45-Second Retry Answer:\n{str(example.get('retry_answer') or '')[:2500]}"
    )


def build_practice_hint_user_prompt(example: dict[str, Any]) -> str:
    """Matches the exact user prompt built by InterviewCopilotLLM.generate_practice_hint."""
    current_question = example.get("current_question") or {}
    recent_turns = example.get("recent_turns") or []
    history = "\n".join(
        f"{str(t.get('role', '')).upper()}: {t.get('text', '')}"
        for t in recent_turns[-5:]
        if t.get("role") in ("interviewer", "candidate")
    )
    ws_note = ""
    ws = example.get("workspace_context")
    if isinstance(ws, dict):
        if ws.get("diagram_summary"):
            ws_note += f"\nCurrent diagram:\n{str(ws['diagram_summary'])[:1500]}"
        if ws.get("code"):
            ws_note += f"\nCurrent code:\n{str(ws['code'])[:1500]}"
    return (
        f"Format: {example.get('interview_type')} ({example.get('target_level')})\n"
        f"Question: {current_question.get('prompt')}\n"
        f"Focus: {current_question.get('carl_focus') or ', '.join(current_question.get('core_requirements') or [])}\n"
        f"Recent exchange:\n{history}{ws_note}"
    )


def build_practice_debrief_user_prompt(example: dict[str, Any]) -> str:
    """Matches the exact user prompt built by InterviewCopilotLLM.generate_practice_debrief."""
    role_title = str(example.get("role_title") or "Software Engineer")
    company = str(example.get("company") or "").strip()
    target_level = str(example.get("target_level") or "Senior (L5)")
    interview_type = str(example.get("interview_type") or "behavioral")
    questions = list(example.get("questions") or [])
    turns = list(example.get("turns") or [])
    workspace_context = example.get("workspace_context") or {}

    per_q_budget = max(1200, 10_000 // max(1, len(questions)))
    transcript_sections: list[str] = []
    for q in questions:
        qid = str(q.get("id") or "")
        q_lines = [
            f"{str(t.get('role') or '').upper()}: {t.get('text', '')}"
            for t in turns
            if str(t.get("question_id") or "") == qid and t.get("role") in ("interviewer", "candidate")
        ]
        excerpt = _head_tail_excerpt("\n".join(q_lines), per_q_budget)
        transcript_sections.append(f"### [{qid}] {q.get('title', 'Question')}\n{excerpt}")

    ws_block = ""
    if isinstance(workspace_context, dict):
        if workspace_context.get("diagram_summary"):
            ws_block += f"\nFINAL DIAGRAM:\n{str(workspace_context['diagram_summary'])[:2000]}\n"
        if workspace_context.get("code"):
            ws_block += f"\nFINAL CODE:\n{str(workspace_context['code'])[:2500]}\n"
        if workspace_context.get("test_output"):
            ws_block += f"\nTEST RESULTS:\n{str(workspace_context['test_output'])[:1000]}\n"

    return (
        f"Role: {role_title}" + (f" at {company}" if company else "") + f" | Target Level: {target_level} | Format: {interview_type}\n"
        f"Answered Questions:\n"
        + "\n".join(f"- {q.get('id')}: {q.get('title')} — {q.get('prompt')}" for q in questions)
        + "\n\nPer-Question Transcript Excerpts:\n"
        + "\n\n".join(transcript_sections)
        + ws_block
    )


def build_user_prompt_for_example(target: str, example: dict[str, Any]) -> str:
    if target == "triage_window":
        return build_triage_user_prompt(example)
    if target == "practice_interviewer":
        return build_practice_interviewer_user_prompt(example)
    if target == "suggest_probes":
        return build_suggest_probes_user_prompt(example)
    if target == "practice_retry_eval":
        return build_practice_retry_eval_user_prompt(example)
    if target == "practice_hint":
        return build_practice_hint_user_prompt(example)
    if target == "practice_debrief":
        return build_practice_debrief_user_prompt(example)
    raise ValueError(f"Unsupported eval target: {target!r}")


def _parse_json_with_strictness(raw_output: str) -> tuple[dict[str, Any] | None, float, str | None]:
    stripped = (raw_output or "").strip()
    if not stripped:
        return None, 0.0, "Empty response; expected raw JSON object."
    try:
        parsed_strict = json.loads(stripped)
        if isinstance(parsed_strict, dict):
            return parsed_strict, 1.0, None
    except json.JSONDecodeError:
        pass
    try:
        recovered = _extract_json_object(stripped)
        if isinstance(recovered, dict):
            return (
                recovered,
                0.75,
                "Response wrapped JSON in prose or markdown fences; emit raw JSON only.",
            )
    except Exception as exc:
        return None, 0.0, f"Invalid JSON output: {exc}"
    return None, 0.0, "JSON output was not a JSON object."


def _word_tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _text_fidelity_score(source_text: str, output_text: str) -> tuple[float, float]:
    """
    Order-aware and multiplicity-aware transcript fidelity score.
    Catches dropped words, fabricated insertions before/after, and scrambled order,
    while allowing small unmistakable STT proper-noun fixes (~10-15% edits).
    Returns `(fidelity_score, raw_similarity)`.
    """
    src_tokens = _word_tokens(source_text)
    out_tokens = _word_tokens(output_text)
    if not src_tokens:
        return (1.0 if not out_tokens else 0.0), (1.0 if not out_tokens else 0.0)
    if not out_tokens:
        return 0.0, 0.0

    # Multiset precision & recall catch insertions and dropped repetitions
    src_counts = Counter(src_tokens)
    out_counts = Counter(out_tokens)
    overlap = sum((src_counts & out_counts).values())
    precision = overlap / len(out_tokens)
    recall = overlap / len(src_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # SequenceMatcher ratio catches reordering and inserted spans
    seq_ratio = difflib.SequenceMatcher(None, src_tokens, out_tokens).ratio()
    combined = min(precision, 0.5 * f1 + 0.5 * seq_ratio)

    # Allow minor STT proper-noun fixes (e.g. 'for deployed' -> 'forward deployed', 'red is' -> 'Redis' ~0.91)
    # while always flagging fabricated insertions or paraphrasing.
    threshold = 0.90
    fidelity = 1.0 if combined >= threshold else max(0.0, combined / threshold)
    return fidelity, combined


def _count_sentences(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    ends = SENTENCE_END_RE.findall(cleaned)
    if not ends:
        return 1
    # If text continues meaningfully after the last punctuation mark, count trailing clause
    last_match = list(SENTENCE_END_RE.finditer(cleaned))[-1]
    trailing = cleaned[last_match.end() :].strip()
    return len(ends) + (1 if len(trailing.split()) >= 3 else 0)


def score_triage_window(
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Deterministic scorer for `triage_window`.
    Returns `(score, side_info)` where `score` is in [0.0, 1.0] and `side_info`
    carries Actionable Side Information (ASI) for GEPA reflection.
    """
    failures: list[str] = []
    suggestions: list[str] = []

    parsed, json_score, json_note = _parse_json_with_strictness(raw_output)
    if json_note:
        failures.append(json_note)
        if parsed is not None:
            suggestions.append("Reinforce emitting raw JSON with zero markdown fences or preamble.")
    if parsed is None:
        return 0.0, {
            "example_id": example.get("id"),
            "subscores": {"json_validity": 0.0},
            "failures": failures,
            "suggestions": ["Require strict JSON object output matching the `plans` schema."],
        }

    raw_plans = parsed.get("plans")
    if not isinstance(raw_plans, list):
        failures.append("Missing top-level `plans` list in JSON response.")
        return round(0.1 * json_score, 4), {
            "example_id": example.get("id"),
            "subscores": {"json_validity": json_score, "schema": 0.0},
            "failures": failures,
            "suggestions": ["Always wrap output items inside {\"plans\": [...]}."],
        }

    incoming_by_id = {u["id"]: u for u in example.get("incoming_utterances") or []}
    expected_plans = (example.get("expected") or {}).get("plans") or []
    diarize = bool(example.get("diarize", True))

    actual_by_id: dict[str, dict[str, Any]] = {}
    extra_ids: list[str] = []
    duplicate_ids: list[str] = []
    malformed_entries = 0

    for item in raw_plans:
        if not isinstance(item, dict):
            malformed_entries += 1
            failures.append("Non-object entry found inside `plans`.")
            continue
        raw_uid = item.get("utterance_id")
        uid = raw_uid.strip() if isinstance(raw_uid, str) else ""
        if uid in incoming_by_id:
            if uid in actual_by_id:
                duplicate_ids.append(uid)
                failures.append(f"Duplicate plan entry for utterance_id={uid!r}.")
            else:
                actual_by_id[uid] = item
        else:
            extra_ids.append(uid or "<empty>")

    if extra_ids:
        failures.append(f"Unknown or missing utterance_id entries: {extra_ids}")
        suggestions.append("Copy `utterance_id` verbatim from `incoming_utterances` with exactly one plan per utterance.")
    if duplicate_ids:
        suggestions.append("Emit exactly one plan object per incoming `utterance_id` without duplicates.")
    if malformed_entries:
        suggestions.append("Every item in `plans` must be a valid JSON object matching the plan schema.")

    schema_scores: list[float] = []
    speaker_scores: list[float] = []
    content_scores: list[float] = []
    card_scores: list[float] = []
    conf_scores: list[float] = []

    for exp in expected_plans:
        uid = exp["utterance_id"]
        orig = incoming_by_id.get(uid, {})
        actual = actual_by_id.get(uid)
        if actual is None:
            failures.append(f"Missing plan for utterance_id={uid!r}.")
            schema_scores.append(0.0)
            speaker_scores.append(0.0)
            content_scores.append(0.0)
            card_scores.append(0.0)
            conf_scores.append(0.0)
            continue

        # 1. Strict schema conformance for this plan
        segs = actual.get("segments")
        raw_action = actual.get("card_action")
        raw_conf = actual.get("confidence")
        has_title_field = "suggested_title" in actual and isinstance(actual.get("suggested_title"), str)
        raw_rationale = actual.get("rationale")

        action = raw_action.strip() if isinstance(raw_action, str) else ""
        conf = raw_conf.strip().lower() if isinstance(raw_conf, str) else ""
        title = actual["suggested_title"].strip() if has_title_field else ""
        rationale = raw_rationale.strip() if isinstance(raw_rationale, str) else ""

        valid_segs: list[dict[str, str]] = []
        if isinstance(segs, list):
            for s in segs:
                if (
                    isinstance(s, dict)
                    and isinstance(s.get("speaker"), str)
                    and s["speaker"].strip().lower() in VALID_SPEAKERS
                    and isinstance(s.get("text"), str)
                    and s["text"].strip()
                ):
                    valid_segs.append(
                        {
                            "speaker": s["speaker"].strip().lower(),
                            "text": s["text"].strip(),
                        }
                    )

        schema_checks = [
            bool(valid_segs) and isinstance(segs, list) and len(valid_segs) == len(segs),
            action in VALID_CARD_ACTIONS,
            conf in VALID_CONFIDENCES,
            has_title_field,
            bool(rationale) and len(rationale.split()) <= 24,
        ]
        schema_scores.append(sum(1.0 for ok in schema_checks if ok) / len(schema_checks))
        if not schema_checks[0]:
            failures.append(f"[{uid}] Invalid or empty `segments` items.")
        if not schema_checks[1]:
            failures.append(f"[{uid}] Invalid `card_action`={raw_action!r}.")
        if not schema_checks[2]:
            failures.append(f"[{uid}] Invalid `confidence`={raw_conf!r}.")
        if not has_title_field:
            failures.append(f"[{uid}] Missing required string field `suggested_title`.")
        if not schema_checks[4]:
            failures.append(f"[{uid}] Missing or overly long `rationale`.")

        # 2. Speaker & rapid-handoff segmentation accuracy
        exp_segs = exp.get("expected_segments") or []
        exp_speakers = [s["speaker"] for s in exp_segs]
        act_speakers = [s["speaker"] for s in valid_segs]

        if not diarize:
            hw_ok = (
                len(valid_segs) == 1
                and valid_segs[0]["speaker"] == orig.get("current_speaker")
                and valid_segs[0]["text"].strip() == str(orig.get("text") or "").strip()
            )
            speaker_scores.append(1.0 if hw_ok else 0.0)
            if not hw_ok:
                failures.append(
                    f"[{uid}] Hardware-separated mode (`diarize=False`) mutated speaker or text."
                )
                suggestions.append(
                    "When speaker labels are hardware-separated, keep exactly 1 segment with original speaker and verbatim text."
                )
        else:
            if act_speakers == exp_speakers:
                speaker_scores.append(1.0)
            elif valid_segs and len(act_speakers) == len(exp_speakers):
                matches = sum(1 for a, b in zip(act_speakers, exp_speakers) if a == b)
                speaker_scores.append(matches / len(exp_speakers))
                failures.append(
                    f"[{uid}] Speaker sequence mismatch: expected {exp_speakers}, got {act_speakers}."
                )
            else:
                overlap = len(set(act_speakers) & set(exp_speakers)) / max(1, len(set(exp_speakers)))
                speaker_scores.append(round(0.25 * overlap, 4))
                failures.append(
                    f"[{uid}] Segment split mismatch: expected {len(exp_speakers)} segment(s) {exp_speakers}, "
                    f"got {len(act_speakers)} {act_speakers}."
                )
                if len(exp_speakers) > 1:
                    suggestions.append(
                        "Split rapid mixed-speaker handoffs cleanly at the exact sentence boundary where speaker role switches."
                    )

        # 3. Order- & multiplicity-aware content fidelity + expected phrase placement
        combined_out_text = " ".join(s["text"] for s in valid_segs)
        fidelity_score, raw_sim = _text_fidelity_score(str(orig.get("text") or ""), combined_out_text)
        phrase_hits = 0
        phrase_total = 0
        for idx, es in enumerate(exp_segs):
            target_seg_text = (
                valid_segs[idx]["text"].lower()
                if idx < len(valid_segs)
                else combined_out_text.lower()
            )
            for phrase in es.get("must_contain") or []:
                phrase_total += 1
                if phrase.lower() in target_seg_text:
                    phrase_hits += 1
                else:
                    failures.append(
                        f"[{uid}] Expected segment {idx} ({es['speaker']}) to contain {phrase!r}."
                    )
        phrase_ratio = (phrase_hits / phrase_total) if phrase_total else 1.0
        if raw_sim < 0.90:
            failures.append(
                f"[{uid}] Transcript fidelity drift: added/dropped/reordered words (similarity={raw_sim:.2f})."
            )
            suggestions.append(
                "Preserve speaker words verbatim in order without adding fabricated text or paraphrasing."
            )
        content_scores.append(0.55 * phrase_ratio + 0.45 * fidelity_score)

        # 4. Card boundary & title accuracy
        exp_action = exp["card_action"]
        require_title = bool(exp.get("require_title", False))
        action_ok = action == exp_action
        if not action_ok:
            failures.append(
                f"[{uid}] Card action mismatch: expected {exp_action!r}, got {action!r}."
            )
            if exp_action == "keep_in_current" and action == "open_new_card":
                suggestions.append(
                    "Do NOT open a new card on follow-up probes or clarifications—reserve `open_new_card` for brand-new primary questions or major design phases."
                )
            elif exp_action == "ignore_backchannel":
                suggestions.append(
                    "Mark pure 1-3 word filler acknowledgements ('Yeah, totally.', 'Uh-huh.') as `ignore_backchannel`."
                )

        if require_title:
            title_words = len(title.split())
            title_ok = has_title_field and 3 <= title_words <= 10
            if not title_ok:
                failures.append(
                    f"[{uid}] Expected 3-10 word `suggested_title` for `open_new_card`, got {title!r}."
                )
        else:
            title_ok = has_title_field and title == ""
            if not title_ok:
                failures.append(
                    f"[{uid}] `suggested_title` must be present and empty unless `card_action == 'open_new_card'`."
                )
        card_scores.append((0.75 if action_ok else 0.0) + (0.25 if title_ok else 0.0))

        # 5. Confidence calibration
        allowed_conf = set(exp.get("allowed_confidences") or ["high", "medium", "low"])
        if conf in allowed_conf:
            conf_scores.append(1.0)
        elif conf in VALID_CONFIDENCES:
            conf_scores.append(0.4)
            failures.append(
                f"[{uid}] Confidence {conf!r} outside expected {sorted(allowed_conf)}."
            )
        else:
            conf_scores.append(0.0)

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    structural_issues = len(extra_ids) + len(duplicate_ids) + malformed_entries
    structural_penalty = min(0.35, 0.10 * structural_issues)

    subscores = {
        "json_validity": round(json_score, 4),
        "schema": round(_avg(schema_scores), 4),
        "speaker_diarization": round(_avg(speaker_scores), 4),
        "content_preservation": round(_avg(content_scores), 4),
        "card_boundary": round(_avg(card_scores), 4),
        "confidence": round(_avg(conf_scores), 4),
        "structural_penalty": round(structural_penalty, 4),
    }

    total = (
        0.10 * subscores["json_validity"]
        + 0.10 * subscores["schema"]
        + 0.30 * subscores["speaker_diarization"]
        + 0.15 * subscores["content_preservation"]
        + 0.27 * subscores["card_boundary"]
        + 0.08 * subscores["confidence"]
        - structural_penalty
    )
    score = round(max(0.0, min(1.0, total)), 4)

    return score, {
        "example_id": example.get("id"),
        "subscores": subscores,
        "failures": failures,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def score_practice_interviewer_reply(
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Deterministic scorer for `practice_interviewer_reply`.
    Returns `(score, side_info)` with Actionable Side Information (ASI) covering:
    JSON/schema validity, `should_advance` decision, CARL accuracy + monotonicity,
    context grounding, spoken-reply conciseness (1-3 sentences), supportive tone,
    and code + prose algorithmic spoiler guards.
    """
    failures: list[str] = []
    suggestions: list[str] = []

    parsed, json_score, json_note = _parse_json_with_strictness(raw_output)
    if json_note:
        failures.append(json_note)
        if parsed is not None:
            suggestions.append("Emit raw JSON only without markdown fences or extra text.")
    if parsed is None:
        return 0.0, {
            "example_id": example.get("id"),
            "subscores": {"json_validity": 0.0},
            "failures": failures,
            "suggestions": ["Require strict JSON output matching the practice interviewer schema."],
        }

    expected = example.get("expected") or {}
    existing_carl = example.get("existing_carl") or {}

    raw_reply = parsed.get("interviewer_reply")
    raw_adv = parsed.get("should_advance")
    raw_carl = parsed.get("carl_signals") if isinstance(parsed.get("carl_signals"), dict) else {}
    raw_nudge = parsed.get("coach_nudge") if isinstance(parsed.get("coach_nudge"), dict) else {}
    raw_nudge_tone = raw_nudge.get("tone")
    raw_nudge_msg = raw_nudge.get("message")

    reply_is_str = isinstance(raw_reply, str)
    nudge_msg_is_str = isinstance(raw_nudge_msg, str)
    nudge_tone_is_str = isinstance(raw_nudge_tone, str)

    reply = raw_reply.strip() if reply_is_str else ""
    nudge_tone = raw_nudge_tone.strip().lower() if nudge_tone_is_str else ""
    nudge_msg = raw_nudge_msg.strip() if nudge_msg_is_str else ""

    # 1. Strict schema conformance (no silent coercion of non-strings)
    carl_valid_dims = sum(
        1
        for d in CARL_DIMS
        if isinstance(raw_carl.get(d), str) and raw_carl[d].strip().lower() in CARL_RANK
    )
    schema_checks = [
        reply_is_str and bool(reply),
        isinstance(raw_adv, bool),
        carl_valid_dims == 4,
        nudge_tone_is_str and nudge_tone in VALID_NUDGE_TONES,
        nudge_msg_is_str and bool(nudge_msg),
    ]
    schema_score = sum(1.0 for ok in schema_checks if ok) / len(schema_checks)
    if not reply_is_str:
        failures.append(f"`interviewer_reply` must be a string, got {type(raw_reply).__name__}.")
    elif not reply:
        failures.append("Empty `interviewer_reply`.")
    if not isinstance(raw_adv, bool):
        failures.append(f"`should_advance` must be a boolean, got {type(raw_adv).__name__}.")
    if carl_valid_dims < 4:
        failures.append(f"`carl_signals` missing or invalid dimensions (valid={carl_valid_dims}/4).")
    if not nudge_tone_is_str or nudge_tone not in VALID_NUDGE_TONES:
        failures.append(f"`coach_nudge.tone` must be 'encourage' or 'tip', got {raw_nudge_tone!r}.")
    if not nudge_msg_is_str:
        failures.append(f"`coach_nudge.message` must be a string, got {type(raw_nudge_msg).__name__}.")
    elif not nudge_msg:
        failures.append("Empty `coach_nudge.message`.")

    # 2. Advance decision accuracy
    exp_adv = bool(expected.get("should_advance", False))
    pred_adv: bool | None = raw_adv if isinstance(raw_adv, bool) else None
    advance_score = 1.0 if pred_adv is exp_adv else 0.0
    if pred_adv is not exp_adv:
        failures.append(
            f"`should_advance` mismatch: expected {exp_adv}, got {pred_adv!r} "
            f"(follow_up_count={example.get('follow_up_count')})."
        )
        if exp_adv:
            suggestions.append(
                "Set `should_advance=true` once max follow-ups are reached or the candidate already gave complete CARL/technical signal."
            )
        else:
            suggestions.append(
                "Keep `should_advance=false` and ask one targeted follow-up when key CARL or technical dimensions are still missing."
            )

    # 3. CARL accuracy & strict monotonicity
    exp_carl = expected.get("expected_carl") or {}
    carl_acc_points: list[float] = []
    monotonic_violations: list[str] = []
    for dim in CARL_DIMS:
        raw_val = raw_carl.get(dim)
        pred_val = raw_val.strip().lower() if isinstance(raw_val, str) else ""
        exp_val = str(exp_carl.get(dim, "missing")).strip().lower()
        prev_val = str(existing_carl.get(dim, "missing")).strip().lower()
        if pred_val not in CARL_RANK:
            carl_acc_points.append(0.0)
            continue
        diff = abs(CARL_RANK[pred_val] - CARL_RANK.get(exp_val, 0))
        if diff == 0:
            carl_acc_points.append(1.0)
        elif diff == 1:
            carl_acc_points.append(0.5)
            failures.append(f"CARL `{dim}` slightly off: expected {exp_val!r}, got {pred_val!r}.")
        else:
            carl_acc_points.append(0.0)
            failures.append(f"CARL `{dim}` far off: expected {exp_val!r}, got {pred_val!r}.")
        if CARL_RANK[pred_val] < CARL_RANK.get(prev_val, 0):
            monotonic_violations.append(f"{dim}: {prev_val} -> {pred_val}")

    carl_acc = sum(carl_acc_points) / len(CARL_DIMS)
    monotonic_score = 1.0 if not monotonic_violations else max(0.0, 1.0 - 0.5 * len(monotonic_violations))
    if monotonic_violations:
        failures.append(
            "CARL monotonicity violated (regressed previously earned signal): "
            + ", ".join(monotonic_violations)
        )
        suggestions.append(
            "Never downgrade a CARL dimension below what the candidate already demonstrated in earlier turns."
        )
    carl_score = carl_acc * monotonic_score

    # 4. Spoken-turn naturalness, sentence & word bounds, and grounding
    reply_words = len(reply.split())
    sentence_count = _count_sentences(reply)
    min_words = int(expected.get("min_reply_words", 8))
    max_words = int(expected.get("max_reply_words", 85))
    max_sentences = int(expected.get("max_reply_sentences", 3))
    has_markdown_wall = bool(
        re.search(r"(^|\n)\s*(#{1,6}\s|```|[-*]\s)", reply)
    )
    length_ok = min_words <= reply_words <= max_words
    sentences_ok = 1 <= sentence_count <= max_sentences
    if not length_ok:
        failures.append(
            f"`interviewer_reply` length out of spoken bounds: {reply_words} words (expected {min_words}-{max_words})."
        )
        suggestions.append("Keep spoken `interviewer_reply` concise (1-3 natural spoken sentences).")
    if reply and not sentences_ok:
        failures.append(
            f"`interviewer_reply` has {sentence_count} sentences (expected 1-{max_sentences} spoken sentences)."
        )
        suggestions.append("Limit `interviewer_reply` to 1-3 crisp spoken sentences.")
    if has_markdown_wall:
        failures.append("`interviewer_reply` contains markdown headers, code fences, or bullet lists unsuitable for speech.")
        suggestions.append("Emit plain conversational prose in `interviewer_reply` (no markdown formatting).")

    must_ref = [str(k) for k in (expected.get("must_reference_any") or []) if str(k).strip()]
    reply_low = reply.lower()
    if must_ref:
        grounded = bool(reply) and any(tok.lower() in reply_low for tok in must_ref)
        if not grounded:
            failures.append(
                f"`interviewer_reply` did not ground in candidate/workspace details (expected one of {must_ref[:5]})."
            )
            suggestions.append(
                "Briefly acknowledge a specific detail from the candidate's answer, diagram, or test output before probing."
            )
    else:
        grounded = bool(reply)

    grounding_and_style = (
        (0.35 if grounded else 0.0)
        + (0.25 if length_ok else 0.05 if reply_words > 0 else 0.0)
        + (0.20 if sentences_ok else 0.0)
        + (0.20 if (reply and not has_markdown_wall) else 0.0)
    )

    # 5. Supportive tone & code/prose no-spoiler safety
    combined_text = f"{reply}\n{nudge_msg}"
    harsh_hits = [pat.pattern for pat in HARSH_TONE_PATTERNS if pat.search(combined_text)]
    forbidden_spoilers = [str(s) for s in (expected.get("forbidden_spoilers") or []) if str(s).strip()]
    spoiler_hits = [s for s in forbidden_spoilers if s.lower() in combined_text.lower()]
    for pat_str in expected.get("forbidden_spoiler_patterns") or []:
        if isinstance(pat_str, str) and pat_str.strip():
            if re.search(pat_str, combined_text, flags=re.IGNORECASE | re.DOTALL):
                spoiler_hits.append(f"pattern:{pat_str}")
    nudge_concise = nudge_msg_is_str and (4 <= len(nudge_msg.split()) <= 45)

    if harsh_hits:
        failures.append(f"Harsh or discouraging phrasing detected: {harsh_hits}")
        suggestions.append("Keep both interviewer reply and coach nudge warm, encouraging, and constructive.")
    if spoiler_hits:
        failures.append(f"Spoiler leaked solution algorithm or code: {spoiler_hits}")
        suggestions.append("Probe with guiding questions instead of giving away the exact algorithm or code.")
    if nudge_msg and not nudge_concise:
        failures.append(
            f"`coach_nudge.message` length ({len(nudge_msg.split())} words) should be 1 crisp coaching sentence (4-45 words)."
        )

    if not (reply_is_str and reply):
        tone_safety_score = 0.0
    elif harsh_hits:
        tone_safety_score = 0.0
    elif spoiler_hits:
        tone_safety_score = 0.0
    else:
        tone_safety_score = 0.80 + (0.20 if nudge_concise else 0.0)

    subscores = {
        "json_validity": round(json_score, 4),
        "schema": round(schema_score, 4),
        "should_advance": round(advance_score, 4),
        "carl_and_monotonicity": round(carl_score, 4),
        "grounding_and_style": round(grounding_and_style, 4),
        "tone_and_no_spoiler": round(tone_safety_score, 4),
    }

    total = (
        0.10 * subscores["json_validity"]
        + 0.15 * subscores["schema"]
        + 0.20 * subscores["should_advance"]
        + 0.22 * subscores["carl_and_monotonicity"]
        + 0.18 * subscores["grounding_and_style"]
        + 0.15 * subscores["tone_and_no_spoiler"]
    )
    score = round(max(0.0, min(1.0, total)), 4)

    return score, {
        "example_id": example.get("id"),
        "subscores": subscores,
        "failures": failures,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


_PRONOUN_POLICE_RE = re.compile(
    r"(?:you\s+said\s+['\"]?we['\"]?|['\"]?i['\"]?\s+(?:vs\.?|versus)\s+['\"]?we['\"]?|stop\s+saying\s+['\"]?we['\"]?|instead\s+of\s+['\"]?we['\"]?)",
    re.I,
)
_MATH_PEDANTRY_RE = re.compile(
    r"(?:back[- ]of[- ]the[- ]envelope|exact\s+(?:bytes|qps|bandwidth|storage)\s+calculation|calculate\s+the\s+exact)",
    re.I,
)


def _token_jaccard(a: str, b: str) -> float:
    sa = {w for w in _word_tokens(a) if len(w) >= 3}
    sb = {w for w in _word_tokens(b) if len(w) >= 3}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_suggest_probes(
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Deterministic scorer for Live Copilot `suggest_probes`."""
    failures: list[str] = []
    suggestions: list[str] = []

    parsed, json_score, json_note = _parse_json_with_strictness(raw_output)
    if json_note:
        failures.append(json_note)
        if parsed is not None:
            suggestions.append("Emit raw JSON only without markdown fences or preamble.")
    if parsed is None:
        return 0.0, {
            "example_id": example.get("id"),
            "subscores": {"json_validity": 0.0},
            "failures": failures,
            "suggestions": ["Require strict JSON object matching quick_read + 3 probes."],
        }

    expected = example.get("expected") or {}
    raw_qr = parsed.get("quick_read")
    raw_probes = parsed.get("probes")
    qr = raw_qr.strip() if isinstance(raw_qr, str) else ""
    qr_ok = isinstance(raw_qr, str) and 5 <= len(qr.split()) <= 32 and _count_sentences(qr) <= 2

    valid_probes: list[dict[str, str]] = []
    if isinstance(raw_probes, list):
        for p in raw_probes:
            if (
                isinstance(p, dict)
                and isinstance(p.get("angle"), str)
                and isinstance(p.get("question"), str)
                and isinstance(p.get("why"), str)
                and p["angle"].strip()
                and p["question"].strip()
                and p["why"].strip()
            ):
                valid_probes.append(
                    {
                        "angle": p["angle"].strip(),
                        "question": p["question"].strip(),
                        "why": p["why"].strip(),
                    }
                )

    count_ok = isinstance(raw_probes, list) and len(raw_probes) == 3 and len(valid_probes) == 3
    if not qr_ok:
        failures.append(f"`quick_read` should be 1 crisp sentence (5-32 words), got {len(qr.split())} words.")
        suggestions.append("Keep `quick_read` to 1 crisp sentence summarizing what's solid vs where to probe next.")
    if not count_ok:
        failures.append(f"Expected exactly 3 valid probe objects, got {len(valid_probes)}.")
        suggestions.append("Always return exactly 3 probes, each with non-empty `angle`, `question`, and `why`.")

    schema_score = (0.35 if qr_ok else 0.0) + (0.65 if count_ok else 0.25 * min(3, len(valid_probes)) / 3.0)

    # Conciseness & question form (angle 1-4 words, question <= 24 words ending with ?, why <= 16 words)
    concise_pts: list[float] = []
    for idx, p in enumerate(valid_probes, 1):
        a_w = len(p["angle"].split())
        q_w = len(p["question"].split())
        w_w = len(p["why"].split())
        a_ok = 1 <= a_w <= 4
        q_ok = 5 <= q_w <= 24 and p["question"].endswith("?")
        w_ok = 3 <= w_w <= 16
        if not a_ok:
            failures.append(f"Probe {idx} `angle` ({a_w} words) should be 1-4 words: {p['angle']!r}.")
        if not q_ok:
            failures.append(f"Probe {idx} `question` ({q_w} words) must be <=24 words ending in '?': {p['question']!r}.")
            suggestions.append("Keep every probe `question` natural, under 24 words, and ending with a question mark.")
        if not w_ok:
            failures.append(f"Probe {idx} `why` ({w_w} words) should be 3-16 words.")
        concise_pts.append((0.25 if a_ok else 0.0) + (0.50 if q_ok else 0.0) + (0.25 if w_ok else 0.0))
    concise_score = sum(concise_pts) / max(3, len(concise_pts)) if concise_pts else 0.0

    # Non-repetition against already_asked / prior_suggested AND probe-to-probe diversity
    forbid_list = [str(x) for x in (expected.get("forbid_repeat_of") or []) if str(x).strip()]
    repeat_hits = 0
    for idx, p in enumerate(valid_probes, 1):
        for prev in forbid_list:
            if _token_jaccard(p["question"], prev) >= 0.48:
                repeat_hits += 1
                failures.append(f"Probe {idx} repeats already-asked/prior question: {p['question']!r} ~ {prev!r}.")
                suggestions.append("Check `Questions already asked or suggested` carefully and never repeat or closely paraphrase them.")
                break
    # Check pairwise diversity among the 3 probes
    dup_pair = False
    for i in range(len(valid_probes)):
        for j in range(i + 1, len(valid_probes)):
            if _token_jaccard(valid_probes[i]["question"], valid_probes[j]["question"]) >= 0.55:
                dup_pair = True
                failures.append(f"Probes {i+1} and {j+1} overlap too closely.")
    novelty_score = max(0.0, 1.0 - 0.35 * repeat_hits - (0.25 if dup_pair else 0.0)) if valid_probes else 0.0

    # Grounding in concrete transcript details
    all_text = (qr + " " + " ".join(f"{p['angle']} {p['question']} {p['why']}" for p in valid_probes)).lower()
    must_ref = [str(k).lower() for k in (expected.get("must_reference_any") or []) if str(k).strip()]
    hits = sum(1 for tok in must_ref if tok in all_text)
    if must_ref:
        grounding_score = 1.0 if hits >= 2 else (0.6 if hits == 1 else 0.0)
        if hits < 2:
            failures.append(f"Probes weakly grounded in transcript details (matched {hits} of {must_ref[:6]}).")
            suggestions.append("Anchor probes in specific systems, mechanisms, or claims the candidate just mentioned.")
    else:
        grounding_score = 1.0 if valid_probes else 0.0

    # Naturalness guards: no pronoun policing, no arithmetic pedantry on system design
    guard_ok = 1.0
    if expected.get("forbid_pronoun_policing") and _PRONOUN_POLICE_RE.search(all_text):
        guard_ok -= 0.6
        failures.append("Pronoun-policing phrasing ('I' vs 'we') detected.")
        suggestions.append("Never lecture or quiz the candidate about using 'we' instead of 'I'.")
    if expected.get("forbid_arithmetic_pedantry") and _MATH_PEDANTRY_RE.search(all_text):
        guard_ok -= 0.5
        failures.append("Forced back-of-the-envelope arithmetic pedantry detected on System Design.")
        suggestions.append("Focus System Design probes on bottlenecks, tradeoffs, and failure modes—not whiteboard arithmetic.")
    guard_score = max(0.0, guard_ok)

    subscores = {
        "json_validity": round(json_score, 4),
        "schema": round(schema_score, 4),
        "conciseness": round(concise_score, 4),
        "novelty": round(novelty_score, 4),
        "grounding": round(grounding_score, 4),
        "naturalness_guards": round(guard_score, 4),
    }
    total = (
        0.10 * subscores["json_validity"]
        + 0.15 * subscores["schema"]
        + 0.25 * subscores["conciseness"]
        + 0.20 * subscores["novelty"]
        + 0.18 * subscores["grounding"]
        + 0.12 * subscores["naturalness_guards"]
    )
    return round(max(0.0, min(1.0, total)), 4), {
        "example_id": example.get("id"),
        "subscores": subscores,
        "failures": failures,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def score_practice_retry_eval(
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Deterministic scorer for 45-second `practice_retry_eval` with anti-sycophancy grading checks."""
    failures: list[str] = []
    suggestions: list[str] = []

    parsed, json_score, json_note = _parse_json_with_strictness(raw_output)
    if json_note:
        failures.append(json_note)
        if parsed is not None:
            suggestions.append("Emit raw JSON only without markdown fences or prose.")
    if parsed is None:
        return 0.0, {
            "example_id": example.get("id"),
            "subscores": {"json_validity": 0.0},
            "failures": failures,
            "suggestions": ["Require strict JSON object matching practice_retry_eval schema."],
        }

    expected = example.get("expected") or {}
    raw_enc = parsed.get("encouragement")
    raw_imp = parsed.get("what_improved")
    raw_pol = parsed.get("one_more_polish")
    raw_ca = parsed.get("carl_after") if isinstance(parsed.get("carl_after"), dict) else {}

    enc = raw_enc.strip() if isinstance(raw_enc, str) else ""
    imp = raw_imp.strip() if isinstance(raw_imp, str) else ""
    pol = raw_pol.strip() if isinstance(raw_pol, str) else ""

    carl_valid = sum(
        1 for d in CARL_DIMS if isinstance(raw_ca.get(d), str) and raw_ca[d].strip().lower() in CARL_RANK
    )
    schema_checks = [
        isinstance(raw_enc, str) and bool(enc),
        isinstance(raw_imp, str) and bool(imp),
        isinstance(raw_pol, str),
        carl_valid == 4,
    ]
    schema_score = sum(1.0 for ok in schema_checks if ok) / len(schema_checks)
    if not schema_checks[0]:
        failures.append("Missing or non-string `encouragement`.")
    if not schema_checks[1]:
        failures.append("Missing or non-string `what_improved`.")
    if not schema_checks[2]:
        failures.append("`one_more_polish` must be a string.")
    if carl_valid < 4:
        failures.append(f"`carl_after` invalid/missing dimensions ({carl_valid}/4).")

    # CARL accuracy + anti-sycophancy check (penalize falsely inflating a missing/partial dim to strong!)
    exp_ca = expected.get("expected_carl_after") or {}
    must_not_inflate = set(expected.get("must_not_inflate") or [])
    acc_pts: list[float] = []
    inflated: list[str] = []
    for dim in CARL_DIMS:
        pv = raw_ca.get(dim).strip().lower() if isinstance(raw_ca.get(dim), str) else ""
        ev = str(exp_ca.get(dim, "missing")).strip().lower()
        if pv not in CARL_RANK:
            acc_pts.append(0.0)
            continue
        diff = abs(CARL_RANK[pv] - CARL_RANK[ev])
        if diff == 0:
            acc_pts.append(1.0)
        elif diff == 1:
            acc_pts.append(0.5)
            failures.append(f"`carl_after.{dim}` slightly off: expected {ev!r}, got {pv!r}.")
        else:
            acc_pts.append(0.0)
            failures.append(f"`carl_after.{dim}` far off: expected {ev!r}, got {pv!r}.")
        if dim in must_not_inflate and CARL_RANK[pv] > CARL_RANK[ev]:
            inflated.append(f"{dim}: expected {ev}, falsely upgraded to {pv}")

    carl_acc = sum(acc_pts) / len(CARL_DIMS)
    sycophancy_mult = 1.0 if not inflated else max(0.0, 1.0 - 0.45 * len(inflated))
    if inflated:
        failures.append("Sycophantic CARL inflation on unimproved dimension(s): " + ", ".join(inflated))
        suggestions.append(
            "Grade `carl_after` honestly from actual evidence in the retry text—never inflate a missing or vague dimension to `strong` just to be polite."
        )
    carl_score = carl_acc * sycophancy_mult

    # Conciseness & grounding in actual before-vs-after diff
    combined = f"{enc} {imp} {pol}"
    enc_ok = bool(enc) and _count_sentences(enc) <= 2 and len(enc.split()) <= 40
    imp_ok = bool(imp) and _count_sentences(imp) <= 2 and len(imp.split()) <= 50
    pol_ok = (not pol) or (_count_sentences(pol) <= 2 and len(pol.split()) <= 40)
    if enc and not enc_ok:
        failures.append(f"`encouragement` too verbose ({len(enc.split())} words, {_count_sentences(enc)} sentences).")
    if imp and not imp_ok:
        failures.append(f"`what_improved` too verbose ({len(imp.split())} words, {_count_sentences(imp)} sentences).")
    if pol and not pol_ok:
        failures.append(f"`one_more_polish` too verbose ({len(pol.split())} words, {_count_sentences(pol)} sentences).")

    must_ref = [str(k).lower() for k in (expected.get("must_reference_any") or []) if str(k).strip()]
    low = combined.lower()
    grounded = (not must_ref) or any(tok in low for tok in must_ref)
    if not grounded:
        failures.append(f"Retry feedback not grounded in actual retry details (expected one of {must_ref[:6]}).")
        suggestions.append("Cite concrete details from the candidate's retry (metrics, actions, or missing gaps) in `what_improved` and `one_more_polish`.")

    style_grounding = (
        (0.45 if grounded else 0.0)
        + (0.20 if enc_ok else 0.0)
        + (0.20 if imp_ok else 0.0)
        + (0.15 if pol_ok else 0.0)
    )

    harsh_hits = [pat.pattern for pat in HARSH_TONE_PATTERNS if pat.search(combined)]
    if harsh_hits:
        failures.append(f"Harsh tone detected: {harsh_hits}")
        suggestions.append("Stay warm and constructive even when a dimension is still missing.")
    tone_score = 0.0 if (harsh_hits or not enc) else 1.0

    subscores = {
        "json_validity": round(json_score, 4),
        "schema": round(schema_score, 4),
        "carl_honesty": round(carl_score, 4),
        "grounding_and_style": round(style_grounding, 4),
        "tone": round(tone_score, 4),
    }
    total = (
        0.10 * subscores["json_validity"]
        + 0.15 * subscores["schema"]
        + 0.40 * subscores["carl_honesty"]
        + 0.22 * subscores["grounding_and_style"]
        + 0.13 * subscores["tone"]
    )
    return round(max(0.0, min(1.0, total)), 4), {
        "example_id": example.get("id"),
        "subscores": subscores,
        "failures": failures,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def score_practice_hint(
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Deterministic scorer for plain-text 2-sentence whisper `practice_hint` with strict no-spoiler guard."""
    failures: list[str] = []
    suggestions: list[str] = []

    text = (raw_output or "").strip()
    if not text:
        return 0.0, {
            "example_id": example.get("id"),
            "subscores": {"format": 0.0},
            "failures": ["Empty hint output."],
            "suggestions": ["Return 1-2 warm plain-text coaching sentences."],
        }

    expected = example.get("expected") or {}

    # 1. Plain text only (no JSON object, no markdown headers/fences/bullets)
    looks_json = text.startswith("{") and text.endswith("}")
    has_md = bool(re.search(r"(^|\n)\s*(#{1,6}\s|```|[-*]\s)", text))
    format_score = 1.0
    if looks_json:
        format_score = 0.0
        failures.append("Hint emitted JSON instead of plain text.")
        suggestions.append("Output plain conversational text only—never JSON or markdown.")
    elif has_md:
        format_score = 0.4
        failures.append("Hint contained markdown headers, code fences, or bullet lists.")
        suggestions.append("Emit plain whisper prose with no markdown headers, code fences, or bullet points.")

    # 2. Brevity: 1-2 sentences, 12-55 words
    wc = len(text.split())
    sc = _count_sentences(text)
    len_ok = 12 <= wc <= 55
    sent_ok = 1 <= sc <= 2
    if not len_ok:
        failures.append(f"Hint word count ({wc}) out of 12-55 word whisper budget.")
        suggestions.append("Keep whisper hints tight (1-2 crisp sentences, ~20-45 words).")
    if not sent_ok:
        failures.append(f"Hint has {sc} sentences (expected 1-2 sentences).")
        suggestions.append("Limit whisper hints strictly to 1-2 sentences.")
    brevity_score = (0.5 if len_ok else 0.1) + (0.5 if sent_ok else 0.0)

    # 3. Grounding in candidate's current situation/workspace
    low = text.lower()
    must_ref = [str(k).lower() for k in (expected.get("must_reference_any") or []) if str(k).strip()]
    grounded = (not must_ref) or any(tok in low for tok in must_ref)
    if not grounded:
        failures.append(f"Hint did not ground in candidate's current block (expected one of {must_ref[:6]}).")
        suggestions.append("Reference what the candidate has tried or where they are stuck before nudging next step.")
    grounding_score = 1.0 if grounded else 0.0

    # 4. Strict no-spoiler + warm tone guard
    harsh_hits = [pat.pattern for pat in HARSH_TONE_PATTERNS if pat.search(text)]
    forbidden = [str(s) for s in (expected.get("forbidden_spoilers") or []) if str(s).strip()]
    spoiler_hits = [s for s in forbidden if s.lower() in low]
    if harsh_hits:
        failures.append(f"Harsh phrasing detected: {harsh_hits}")
    if spoiler_hits:
        failures.append(f"Hint spoiled the solution/data-structure/code: {spoiler_hits}")
        suggestions.append(
            "NEVER name the exact data structure, algorithm, library, or code fix—ask a guiding question about the invariant or bottleneck instead."
        )
    safety_score = 0.0 if (harsh_hits or spoiler_hits) else 1.0

    subscores = {
        "plain_text_format": round(format_score, 4),
        "brevity": round(brevity_score, 4),
        "grounding": round(grounding_score, 4),
        "no_spoiler_and_tone": round(safety_score, 4),
    }
    total = (
        0.18 * subscores["plain_text_format"]
        + 0.24 * subscores["brevity"]
        + 0.23 * subscores["grounding"]
        + 0.35 * subscores["no_spoiler_and_tone"]
    )
    return round(max(0.0, min(1.0, total)), 4), {
        "example_id": example.get("id"),
        "subscores": subscores,
        "failures": failures,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def score_practice_debrief(
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Deterministic scorer for end-of-practice supportive CARL debrief (`practice_debrief`)."""
    failures: list[str] = []
    suggestions: list[str] = []

    parsed, json_score, json_note = _parse_json_with_strictness(raw_output)
    if json_note:
        failures.append(json_note)
        if parsed is not None:
            suggestions.append("Emit raw JSON only with zero markdown fences or prose preamble.")
    if parsed is None:
        return 0.0, {
            "example_id": example.get("id"),
            "subscores": {"json_validity": 0.0},
            "failures": failures,
            "suggestions": ["Return strictly valid JSON matching the practice_debrief schema."],
        }

    expected = example.get("expected") or {}
    exp_qids = [str(q) for q in (expected.get("question_ids") or [])]
    exp_carl_map = expected.get("expected_carl") or {}
    must_not_inflate_map = expected.get("must_not_inflate") or {}
    exp_drill_qids = [str(q) for q in (expected.get("retry_drill_question_ids") or exp_qids)]

    overall = str(parsed.get("overall_encouragement") or "").strip() if isinstance(parsed.get("overall_encouragement"), str) else ""
    primer = str(parsed.get("carl_primer") or "").strip() if isinstance(parsed.get("carl_primer"), str) else ""
    raw_strengths = parsed.get("top_strengths") if isinstance(parsed.get("top_strengths"), list) else []
    strengths = [s.strip() for s in raw_strengths if isinstance(s, str) and s.strip()]
    raw_reviews = parsed.get("question_reviews") if isinstance(parsed.get("question_reviews"), list) else []
    raw_drill = parsed.get("retry_drill") if isinstance(parsed.get("retry_drill"), dict) else {}

    reviews_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_reviews:
        if isinstance(item, dict) and isinstance(item.get("question_id"), str) and item["question_id"].strip():
            reviews_by_id[item["question_id"].strip()] = item

    # 1. Schema & exact question_id preservation
    matched_qids = [qid for qid in exp_qids if qid in reviews_by_id]
    missing_qids = [qid for qid in exp_qids if qid not in reviews_by_id]
    if missing_qids:
        failures.append(f"Missing or mangled `question_id`(s) in `question_reviews`: {missing_qids}")
        suggestions.append("Copy every answered `question_id` verbatim into `question_reviews` (never rename to Q1/Q2).")

    per_review_schema: list[float] = []
    for qid in exp_qids:
        rev = reviews_by_id.get(qid)
        if rev is None:
            per_review_schema.append(0.0)
            continue
        carl = rev.get("carl") if isinstance(rev.get("carl"), dict) else {}
        carl_ok = sum(
            1 for d in CARL_DIMS if isinstance(carl.get(d), str) and carl[d].strip().lower() in CARL_RANK
        ) == 4
        fields_ok = [
            isinstance(rev.get("title"), str) and bool(rev["title"].strip()),
            isinstance(rev.get("prompt"), str) and bool(rev["prompt"].strip()),
            carl_ok,
            isinstance(rev.get("strength"), str) and bool(rev["strength"].strip()),
            isinstance(rev.get("coaching_tip"), str) and bool(rev["coaching_tip"].strip()),
            isinstance(rev.get("carl_reframe"), str) and bool(rev["carl_reframe"].strip()),
        ]
        if not carl_ok:
            failures.append(f"[{qid}] Invalid or incomplete `carl` dimensions.")
        per_review_schema.append(sum(1.0 for ok in fields_ok if ok) / len(fields_ok))

    drill_qid = str(raw_drill.get("question_id") or "").strip() if isinstance(raw_drill.get("question_id"), str) else ""
    drill_focus = str(raw_drill.get("focus") or "").strip() if isinstance(raw_drill.get("focus"), str) else ""
    drill_prompt = str(raw_drill.get("prompt") or "").strip() if isinstance(raw_drill.get("prompt"), str) else ""
    drill_schema_ok = bool(drill_qid and drill_focus and drill_prompt)
    if not drill_schema_ok:
        failures.append("Incomplete `retry_drill` object (needs `question_id`, `focus`, `prompt`).")

    top_schema_checks = [
        bool(overall),
        2 <= len(strengths) <= 4,
        bool(primer),
        drill_schema_ok,
        len(matched_qids) == len(exp_qids),
    ]
    if not (2 <= len(strengths) <= 4):
        failures.append(f"`top_strengths` count ({len(strengths)}) outside 2-4 range.")
    top_schema_score = sum(1.0 for ok in top_schema_checks if ok) / len(top_schema_checks)
    review_schema_score = (sum(per_review_schema) / len(per_review_schema)) if per_review_schema else 0.0
    schema_score = 0.5 * top_schema_score + 0.5 * review_schema_score

    # 2. CARL accuracy + anti-sycophancy + highest-leverage retry drill selection
    carl_pts: list[float] = []
    inflated: list[str] = []
    for qid in exp_qids:
        rev = reviews_by_id.get(qid)
        exp_c = exp_carl_map.get(qid) or {}
        no_inf = set(must_not_inflate_map.get(qid) or [])
        if rev is None or not isinstance(rev.get("carl"), dict):
            carl_pts.extend([0.0] * len(CARL_DIMS))
            continue
        raw_c = rev["carl"]
        for dim in CARL_DIMS:
            pv = raw_c.get(dim).strip().lower() if isinstance(raw_c.get(dim), str) else ""
            ev = str(exp_c.get(dim, "missing")).strip().lower()
            if pv not in CARL_RANK:
                carl_pts.append(0.0)
                continue
            diff = abs(CARL_RANK[pv] - CARL_RANK[ev])
            if diff == 0:
                carl_pts.append(1.0)
            elif diff == 1:
                carl_pts.append(0.5)
                failures.append(f"[{qid}] `carl.{dim}` slightly off: expected {ev!r}, got {pv!r}.")
            else:
                carl_pts.append(0.0)
                failures.append(f"[{qid}] `carl.{dim}` far off: expected {ev!r}, got {pv!r}.")
            if dim in no_inf and CARL_RANK[pv] > CARL_RANK[ev]:
                inflated.append(f"{qid}.{dim}: expected {ev}, got {pv}")

    carl_raw = (sum(carl_pts) / len(carl_pts)) if carl_pts else 0.0
    if inflated:
        failures.append("Sycophantic CARL inflation on unprovided dimension(s): " + ", ".join(inflated))
        suggestions.append("Grade each question's CARL honestly from transcript evidence; never mark missing Results or Learnings as `strong`.")
    carl_penalized = carl_raw * max(0.0, 1.0 - 0.35 * len(inflated))

    drill_target_ok = drill_qid in exp_drill_qids
    if not drill_target_ok:
        failures.append(f"`retry_drill.question_id`={drill_qid!r} did not pick weakest question ({exp_drill_qids}).")
        suggestions.append("Point `retry_drill.question_id` to the question with the weakest/missing CARL signal.")
    carl_and_drill = 0.72 * carl_penalized + 0.28 * (1.0 if drill_target_ok else 0.0)

    # 3. Grounding in candidate specifics & UI-friendly conciseness
    text_parts = [overall, primer, drill_focus, drill_prompt, *strengths]
    concise_checks: list[bool] = [
        bool(overall) and _count_sentences(overall) <= 3 and len(overall.split()) <= 68,
        bool(primer) and _count_sentences(primer) <= 2 and len(primer.split()) <= 40,
        bool(drill_focus) and len(drill_focus.split()) <= 14,
        bool(drill_prompt) and _count_sentences(drill_prompt) <= 3 and len(drill_prompt.split()) <= 60,
    ]
    if overall and not concise_checks[0]:
        failures.append(f"`overall_encouragement` too verbose ({len(overall.split())} words, {_count_sentences(overall)} sentences).")
        suggestions.append("Keep `overall_encouragement` to 2-3 punchy sentences (under 65 words) so Q1 stays above the fold.")
    if primer and not concise_checks[1]:
        failures.append(f"`carl_primer` too verbose ({len(primer.split())} words).")

    for qid in exp_qids:
        rev = reviews_by_id.get(qid)
        if not rev:
            concise_checks.extend([False, False, False])
            continue
        st = str(rev.get("strength") or "").strip()
        ct = str(rev.get("coaching_tip") or "").strip()
        rf = str(rev.get("carl_reframe") or "").strip()
        text_parts.extend([st, ct, rf])
        st_ok = bool(st) and _count_sentences(st) <= 2 and len(st.split()) <= 48
        ct_ok = bool(ct) and _count_sentences(ct) <= 2 and len(ct.split()) <= 48
        rf_ok = bool(rf) and _count_sentences(rf) <= 3 and len(rf.split()) <= 75
        if st and not st_ok:
            failures.append(f"[{qid}] `strength` too long ({len(st.split())} words, {_count_sentences(st)} sents).")
        if ct and not ct_ok:
            failures.append(f"[{qid}] `coaching_tip` too long ({len(ct.split())} words, {_count_sentences(ct)} sents).")
        if rf and not rf_ok:
            failures.append(f"[{qid}] `carl_reframe` too long ({len(rf.split())} words, {_count_sentences(rf)} sents).")
        concise_checks.extend([st_ok, ct_ok, rf_ok])

    combined = " ".join(text_parts)
    low = combined.lower()
    must_ref = [str(k).lower() for k in (expected.get("must_reference_any") or []) if str(k).strip()]
    hits = [tok for tok in must_ref if tok in low]
    grounding_ratio = 1.0 if (not must_ref or len(hits) >= min(2, len(must_ref))) else (0.5 if hits else 0.0)
    if grounding_ratio < 1.0:
        failures.append(f"Debrief lacked concrete transcript/workspace grounding (hits={hits}, expected >=2 of {must_ref[:6]}).")
        suggestions.append("Ground `top_strengths`, `strength`, and `carl_reframe` in exact numbers, systems, or details from the candidate's transcript/code/diagram.")

    concise_ratio = sum(1.0 for ok in concise_checks if ok) / max(1, len(concise_checks))
    grounding_and_conciseness = 0.50 * grounding_ratio + 0.50 * concise_ratio

    # 4. Warm tone guard
    harsh_hits = [pat.pattern for pat in HARSH_TONE_PATTERNS if pat.search(combined)]
    if harsh_hits:
        failures.append(f"Harsh phrasing detected: {harsh_hits}")
        suggestions.append("Keep coaching warm, encouraging, and strengths-first even on missing CARL dimensions.")
    tone_score = 0.0 if (harsh_hits or not overall) else 1.0

    subscores = {
        "json_validity": round(json_score, 4),
        "schema_and_ids": round(schema_score, 4),
        "carl_and_drill": round(carl_and_drill, 4),
        "grounding_and_conciseness": round(grounding_and_conciseness, 4),
        "tone": round(tone_score, 4),
    }
    total = (
        0.10 * subscores["json_validity"]
        + 0.20 * subscores["schema_and_ids"]
        + 0.35 * subscores["carl_and_drill"]
        + 0.25 * subscores["grounding_and_conciseness"]
        + 0.10 * subscores["tone"]
    )
    return round(max(0.0, min(1.0, total)), 4), {
        "example_id": example.get("id"),
        "subscores": subscores,
        "failures": failures,
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def score_example(
    target: str,
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    if target == "triage_window":
        return score_triage_window(raw_output, example)
    if target == "practice_interviewer":
        return score_practice_interviewer_reply(raw_output, example)
    if target == "suggest_probes":
        return score_suggest_probes(raw_output, example)
    if target == "practice_retry_eval":
        return score_practice_retry_eval(raw_output, example)
    if target == "practice_hint":
        return score_practice_hint(raw_output, example)
    if target == "practice_debrief":
        return score_practice_debrief(raw_output, example)
    raise ValueError(f"Unsupported eval target: {target!r}")
