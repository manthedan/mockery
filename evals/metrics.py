from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mockery.llm import _extract_json_object
from mockery.prompts import (
    practice_tone_instruction,
    render_prompt,
    triage_diarize_rules,
    validate_prompt_template,
)

TARGET_TO_PROMPT_NAME: dict[str, str] = {
    "triage_window": "triage_window",
    "practice_interviewer": "practice_interviewer_reply",
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


def build_user_prompt_for_example(target: str, example: dict[str, Any]) -> str:
    if target == "triage_window":
        return build_triage_user_prompt(example)
    if target == "practice_interviewer":
        return build_practice_interviewer_user_prompt(example)
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


def score_example(
    target: str,
    raw_output: str,
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    if target == "triage_window":
        return score_triage_window(raw_output, example)
    if target == "practice_interviewer":
        return score_practice_interviewer_reply(raw_output, example)
    raise ValueError(f"Unsupported eval target: {target!r}")
