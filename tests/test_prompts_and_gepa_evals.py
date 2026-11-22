from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from evals.metrics import (
    TARGET_TO_PROMPT_NAME,
    build_user_prompt_for_example,
    load_jsonl_dataset,
    render_system_prompt_for_example,
    score_practice_interviewer_reply,
    score_triage_window,
    split_dataset,
    validate_candidate_prompt,
    validate_dataset,
)
from evals.optimize_gepa import (
    TARGET_TEMPERATURES,
    _sanitize_error_message,
    build_prompt_diff,
    compute_max_tokens,
    evaluate_candidate_on_examples,
    main as gepa_cli_main,
    make_cli_reflection_lm,
    make_cli_task_runner,
    parse_cli_transport,
    preflight_gepa_dependencies,
    resolve_task_endpoint_and_key,
)
from mockery.llm import InterviewCopilotLLM, LLMRuntimeConfig
from mockery.models import InterviewSession, QAGroup, SessionConfig, Utterance
from mockery.prompts import (
    PROMPT_REGISTRY,
    calibration_rubric_for,
    find_missing_placeholders,
    get_prompt_spec,
    list_prompts,
    practice_tone_instruction,
    probe_guidance_for,
    render_prompt,
    render_prompt_template,
    triage_diarize_rules,
    validate_prompt_template,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "mockery"
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"


def test_prompt_registry_completeness_and_strict_placeholder_validation() -> None:
    expected_names = {
        "triage_window",
        "practice_interviewer_reply",
        "suggest_probes",
        "classify_topic_shift",
        "check_speaker_role_orientation",
        "regroup_cards_full_context",
        "practice_plan_framing",
        "practice_plan_tailored",
        "practice_hint",
        "practice_debrief",
        "practice_retry_eval",
        "calibration_report",
    }
    assert set(PROMPT_REGISTRY.keys()) == expected_names

    tier1_names = {spec.name for spec in list_prompts(tier=1)}
    assert tier1_names == {"triage_window", "practice_interviewer_reply"}

    for name, spec in PROMPT_REGISTRY.items():
        assert spec.name == name
        assert spec.description.strip()
        assert spec.default_text.strip()
        assert spec.output_format in ("json", "markdown", "text")
        assert find_missing_placeholders(name, spec.default_text) == []
        assert validate_prompt_template(name, spec.default_text) == []

    # Literal JSON braces stay intact and no double-substitution occurs across values
    rendered = render_prompt_template(
        '{"json": "ok"} A={a} B={b}',
        ("a", "b"),
        a="{b}",
        b="VALUE_B",
    )
    assert rendered == '{"json": "ok"} A={b} B=VALUE_B'

    with pytest.raises(KeyError):
        render_prompt("triage_window")
    with pytest.raises(KeyError):
        render_prompt("triage_window", diarize_rules="ok", extra_key="nope")
    with pytest.raises(KeyError):
        get_prompt_spec("nonexistent_prompt")

    # Strict validation rejects missing, double-braced, and unknown placeholders
    assert find_missing_placeholders("triage_window", "No placeholder here") == ["diarize_rules"]
    double_errs = validate_prompt_template("triage_window", "Use {{diarize_rules}} here.")
    assert any("double-braced" in e.lower() for e in double_errs)
    assert any("missing required" in e.lower() for e in double_errs)

    unknown_errs = validate_prompt_template(
        "triage_window",
        "Rules: {diarize_rules} Extra: {unexpected_var}",
    )
    assert any("{unexpected_var}" in e for e in unknown_errs)


def test_src_mockery_never_imports_gepa_litellm_or_evals() -> None:
    py_files = sorted(SRC_DIR.rglob("*.py"))
    assert py_files
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in ("gepa", "litellm", "evals"), f"{path} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                assert mod not in ("gepa", "litellm", "evals"), f"{path} imports from {node.module}"


def test_llm_runtime_prompt_parity_all_12_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise ALL 12 extracted prompts through InterviewCopilotLLM and assert independent baseline invariants."""
    cfg = LLMRuntimeConfig(tmp_path)
    cfg.provider = "openai"
    cfg.api_key = "test-key"
    cfg.fast_model = "gpt-4o-mini"
    cfg.calibration_model = "gpt-4o"
    llm = InterviewCopilotLLM(cfg)

    captured: list[tuple[str, str]] = []

    async def fake_complete_text(**kwargs: Any) -> str:
        captured.append((kwargs["system"], kwargs["user"]))
        return "{}"

    monkeypatch.setattr(llm, "_complete_text", fake_complete_text)

    session = InterviewSession(
        config=SessionConfig(role_title="Senior Engineer", interview_type="system_design")
    )
    utts = [
        Utterance(id=f"utt_{i}", speaker=("interviewer" if i % 2 == 0 else "candidate"), text=f"Turn {i} text.", start_ms=i * 2000, end_ms=(i + 1) * 2000, final=True)
        for i in range(6)
    ]
    session.utterances.extend(utts)
    qa = QAGroup(
        id="qa_1",
        index=1,
        title="Caching Design",
        utterance_ids=[u.id for u in utts],
        interview_type="system_design",
    )
    session.qa_groups.append(qa)

    async def _exercise_all_12() -> None:
        # 1a & 1b: triage_window (diarize=True & False)
        await llm.triage_window(session, [utts[1]], diarize=True)
        await llm.triage_window(session, [utts[1]], diarize=False)
        # 2: suggest_probes
        await llm.suggest_probes(session, qa)
        # 3: classify_topic_shift
        await llm.classify_topic_shift(session, qa, "Let's switch to a new topic.")
        # 4: check_speaker_role_orientation (needs >= 6 finals with 2 speakers)
        await llm.check_speaker_role_orientation(session)
        # 5: regroup_cards_full_context (needs >= 3 non-warmup utts)
        await llm.regroup_cards_full_context(session)
        # 6: calibration_report
        await llm.generate_calibration_report(session)
        # 7: practice_plan_framing (system_design with bank candidate + JD)
        await llm.generate_practice_plan(
            role_title="Senior Engineer",
            company="Acme",
            interview_type="system_design",
            target_level="Senior (L5)",
            question_source="mixed",
            question_count=1,
            bank_candidates=[{"id": "sd_1", "title": "URL Shortener", "prompt": "Design a URL shortener."}],
            jd_text="Build low-latency distributed systems.",
        )
        # 8: practice_plan_tailored (behavioral)
        await llm.generate_practice_plan(
            role_title="Senior Engineer",
            company="Acme",
            interview_type="behavioral",
            target_level="Senior (L5)",
            question_source="resume_jd_tailored",
            question_count=2,
            bank_candidates=[],
            redacted_resume="Led migration of billing pipeline.",
            jd_text="Own high-reliability backend services.",
        )
        # 9: practice_interviewer_reply
        await llm.generate_practice_interviewer_reply(
            role_title="Senior Engineer",
            company="Acme",
            interview_type="system_design",
            target_level="Senior (L5)",
            coach_style="supportive",
            current_question={"prompt": "Design a cache", "follow_ups": []},
            next_question=None,
            recent_turns=[],
            candidate_answer="Client hits cache then DB.",
            follow_up_count=0,
            existing_carl={"context": "missing", "action": "missing", "result": "missing", "learning": "missing"},
        )
        # 10: practice_hint
        await llm.generate_practice_hint(
            interview_type="coding",
            target_level="Senior (L5)",
            current_question={"prompt": "Two sum"},
            recent_turns=[],
        )
        # 11: practice_debrief
        await llm.generate_practice_debrief(
            role_title="Senior Engineer",
            company="Acme",
            interview_type="behavioral",
            target_level="Senior (L5)",
            questions=[{"id": "q1", "title": "Ownership", "prompt": "Tell me about a time..."}],
            turns=[{"role": "candidate", "question_id": "q1", "text": "I owned the rollout."}],
        )
        # 12: practice_retry_eval
        await llm.evaluate_practice_retry(
            role_title="Senior Engineer",
            target_level="Senior (L5)",
            drill_focus="Result",
            drill_prompt="Quantify the result in 45s",
            original_answer="I shipped it.",
            retry_answer="Latency dropped by 40%.",
        )

    asyncio.run(_exercise_all_12())
    assert len(captured) == 13

    # Independent baseline assertions (not only comparing against render_prompt)
    assert "Interview Turn Triage planner" in captured[0][0]
    assert "SPEAKER DIARIZATION & RAPID HANDOFF SPLITTING (CRITICAL)" in captured[0][0]
    assert "SPEAKER LABELS ARE HARDWARE-SEPARATED" in captured[1][0]
    assert "real-time mock interview co-pilot" in captured[2][0]
    assert "This is a SYSTEM DESIGN interview." in captured[2][0]
    assert captured[3][0] == "Classify whether an interview turn pivots to a new unrelated question. Return ONLY valid JSON."
    assert captured[4][0] == "Determine whether INTERVIEWER and CANDIDATE voice labels are globally inverted. Return ONLY valid JSON."
    assert captured[5][0] == "Segment an interview transcript into clean Q&A or phase card boundaries. Return ONLY valid JSON."
    assert "INTERVIEW-TYPE EVALUATION RUBRIC (system_design):" in captured[6][0]
    assert "### 1. Calibration Check: Your Blind Call vs. Transcript Evidence" in captured[6][0]
    assert '"tailored_prompt"' in captured[7][0]
    assert "[PERSON_NAME_1]" in captured[8][0]
    assert "You are proctoring a live system design mock interview for Senior Engineer (Senior (L5))." in captured[9][0]
    assert "whispering a quick hint" in captured[10][0]
    assert "teaching the CARL framework" in captured[11][0]
    assert "evaluating a candidate's 45-second retry drill" in captured[12][0]

    # Also verify exact registry rendering alignment
    assert captured[0][0] == render_prompt("triage_window", diarize_rules=triage_diarize_rules(True))
    assert captured[1][0] == render_prompt("triage_window", diarize_rules=triage_diarize_rules(False))
    assert captured[2][0] == render_prompt("suggest_probes", probe_guidance=probe_guidance_for("system_design"))
    assert captured[6][0] == render_prompt(
        "calibration_report",
        interview_type="system_design",
        rubric_block=calibration_rubric_for("system_design"),
    )
    assert captured[9][0] == render_prompt(
        "practice_interviewer_reply",
        interview_type_label="system design",
        role_title="Senior Engineer",
        target_level="Senior (L5)",
        tone_instruction=practice_tone_instruction("supportive"),
    )


def _walk_decoded_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
        # Also parse nested JSON strings (like reference_response)
        stripped = obj.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                nested = json.loads(stripped)
                out.extend(_walk_decoded_strings(nested))
            except json.JSONDecodeError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_decoded_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_decoded_strings(item))
    return out


def test_eval_datasets_schema_splits_runtime_modes_and_decoded_pii() -> None:
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_re = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
    ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    url_id_re = re.compile(r"https?://|linkedin\.com|github\.com|drive\.google\.com", re.I)
    street_re = re.compile(r"\b\d{2,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Ln|Lane)\b")

    for target, filename in (
        ("triage_window", "triage_window.jsonl"),
        ("practice_interviewer", "practice_interviewer.jsonl"),
    ):
        path = DATASETS_DIR / filename
        rows = load_jsonl_dataset(path)
        validate_dataset(target, rows)
        train, val = split_dataset(rows)
        assert len(train) >= 6
        assert len(val) >= 3
        assert {r["id"] for r in train}.isdisjoint({r["id"] for r in val})

        for row in rows:
            for s in _walk_decoded_strings(row):
                assert not email_re.search(s), f"Email-like PII in {filename}: {s[:80]}"
                assert not phone_re.search(s), f"Phone-like PII in {filename}: {s[:80]}"
                assert not ssn_re.search(s), f"SSN-like PII in {filename}: {s[:80]}"
                assert not url_id_re.search(s), f"URL/profile identifier in {filename}: {s[:80]}"
                assert not street_re.search(s), f"Street address in {filename}: {s[:80]}"

            if target == "triage_window":
                assert row["session_index"]["capture_mode"] in ("in_person", "remote_meet")

            sys_prompt = render_system_prompt_for_example(target, row)
            user_prompt = build_user_prompt_for_example(target, row)
            assert sys_prompt.strip()
            assert user_prompt.strip()
            assert isinstance(json.loads(row["reference_response"]), dict)

    # Custom dataset validation rejects train-only leaks, unknown splits, and missing fields
    tw_rows = load_jsonl_dataset(DATASETS_DIR / "triage_window.jsonl")
    train_only = [r for r in tw_rows if r["split"] == "train"]
    with pytest.raises(ValueError, match="disjoint train and val"):
        split_dataset(train_only)

    bad_split = [dict(tw_rows[0], split="dev"), dict(tw_rows[-1], split="val")]
    with pytest.raises(ValueError, match="invalid `split`"):
        validate_dataset("triage_window", bad_split)
    with pytest.raises(ValueError, match="Unrecognized dataset split"):
        split_dataset(bad_split)

    missing_field = [dict(tw_rows[0])]
    missing_field[0].pop("incoming_utterances")
    with pytest.raises(ValueError, match="incoming_utterances"):
        validate_dataset("triage_window", missing_field)


def test_deterministic_metrics_hardened_against_gaming_and_spoilers() -> None:
    tw_rows = load_jsonl_dataset(DATASETS_DIR / "triage_window.jsonl")
    handoff_ex = next(r for r in tw_rows if r["id"] == "tw_02_rapid_handoff_split")
    single_ex = next(r for r in tw_rows if r["id"] == "tw_01_single_mic_speaker_flip")
    hw_ex = next(r for r in tw_rows if r["id"] == "tw_06_hardware_separated_preserve")

    good_score, good_asi = score_triage_window(handoff_ex["reference_response"], handoff_ex)
    assert good_score == 1.0
    assert good_asi["failures"] == []

    # Regression: malformed null plan + duplicate ID + missing suggested_title + fabricated insertions CANNOT score 1.0
    orig_text = single_ex["incoming_utterances"][0]["text"]
    gamed_triage = json.dumps(
        {
            "plans": [
                None,
                {
                    "utterance_id": "utt_02",
                    "segments": [
                        {
                            "speaker": "candidate",
                            "text": f"Fabricated preamble unrelated words before. {orig_text} Fabricated epilogue extra text inserted.",
                        }
                    ],
                    "card_action": "keep_in_current",
                    "confidence": "high",
                    "rationale": "First entry missing suggested_title",
                },
                {
                    "utterance_id": "utt_02",
                    "segments": [{"speaker": "candidate", "text": orig_text}],
                    "card_action": "keep_in_current",
                    "suggested_title": "",
                    "confidence": "high",
                    "rationale": "Duplicate overwrite attempt",
                },
            ]
        }
    )
    gamed_score, gamed_asi = score_triage_window(gamed_triage, single_ex)
    assert gamed_score < 0.75
    assert any("Non-object entry" in f for f in gamed_asi["failures"])
    assert any("Duplicate plan entry" in f for f in gamed_asi["failures"])
    assert any("Missing required string field `suggested_title`" in f for f in gamed_asi["failures"])
    assert any("Transcript fidelity drift" in f for f in gamed_asi["failures"])

    # Invalid JSON -> 0.0
    bad_json_score, bad_json_asi = score_triage_window("not valid json at all", handoff_ex)
    assert bad_json_score == 0.0
    assert bad_json_asi["failures"]

    # Hardware-separated mutation penalty
    mutated_hw = json.dumps(
        {
            "plans": [
                {
                    "utterance_id": "utt_31",
                    "segments": [{"speaker": "candidate", "text": "Paraphrased text"}],
                    "card_action": "keep_in_current",
                    "suggested_title": "",
                    "confidence": "high",
                    "rationale": "Wrongly altered speaker",
                }
            ]
        }
    )
    hw_score, hw_asi = score_triage_window(mutated_hw, hw_ex)
    assert hw_score < 0.6
    assert any("Hardware-separated" in f for f in hw_asi["failures"])

    # Practice interviewer metrics: prose spoiler, >3 sentences, non-string coercion, monotonicity, harsh tone
    pi_rows = load_jsonl_dataset(DATASETS_DIR / "practice_interviewer.jsonl")
    mono_ex = next(r for r in pi_rows if r["id"] == "pi_08_carl_monotonicity_guard")
    coding_ex = next(r for r in pi_rows if r["id"] == "pi_06_coding_brute_force_no_spoiler")

    pi_good, pi_good_asi = score_practice_interviewer_reply(mono_ex["reference_response"], mono_ex)
    assert pi_good == 1.0
    assert pi_good_asi["failures"] == []

    # Prose algorithm spoiler + 4 sentences on Two Sum must NOT get full credit
    prose_spoiler = json.dumps(
        {
            "interviewer_reply": (
                "Great start passing those four tests with the nested loop. "
                "Instead of quadratic time, use a hash map to store each number and look up target minus num as you iterate. "
                "When the complement is already in the map, return the two indices immediately. "
                "Can you code that up now?"
            ),
            "should_advance": False,
            "carl_signals": {"context": "partial", "action": "partial", "result": "partial", "learning": "missing"},
            "coach_nudge": {"tone": "tip", "message": "Great instinct verifying a clean baseline first—now optimize the lookup."},
        }
    )
    spoiler_score, spoiler_asi = score_practice_interviewer_reply(prose_spoiler, coding_ex)
    assert spoiler_score < 0.85
    assert any("Spoiler leaked" in f for f in spoiler_asi["failures"])
    assert any("4 sentences" in f for f in spoiler_asi["failures"])

    # Non-string fields must fail schema rather than being coerced to strings
    non_string_reply = json.dumps(
        {
            "interviewer_reply": ["Not", "a", "string"],
            "should_advance": False,
            "carl_signals": {"context": "partial", "action": "partial", "result": "partial", "learning": "missing"},
            "coach_nudge": {"tone": "tip", "message": 12345},
        }
    )
    non_str_score, non_str_asi = score_practice_interviewer_reply(non_string_reply, coding_ex)
    assert non_str_score < 0.65
    assert any("`interviewer_reply` must be a string" in f for f in non_str_asi["failures"])
    assert any("`coach_nudge.message` must be a string" in f for f in non_str_asi["failures"])

    # CARL monotonicity violation + harsh tone
    pi_mono_bad = json.dumps(
        {
            "interviewer_reply": "You failed to explain the spike clearly. Move on.",
            "should_advance": False,
            "carl_signals": {"context": "missing", "action": "partial", "result": "partial", "learning": "strong"},
            "coach_nudge": {"tone": "tip", "message": "That was a weak answer overall."},
        }
    )
    mono_score, mono_asi = score_practice_interviewer_reply(pi_mono_bad, mono_ex)
    assert mono_score < 0.45
    assert any("monotonicity" in f.lower() for f in mono_asi["failures"])
    assert any("harsh" in f.lower() for f in mono_asi["failures"])

    # Candidate prompt validation rejects dropped/double-braced/unknown placeholders before LLM calls
    ok, errs = validate_candidate_prompt("triage_window", "Emit valid JSON with {{diarize_rules}} and {extra}.")
    assert not ok
    assert any("double-braced" in e.lower() for e in errs)
    assert any("{extra}" in e for e in errs)

    invalid_eval = evaluate_candidate_on_examples(
        target="triage_window",
        candidate_text="Emit valid JSON without required placeholders.",
        examples=tw_rows[:2],
        runner=lambda *_: (_ for _ in ()).throw(AssertionError("Runner should not be called for invalid prompt")),
    )
    assert invalid_eval["valid_prompt"] is False
    assert invalid_eval["mean_score"] == 0.0


def test_runtime_task_settings_endpoint_key_isolation_and_error_sanitization() -> None:
    # Runtime settings parity with src/mockery/llm.py
    assert TARGET_TEMPERATURES == {
        "triage_window": 0.0,
        "practice_interviewer": 0.4,
        "suggest_probes": 0.35,
        "practice_retry_eval": 0.35,
        "practice_hint": 0.4,
        "practice_debrief": 0.35,
    }
    assert compute_max_tokens("triage_window", {"incoming_utterances": [{}]}) == 450
    assert compute_max_tokens("triage_window", {"incoming_utterances": [{}, {}, {}]}) == 660
    assert compute_max_tokens("practice_interviewer", {}) == 480
    assert compute_max_tokens("suggest_probes", {}) == 420
    assert compute_max_tokens("practice_retry_eval", {}) == 380
    assert compute_max_tokens("practice_hint", {}) == 160

    # Overriding --task-base-url must NEVER leak unrelated OMLX_API_KEY or LLM_API_KEY
    fake_env = {
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "sk-secret-openai-key",
        "OMLX_API_KEY": "omlx-secret-local-key",
    }
    url_override, key_override = resolve_task_endpoint_and_key(
        "https://other-host.example/v1", None, env=fake_env
    )
    assert url_override == "https://other-host.example/v1"
    assert key_override == "local-no-key"

    # Matching default LLM_BASE_URL inherits LLM_API_KEY
    url_default, key_default = resolve_task_endpoint_and_key(None, None, env=fake_env)
    assert url_default == "https://api.openai.com/v1"
    assert key_default == "sk-secret-openai-key"

    # Default localhost oMLX inherits OMLX_API_KEY when no LLM_BASE_URL is set
    url_omlx, key_omlx = resolve_task_endpoint_and_key(
        None, None, env={"OMLX_API_KEY": "omlx-secret-local-key"}
    )
    assert url_omlx == "http://localhost:8000/v1"
    assert key_omlx == "omlx-secret-local-key"

    # Error sanitization redacts bearer/API tokens
    redacted = _sanitize_error_message(
        RuntimeError("401 Unauthorized: Bearer tok_secret12345 api_key=sk-1234567890abcdef\nnext line")
    )
    assert "tok_secret12345" not in redacted
    assert "sk-1234567890abcdef" not in redacted
    assert "\n" not in redacted
    assert "[REDACTED]" in redacted


def test_optimize_gepa_dry_run_eval_outage_and_stubbed_gepa_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_before = (SRC_DIR / "prompts.py").read_text(encoding="utf-8")

    # 1. Dry-run succeeds offline for both targets
    for target in sorted(TARGET_TO_PROMPT_NAME.keys()):
        out_dir = tmp_path / f"run_{target}"
        rc = gepa_cli_main(["--target", target, "--dry-run", "--output-dir", str(out_dir)])
        assert rc == 0

        report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "ok"
        assert report["mode"] == "dry_run"
        assert report["baseline"]["train_mean_score"] >= 0.95
        assert report["baseline"]["val_mean_score"] >= 0.95
        assert report["candidate"]["changed"] is False

    # 2. Unified diff does NOT concatenate -old and +new when both lack trailing newlines
    diff = build_prompt_diff("triage_window", "line1\nold_final", "line1\nnew_final")
    assert "-old_final\n+new_final\n" in diff

    # 3. Complete task backend outage in --eval-only returns exit code 1 (NOT 0!)
    def failing_runner_factory(**_kwargs: Any) -> Any:
        def _boom(_sys: str, _usr: str, _ex: dict[str, Any]) -> str:
            raise ConnectionError("Connection refused to http://localhost:8000/v1 with Bearer sk-secret123456")

        return _boom

    monkeypatch.setattr("evals.optimize_gepa.make_openai_compatible_task_runner", failing_runner_factory)
    outage_dir = tmp_path / "run_outage"
    rc_outage = gepa_cli_main(
        ["--target", "triage_window", "--eval-only", "--output-dir", str(outage_dir)]
    )
    assert rc_outage == 1
    outage_report = json.loads((outage_dir / "report.json").read_text(encoding="utf-8"))
    assert outage_report["status"] == "backend_error"
    assert outage_report["baseline"]["train"]["backend_failed"] is True
    assert "sk-secret123456" not in (outage_dir / "report.json").read_text(encoding="utf-8")

    # 4. Preflight fails fast BEFORE sending any baseline requests when litellm or gepa is missing
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name in ("gepa", "litellm") else real_find_spec(name, *a, **kw),
    )
    with pytest.raises(RuntimeError, match="uv sync --extra gepa"):
        preflight_gepa_dependencies("openai/gpt-5.1")

    # 5. Live GEPA orchestration path with stubbed gepa.optimize_anything + litellm
    tw_spec = get_prompt_spec("triage_window")
    tuned_prompt = tw_spec.default_text + "\n4. EXTRA: Emit raw JSON only."

    class _FakeGEPAResult:
        def __init__(self, best_cand: dict[str, str], scores: list[float], best_idx: int) -> None:
            self.best_candidate = best_cand
            self.val_aggregate_scores = scores
            self.best_idx = best_idx
            self.total_metric_calls = 12

    class _FakeCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_oa = types.ModuleType("gepa.optimize_anything")
    fake_oa.EngineConfig = _FakeCfg  # type: ignore[attr-defined]
    fake_oa.ReflectionConfig = _FakeCfg  # type: ignore[attr-defined]
    fake_oa.GEPAConfig = _FakeCfg  # type: ignore[attr-defined]

    def _fake_optimize_anything(**kwargs: Any) -> _FakeGEPAResult:
        ev = kwargs["evaluator"]
        # Exercise evaluator callback on a real training example
        score, side_info = ev(kwargs["seed_candidate"], kwargs["dataset"][0])
        assert score == 1.0
        assert side_info["failures"] == []
        return _FakeGEPAResult({"system_prompt": tuned_prompt}, [0.85, 0.99], 1)

    fake_oa.optimize_anything = _fake_optimize_anything  # type: ignore[attr-defined]
    fake_gepa_pkg = types.ModuleType("gepa")
    fake_gepa_pkg.optimize_anything = fake_oa  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gepa", fake_gepa_pkg)
    monkeypatch.setitem(sys.modules, "gepa.optimize_anything", fake_oa)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: object() if name in ("gepa", "litellm") else real_find_spec(name, *a, **kw),
    )
    monkeypatch.setattr(
        "evals.optimize_gepa.make_openai_compatible_task_runner",
        lambda **_kw: (lambda _sys, _usr, ex: ex["reference_response"]),
    )

    gepa_dir = tmp_path / "run_stubbed_gepa"
    rc_gepa = gepa_cli_main(["--target", "triage_window", "--output-dir", str(gepa_dir)])
    assert rc_gepa == 0
    gepa_report = json.loads((gepa_dir / "report.json").read_text(encoding="utf-8"))
    assert gepa_report["status"] == "ok"
    assert gepa_report["mode"] == "gepa_optimize"
    assert gepa_report["gepa"]["best_val_score"] == 0.99
    assert gepa_report["candidate"]["changed"] is True
    assert gepa_report["candidate"]["rejected_invalid_candidate"] is False
    assert "+4. EXTRA: Emit raw JSON only." in (gepa_dir / "prompt.diff").read_text(encoding="utf-8")

    # 6. Invalid candidate returned by GEPA (dropped placeholder) is REJECTED and never saved as winner
    fake_oa.optimize_anything = lambda **_kw: _FakeGEPAResult(  # type: ignore[attr-defined]
        {"system_prompt": "Broken prompt missing required placeholder completely."},
        [0.1],
        0,
    )
    invalid_dir = tmp_path / "run_invalid_cand"
    rc_inv = gepa_cli_main(["--target", "triage_window", "--output-dir", str(invalid_dir)])
    assert rc_inv == 0
    inv_report = json.loads((invalid_dir / "report.json").read_text(encoding="utf-8"))
    assert inv_report["candidate"]["rejected_invalid_candidate"] is True
    assert inv_report["candidate"]["changed"] is False
    assert (invalid_dir / "candidate_prompt.txt").read_text(encoding="utf-8") == tw_spec.default_text
    assert (invalid_dir / "prompt.diff").read_text(encoding="utf-8") == ""

    # Verify prompts.py was NEVER mutated
    assert (SRC_DIR / "prompts.py").read_text(encoding="utf-8") == prompts_before


def test_gepa_cli_transport_parsing_preflight_and_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert parse_cli_transport("http://localhost:8000/v1") is None
    assert parse_cli_transport("cli://claude") == ("claude", "")
    assert parse_cli_transport("cli://claude/sonnet") == ("claude", "sonnet")
    assert parse_cli_transport("cli://agy/gemini-4-neon:high") == ("agy", "gemini-4-neon:high")
    assert parse_cli_transport("cli://codex/gpt-6.1-sol") == ("codex", "gpt-6.1-sol")
    with pytest.raises(ValueError, match="Unsupported CLI backend"):
        parse_cli_transport("cli://unknown_cli/model")

    # cli:// reflection does NOT require litellm (only gepa + CLI binary on PATH)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name == "litellm" else (object() if name == "gepa" else real_find_spec(name, *a, **kw)),
    )
    monkeypatch.setattr("evals.optimize_gepa.shutil.which", lambda name: f"/usr/local/bin/{name}")

    fake_oa = types.ModuleType("gepa.optimize_anything")
    fake_oa.EngineConfig = object  # type: ignore[attr-defined]
    fake_oa.ReflectionConfig = object  # type: ignore[attr-defined]
    fake_oa.GEPAConfig = object  # type: ignore[attr-defined]
    fake_oa.optimize_anything = lambda **_kw: None  # type: ignore[attr-defined]
    fake_gepa_pkg = types.ModuleType("gepa")
    fake_gepa_pkg.optimize_anything = fake_oa  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa_pkg)
    monkeypatch.setitem(sys.modules, "gepa.optimize_anything", fake_oa)

    preflight_gepa_dependencies("cli://claude/sonnet")

    # Stub InterviewCopilotLLM._complete_text to verify task runner & callable reflection LM wiring
    seen: list[dict[str, Any]] = []

    async def fake_complete(self: Any, **kwargs: Any) -> str:
        seen.append({"cli": self.config.cli_backend, **kwargs})
        return '{"ok": true}'

    monkeypatch.setattr(InterviewCopilotLLM, "_complete_text", fake_complete)

    runner = make_cli_task_runner(
        target="triage_window",
        model="haiku",
        base_url="cli://claude",
        temperature=0.0,
    )
    assert runner("sys prompt", "user prompt", {"incoming_utterances": [{}]}) == '{"ok": true}'
    assert seen[-1]["cli"] == "claude"
    assert seen[-1]["model"] == "haiku"
    assert seen[-1]["expect_json"] is True

    reflect_fn = make_cli_reflection_lm("cli://agy/gemini-4-neon:high")
    assert reflect_fn("Improve this prompt") == '{"ok": true}'
    assert seen[-1]["cli"] == "agy"
    assert seen[-1]["model"] == "gemini-4-neon:high"
    assert seen[-1]["expect_json"] is False

    assert reflect_fn([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]) == '{"ok": true}'
    assert seen[-1]["system"] == "S"
    assert seen[-1]["user"] == "U"


def test_practice_debrief_dataset_and_scorer() -> None:
    from evals.metrics import score_practice_debrief

    rows = load_jsonl_dataset(DATASETS_DIR / "practice_debrief.jsonl")
    validate_dataset("practice_debrief", rows)
    train, val = split_dataset(rows)
    assert len(train) >= 6
    assert len(val) >= 4

    ex = rows[0]
    assert render_system_prompt_for_example("practice_debrief", ex).strip()
    assert build_user_prompt_for_example("practice_debrief", ex).strip()
    assert compute_max_tokens("practice_debrief", ex) == 1600
    assert TARGET_TEMPERATURES["practice_debrief"] == 0.35

    good_json = json.dumps(
        {
            "overall_encouragement": "Strong ownership on your webhook outage story, and a crisp start on cross-team trade-offs.",
            "top_strengths": [
                "Led incident command and isolated the poison-pill partner into a dead-letter shard within 28 minutes.",
                "Followed up with an automated canary check that prevented repeat certificate outages.",
            ],
            "carl_primer": "CARL turns strong engineering work into unmistakable L5 ownership by naming measurable impact and lasting takeaways.",
            "question_reviews": [
                {
                    "question_id": "q_01",
                    "title": "Production Outage Mitigation",
                    "prompt": "Tell me about a high-severity production outage you owned end to end.",
                    "carl": {"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                    "strength": "Clear metrics on the 1.8M backlog, 28-minute hotfix, and proactive canary check.",
                    "coaching_tip": "Highlight how you coordinated updates with support during the drain.",
                    "carl_reframe": "Context: 1.8M backed-up webhooks. Action: isolated dead-letter shard and shipped jitter fix. Result: drained in 28 minutes with zero lost orders and added cert canaries.",
                },
                {
                    "question_id": "q_02",
                    "title": "Cross-Team API Disagreement",
                    "prompt": "Tell me about a time you disagreed with another team on a technical approach.",
                    "carl": {"context": "strong", "action": "partial", "result": "missing", "learning": "missing"},
                    "strength": "You framed the GraphQL vs REST caching tension concisely.",
                    "coaching_tip": "Name the exact contract you agreed on, the latency outcome, and what changed for future API reviews.",
                    "carl_reframe": "I met the mobile lead to benchmark REST cache hit rates against GraphQL queries, agreed on a thin BFF endpoint that kept p99 under 40ms, and documented an API decision rubric.",
                },
            ],
            "retry_drill": {
                "question_id": "q_02",
                "focus": "Quantify Result & Add Learning (45s)",
                "prompt": "In 45 seconds, close your GraphQL vs REST story with the concrete outcome and one team-wide lesson.",
            },
        }
    )
    score, info = score_practice_debrief(good_json, ex)
    assert score >= 0.95, info

    # Mangled question_id + inflated CARL + wrong retry_drill should drop sharply
    bad_json = json.dumps(
        {
            "overall_encouragement": "Great job.",
            "top_strengths": ["Nice answer."],
            "carl_primer": "Use CARL.",
            "question_reviews": [
                {
                    "question_id": "Q1",
                    "title": "Outage",
                    "prompt": "Outage",
                    "carl": {"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                    "strength": "Good.",
                    "coaching_tip": "Keep going.",
                    "carl_reframe": "Reframe.",
                }
            ],
            "retry_drill": {"question_id": "q_01", "focus": "Polish", "prompt": "Try again."},
        }
    )
    bad_score, bad_info = score_practice_debrief(bad_json, ex)
    assert bad_score < 0.55, bad_info


def test_practice_debrief_retries_once_on_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = LLMRuntimeConfig(tmp_path)
    cfg.provider = "openai"
    cfg.api_key = "test-key"
    cfg.fast_model = "gpt-4o-mini"
    cfg.calibration_model = "gpt-4o"
    llm = InterviewCopilotLLM(cfg)

    calls: list[dict[str, Any]] = []
    valid_reply = json.dumps(
        {
            "overall_encouragement": "Strong ownership across your migration.",
            "top_strengths": ["Clear metrics.", "Calm execution."],
            "carl_primer": "CARL turns actions into clear L5 impact.",
            "question_reviews": [
                {
                    "question_id": "q1",
                    "title": "Ownership",
                    "prompt": "Tell me about a time...",
                    "carl": {"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                    "strength": "Owned rollout cleanly.",
                    "coaching_tip": "Add a concrete metric.",
                    "carl_reframe": "I led the rollout, cutting errors by [X%].",
                }
            ],
            "retry_drill": {
                "question_id": "q1",
                "focus": "Quantify Result",
                "prompt": "In 45 seconds, add the outcome metric.",
            },
        }
    )

    async def fake_complete(**kwargs: Any) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            return '{"overall_encouragement": "Great work on "migration"!", "question_reviews": []}'
        return valid_reply

    monkeypatch.setattr(llm, "_complete_text", fake_complete)
    result = asyncio.run(
        llm.generate_practice_debrief(
            role_title="Senior Engineer",
            company="Acme",
            interview_type="behavioral",
            target_level="Senior (L5)",
            questions=[{"id": "q1", "title": "Ownership", "prompt": "Tell me about a time..."}],
            turns=[{"role": "candidate", "question_id": "q1", "text": "I owned the rollout."}],
        )
    )
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.35
    assert calls[1]["temperature"] == 0.0
    assert "invalid JSON syntax" in calls[1]["user"]
    assert result["overall_encouragement"] == "Strong ownership across your migration."
