from __future__ import annotations

import io
import json
import random
import string
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mockery.llm import _extract_json_object
from mockery.practice import estimate_carl_heuristic
from mockery.profiles import CandidateProfileStore, ResumeRedactionPrefs, _mask_pii_preview
from mockery.prompts import render_prompt_template
from mockery.redactor import RampartRedactor, _resolve_public_http_url, extract_document_text
from mockery.server import create_app
from mockery.state_machines import validate_practice_invariants


def _rand_str(rng: random.Random, min_len: int = 0, max_len: int = 80, alphabet: str | None = None) -> str:
    chars = alphabet or (
        string.ascii_letters
        + string.digits
        + " \n\t{}[]\"'\\/:;,.!?@#$%^&*()-_+=<>`~"
        + "éñü🚀•—–"
    )
    return "".join(rng.choice(chars) for _ in range(rng.randint(min_len, max_len)))


def test_fuzz_extract_json_object_adversarial_inputs() -> None:
    """Fuzz `_extract_json_object` with 500 valid-wrapped and hostile malformed LLM outputs."""
    rng = random.Random(20261121)
    t0 = time.perf_counter()

    # 1. Valid JSON dicts wrapped in noisy LLM envelopes MUST always decode cleanly
    for i in range(250):
        payload = {
            "id": f"k_{i}",
            "text": _rand_str(rng, 0, 60),
            "nested": {"ok": rng.choice([True, False]), "nums": [rng.randint(-999, 999) for _ in range(3)]},
        }
        raw_json = json.dumps(payload, ensure_ascii=False)
        envelope = rng.choice(
            [
                raw_json,
                f"```json\n{raw_json}\n```",
                f"<think>reasoning with {{fake: 'braces'}} and [1,2]</think>\n{raw_json}",
                f"Sure, here is the JSON:\n{raw_json}\nHope that helps!",
                f"<think>nested <think> inner</think>\n```JSON\n{raw_json}\n```\nTrailing note.",
            ]
        )
        got = _extract_json_object(envelope)
        assert isinstance(got, dict)
        assert got["id"] == payload["id"]

    # 2. Hostile / corrupted / deeply nested inputs MUST never hang or raise unexpected error types
    hostile_cases = [
        "",
        "   \n\t  ",
        "[]",
        "12345",
        '"just a string"',
        "{",
        "}" * 500,
        "{" * 250,
        "<think>" + "a" * 5000,
        "<think>a</think>" * 100 + '{"ok": 1}',
        '{"a": "unescaped "quote" inside"}',
        '{"a": 1, "b": }',
        "```json\n{truncated",
        '{"emoji": "🔥", "null": "\x00"}',
    ]
    for _ in range(250):
        hostile_cases.append(_rand_str(rng, 0, 200))

    for raw in hostile_cases:
        try:
            got = _extract_json_object(raw)
            # If recovery succeeded, it must return a JSON value without crashing
            assert got is not None or raw.strip() == "null"
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass

    elapsed = time.perf_counter() - t0
    assert elapsed < 2.5, f"_extract_json_object fuzz took too long ({elapsed:.2f}s) — possible ReDoS"


def test_fuzz_redactor_overlapping_pii_custom_terms_and_doc_extract() -> None:
    """Fuzz RampartRedactor, PII masking, SSRF guard, and document extraction."""
    rng = random.Random(424242)
    redactor = RampartRedactor()
    t0 = time.perf_counter()

    hostile_custom_terms = [
        "",
        "   ",
        "[a-z]+",
        "(unclosed",
        "C++",
        "foo\\bar",
        ".*?",
        "FinScale",
        "Project$Alpha^2",
        "a" * 120,
    ]

    for i in range(180):
        email = f"user.{i}@example-corp{i % 7}.io"
        phone = f"(415) 555-{1000 + (i % 8999)}"
        ssn = f"{100 + (i % 700):03d}-{10 + (i % 80):02d}-{1000 + (i % 8999):04d}"
        url = f"https://github.com/dev{i}/repo?email={email}"
        custom = rng.choice(hostile_custom_terms)
        parts = [
            f"Jordan Vance {i}" if rng.random() < 0.7 else _rand_str(rng, 2, 24),
            f"San Francisco, CA | {email} | {phone} | {url}",
            f"Tax ID {ssn} | IP 192.168.1.{i % 250}",
            f"Worked on {custom} building distributed ledgers." if custom.strip() else "Built APIs.",
            _rand_str(rng, 0, 120),
        ]
        rng.shuffle(parts)
        raw_text = "\n".join(parts)

        prefs_custom = [custom] if custom else []
        res = redactor.redact(
            raw_text,
            keep_location=rng.choice([True, False]),
            hide_employers_and_schools=rng.choice([True, False]),
            custom_terms=prefs_custom,
            unredacted_values=[],
            use_rampart=False,
        )
        assert isinstance(res.redacted_text, str)

        # Invariant 1: spans must be valid, in-bounds, and non-overlapping
        prev_end = -1
        for sp in sorted(res.spans, key=lambda s: (s["start"], s["end"])):
            assert 0 <= sp["start"] < sp["end"] <= len(raw_text)
            assert sp["start"] >= prev_end, f"Overlapping redaction spans: {sp} after prev_end={prev_end}"
            prev_end = sp["end"]
            masked = _mask_pii_preview(sp.get("value", ""), sp.get("label", ""))
            assert isinstance(masked, str) and len(masked) > 0

        # Invariant 2: core high-risk PII (email, phone, SSN) must NEVER leak into redacted_text
        assert email not in res.redacted_text, f"Email leaked in iter {i}: {res.redacted_text}"
        assert phone not in res.redacted_text, f"Phone leaked in iter {i}: {res.redacted_text}"
        assert ssn not in res.redacted_text, f"SSN leaked in iter {i}: {res.redacted_text}"

    # Hostile document bytes (corrupt PDF, corrupt ZIP/DOCX, weird RTF, invalid UTF-8, empty)
    bad_zip_buf = io.BytesIO()
    with zipfile.ZipFile(bad_zip_buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document><unclosed>")
    hostile_docs: list[tuple[str, bytes]] = [
        ("", b""),
        ("empty.pdf", b""),
        ("corrupt.pdf", b"%PDF-1.7\n\x00\xffbroken-stream-not-a-pdf"),
        ("truncated.pdf", b"%PDF-"),
        ("corrupt.docx", b"PK\x03\x04not-a-real-zip"),
        ("badxml.docx", bad_zip_buf.getvalue()),
        ("weird.rtf", b"{\\rtf1\\ansi \\par Hello \\u1234? {{{}}}"),
        ("binary.txt", bytes(rng.getrandbits(8) for _ in range(512))),
        ("ligatures.txt", "ef\ufb01cient work\ufb02ow \x00 null".encode("utf-8")),
    ]
    for fname, raw_bytes in hostile_docs:
        try:
            out = extract_document_text(fname, raw_bytes)
            assert isinstance(out, str)
        except ValueError:
            pass

    # SSRF guard fuzzing — private/local/non-http URLs MUST always be rejected with ValueError
    ssrf_urls = [
        "",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://127.0.0.1:6379/_INFO",
        "http://localhost/jobs",
        "http://localhost.localdomain:8080/a",
        "http://127.0.0.1/admin",
        "http://[::1]/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://192.168.1.1/router",
        "http://0.0.0.0/",
        "http://user:pass@example.com/job",
        "https://service.local/job",
        "https://db.internal/job",
    ]
    for bad_url in ssrf_urls:
        with pytest.raises(ValueError):
            _resolve_public_http_url(bad_url)

    assert (time.perf_counter() - t0) < 4.0


def test_fuzz_prompt_template_and_carl_monotonicity() -> None:
    """Fuzz single-pass placeholder rendering and strict CARL monotonicity."""
    rng = random.Random(9001)

    for _ in range(200):
        a_val = rng.choice(["{b}", "{{b}}", "\\1 $1 {c}", _rand_str(rng, 0, 40)])
        b_val = rng.choice(["{a}", '{"json": 1}', _rand_str(rng, 0, 40)])
        rendered = render_prompt_template(
            '{"schema": true} A={a} B={b}',
            ("a", "b"),
            a=a_val,
            b=b_val,
        )
        assert rendered == f'{{"schema": true}} A={a_val} B={b_val}'

    rank = {"missing": 0, "partial": 1, "strong": 2}
    corpus = [
        "",
        "uh yeah",
        "When I led the legacy Postgres migration at my company, I designed dual-write verification.",
        "Latency dropped by 45% from 180 ms to 99 ms across 50k rps.",
        "I learned to always run shadow diffs first and turned it into a team template.",
        "No metrics here just chatting.",
    ]
    for _ in range(150):
        carl = {"context": "missing", "action": "missing", "result": "missing", "learning": "missing"}
        for _turn in range(6):
            text = rng.choice(corpus) + " " + _rand_str(rng, 0, 40)
            nxt = estimate_carl_heuristic(text, existing=carl)
            for dim in ("context", "action", "result", "learning"):
                assert nxt[dim] in rank
                assert rank[nxt[dim]] >= rank[carl[dim]], f"CARL downgraded on {dim}: {carl} -> {nxt}"
            carl = nxt


def test_fuzz_stateful_practice_api_profile_and_asr_endpoints(tmp_path: Path) -> None:
    """
    Run 160 random interleaved API actions across Practice Studio, Profile/Redaction, and ASR
    with intermittent LLM outages and hostile payloads. Asserts ZERO HTTP 500s and valid FSM invariants.
    """
    rng = random.Random(1337)
    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime

    async def flaky_plan(**kwargs: Any) -> tuple[list[dict[str, Any]], str]:
        if rng.random() < 0.2:
            raise RuntimeError("Fuzzed LLM outage on plan")
        fmt = kwargs.get("interview_type") or "behavioral"
        cnt = max(1, min(4, int(kwargs.get("question_count") or 2)))
        qs = [
            {
                "id": f"pq_{i}_{rng.randint(1, 999)}",
                "source": "bank",
                "category": fmt,
                "title": f"Fuzz Q{i}",
                "competency": "Systems",
                "prompt": f"Solve fuzz scenario {i}.",
            }
            for i in range(cnt)
        ]
        return qs, "Welcome to fuzz practice."

    async def flaky_reply(**kwargs: Any) -> dict[str, Any]:
        if rng.random() < 0.25:
            raise RuntimeError("Fuzzed LLM outage on reply")
        return {
            "interviewer_reply": "Tell me more about the latency tradeoff.",
            "should_advance": rng.choice([True, False]),
            "carl_signals": {
                "context": rng.choice(["strong", "partial", "missing"]),
                "action": rng.choice(["strong", "partial", "missing"]),
                "result": rng.choice(["strong", "partial", "missing"]),
                "learning": rng.choice(["strong", "partial", "missing"]),
            },
            "coach_nudge": {"tone": rng.choice(["encourage", "tip"]), "message": "Keep going!"},
        }

    async def flaky_hint(**kwargs: Any) -> str:
        if rng.random() < 0.25:
            raise RuntimeError("Fuzzed LLM outage on hint")
        return "Focus on the bottleneck first."

    async def flaky_debrief(**kwargs: Any) -> dict[str, Any]:
        if rng.random() < 0.25:
            raise RuntimeError("Fuzzed LLM outage on debrief")
        qs = kwargs.get("questions") or []
        qid = qs[0]["id"] if qs else "pq_0"
        return {
            "overall_encouragement": "Great fuzz rep!",
            "top_strengths": ["Clear thinking"],
            "carl_primer": "CARL structures impact.",
            "question_reviews": [
                {
                    "question_id": qid,
                    "title": "Fuzz Q",
                    "prompt": "Prompt",
                    "carl": {"context": "strong", "action": "partial", "result": "missing", "learning": "missing"},
                    "strength": "Good start.",
                    "coaching_tip": "Add result.",
                    "carl_reframe": "I reduced latency by [X%].",
                }
            ],
            "retry_drill": {"question_id": qid, "focus": "Result", "prompt": "Quantify in 45s."},
        }

    async def flaky_retry(**kwargs: Any) -> dict[str, Any]:
        if rng.random() < 0.25:
            raise RuntimeError("Fuzzed LLM outage on retry")
        return {
            "encouragement": "Nice retry!",
            "what_improved": "Stronger metric.",
            "one_more_polish": "Add learning.",
            "carl_after": {"context": "strong", "action": "strong", "result": "strong", "learning": "partial"},
        }

    rt.llm.generate_practice_plan = flaky_plan
    rt.llm.generate_practice_interviewer_reply = flaky_reply
    rt.llm.generate_practice_hint = flaky_hint
    rt.llm.generate_practice_debrief = flaky_debrief
    rt.llm.evaluate_practice_retry = flaky_retry

    session_ids: list[str] = []

    def _assert_no_500(res: Any, label: str) -> None:
        assert res.status_code != 500, f"Raw 500 on {label}: {res.text}"
        if res.status_code == 503:
            body = res.json()
            assert "detail" in body and body["detail"], f"Non-structured 503 on {label}: {body}"

    with TestClient(app, raise_server_exceptions=False) as client:
        for step in range(180):
            op = rng.choices(
                [
                    "start",
                    "turn",
                    "workspace",
                    "hint",
                    "next",
                    "debrief",
                    "retry",
                    "export_md",
                    "delete",
                    "profile_paste",
                    "profile_file",
                    "profile_prefs",
                    "profile_jd",
                    "asr_pcm",
                ],
                weights=[14, 24, 18, 10, 12, 12, 12, 8, 6, 8, 6, 8, 6, 8],
                k=1,
            )[0]

            if op == "start" or not session_ids:
                res = client.post(
                    "/api/practice/start",
                    json={
                        "role_title": rng.choice(
                            ["", "Staff Engineer", "<img src=x onerror=alert(1)>", "A" * 240]
                        ),
                        "interview_type": rng.choice(["behavioral", "system_design", "coding", "general"]),
                        "question_source": rng.choice(["mixed", "bank_only", "resume_jd_tailored"]),
                        "target_level": rng.choice(["L4", "L5", "L6", "L7", "EM"]),
                        "question_count": rng.randint(1, 4),
                    },
                )
                _assert_no_500(res, f"start step {step}")
                if res.status_code == 200:
                    sid = res.json()["session"]["id"]
                    session_ids.append(sid)
            else:
                sid = rng.choice(session_ids)
                if op == "turn":
                    res = client.post(
                        f"/api/practice/{sid}/turn",
                        json={
                            "text": rng.choice(["", "   ", _rand_str(rng, 1, 180), "```js\nwhile(1){}\n```"]),
                            "expected_question_id": rng.choice([None, "stale_qid_999"]),
                            "workspace_context": rng.choice(
                                [
                                    None,
                                    {"code": "```py\nprint(1)\n```", "language": "python;rm -rf /"},
                                    {"diagram_graph": {"nodes": [{"id": "n1", "x": -9999, "y": 99999}], "edges": []}},
                                ]
                            ),
                        },
                    )
                    _assert_no_500(res, f"turn step {step}")
                elif op == "workspace":
                    # Hostile diagram graphs, huge node counts, self-loops, XSS labels, non-numeric coords
                    hostile_nodes = [
                        {
                            "id": f"n{k}",
                            "type": rng.choice(["service", "db", "unknown_type", ""]),
                            "label": rng.choice(["<script>alert(1)</script>", _rand_str(rng, 0, 120)]),
                            "note": _rand_str(rng, 0, 200),
                            "x": rng.choice([-500, 0, 420, 99999, "bad_x"]),
                            "y": rng.choice([-200, 100, 99999, None]),
                        }
                        for k in range(rng.choice([0, 3, 60]))
                    ]
                    hostile_edges = [
                        {"id": f"e{k}", "from": "n0", "to": rng.choice(["n0", "n1", "missing"]), "label": _rand_str(rng, 0, 90)}
                        for k in range(rng.choice([0, 4, 25]))
                    ]
                    res = client.post(
                        f"/api/practice/{sid}/workspace",
                        json={
                            "workspace_context": {
                                "diagram_graph": {"version": 1, "nodes": hostile_nodes, "edges": hostile_edges},
                                "diagram_summary": _rand_str(rng, 0, 300),
                                "code": "````\n" + _rand_str(rng, 0, 400) + "\n````",
                                "language": rng.choice(["javascript", "python", "../../../etc/passwd"]),
                                "test_output": _rand_str(rng, 0, 200),
                            }
                        },
                    )
                    _assert_no_500(res, f"workspace step {step}")
                elif op == "hint":
                    res = client.post(f"/api/practice/{sid}/hint", json={})
                    _assert_no_500(res, f"hint step {step}")
                elif op == "next":
                    res = client.post(
                        f"/api/practice/{sid}/next",
                        json={"expected_question_id": rng.choice([None, "wrong_id"])},
                    )
                    _assert_no_500(res, f"next step {step}")
                elif op == "debrief":
                    res = client.post(f"/api/practice/{sid}/debrief")
                    _assert_no_500(res, f"debrief step {step}")
                elif op == "retry":
                    res = client.post(
                        f"/api/practice/{sid}/retry",
                        json={"answer_text": rng.choice(["", _rand_str(rng, 5, 160)])},
                    )
                    _assert_no_500(res, f"retry step {step}")
                elif op == "export_md":
                    res = client.get(f"/api/practice/{sid}/markdown")
                    _assert_no_500(res, f"markdown step {step}")
                elif op == "delete":
                    res = client.delete(f"/api/practice/sessions/{sid}")
                    _assert_no_500(res, f"delete step {step}")
                    if res.status_code == 200 and sid in session_ids:
                        session_ids.remove(sid)
                elif op == "profile_paste":
                    res = client.post(
                        "/api/profile/resume/text",
                        json={
                            "filename": rng.choice(["", "../../evil.pdf", "resume.txt"]),
                            "text": f"Alex Rivera\nalex.{step}@acme.dev | (415) 555-0199\n" + _rand_str(rng, 0, 120),
                        },
                    )
                    _assert_no_500(res, f"profile_paste step {step}")
                elif op == "profile_file":
                    fname, payload = rng.choice(
                        [
                            ("corrupt.pdf", b"%PDF-1.7\n\x00\xffcorrupt"),
                            ("bad.docx", b"PK\x03\x04broken"),
                            ("ok.txt", f"Taylor Kim\ntaylor.{step}@corp.io | 415-555-0144\nBuilt ledgers.".encode()),
                        ]
                    )
                    res = client.post(
                        "/api/profile/resume/file",
                        files={"file": (fname, payload, "application/octet-stream")},
                    )
                    _assert_no_500(res, f"profile_file step {step}")
                elif op == "profile_prefs":
                    res = client.post(
                        "/api/profile/resume/redact",
                        json={
                            "keep_location": rng.choice([True, False]),
                            "hide_employers_and_schools": rng.choice([True, False]),
                            "custom_terms": [rng.choice(["[a-z(", "C++", "Acme", "", ".*"])],
                            "unredacted_values": [rng.choice(["", "alex@acme.dev"])],
                            "use_rampart": False,
                        },
                    )
                    _assert_no_500(res, f"profile_prefs step {step}")
                elif op == "profile_jd":
                    res = client.post(
                        "/api/profile/jd",
                        json=rng.choice(
                            [
                                {"url": "http://127.0.0.1/secret", "text": ""},
                                {"url": "http://169.254.169.254/latest/meta-data/", "text": ""},
                                {"title": "Staff Eng", "company": "Stripe", "text": _rand_str(rng, 0, 180)},
                            ]
                        ),
                    )
                    _assert_no_500(res, f"profile_jd step {step}")
                elif op == "asr_pcm":
                    raw_pcm = bytes(rng.getrandbits(8) for _ in range(rng.choice([0, 1, 3, 17, 640])))
                    res = client.post(
                        "/api/asr/transcribe",
                        content=raw_pcm,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    _assert_no_500(res, f"asr_pcm step {step}")

        # Final invariant audit: every surviving session in store must pass FSM invariants & markdown export!
        for sid in list(session_ids):
            sess = rt.practice_store.get(sid)
            if sess is not None:
                violations = validate_practice_invariants(sess)
                assert violations == [], f"FSM invariant violation on {sid}: {violations}"
                assert sess.phase != "debriefing", f"Session {sid} stuck in debriefing after fuzzing!"
                md = sess.to_markdown()
                assert isinstance(md, str) and len(md) > 0
