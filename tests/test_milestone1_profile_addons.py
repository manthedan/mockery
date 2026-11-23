from __future__ import annotations

import asyncio
import io
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mockery.addons import AddonManager
from mockery.banks import QuestionBankStore
from mockery.profiles import CandidateProfileStore, ResumeRedactionPrefs
from mockery.redactor import (
    RampartRedactor,
    clean_job_listing_text,
    extract_document_text,
    fetch_job_listing,
    parse_job_title_and_company,
)
from mockery.server import create_app


SYNTHETIC_RESUME = """Jordan Rivera
Austin, TX | jordan.rivera@example.org | (415) 555-0192 | https://github.com/jrivera-demo
742 Evergreen Terrace, Austin, TX 78701

EXPERIENCE
Staff Software Engineer | Acme Cloud Systems (2021 - Present)
- Designed a multi-region event ingestion pipeline handling 250k events/sec in Python, TypeScript, and PostgreSQL.
- Reduced p99 latency on Project Nighthawk from 420ms to 65ms using Redis caching and query batching.

Senior Software Engineer | Northstar Analytics (2018 - 2021)
- Led migration from monolithic GraphQL service to federated TypeScript microservices.

EDUCATION
State University — B.S. Computer Science (2018)
"""

SYNTHETIC_JD = """Senior Software Engineer, Full-stack

About the role
Build responsive TypeScript and React interfaces backed by high-throughput Python and PostgreSQL APIs.
You will own end-to-end streaming inference UIs, rate limiting, and distributed observability.

Qualifications
- 5+ years building production full-stack applications in TypeScript, React, and Python
- Strong experience with distributed systems, caching, and web security

Equal Opportunity Employer
We are an equal opportunity employer and value diversity at our company. All qualified applicants
will receive consideration for employment without regard to race, color, religion, or national origin.
"""


def _make_synthetic_docx(paragraphs: list[str]) -> bytes:
    buf = io.BytesIO()
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_extract_document_text_txt_and_docx() -> None:
    txt = extract_document_text("resume.txt", SYNTHETIC_RESUME.encode("utf-8"))
    assert "Jordan Rivera" in txt
    assert "Acme Cloud Systems" in txt

    docx_bytes = _make_synthetic_docx(["Jordan Rivera", "Senior Engineer at Acme Cloud Systems"])
    extracted = extract_document_text("resume.docx", docx_bytes)
    assert "Jordan Rivera" in extracted
    assert "Acme Cloud Systems" in extracted


def test_redactor_rules_and_options() -> None:
    redactor = RampartRedactor()
    res = redactor.redact(
        SYNTHETIC_RESUME,
        keep_location=True,
        hide_employers_and_schools=False,
        custom_terms=["Project Nighthawk"],
        use_rampart=False,
    )
    assert "jordan.rivera@example.org" not in res.redacted_text
    assert "415" not in res.redacted_text
    assert "github.com/jrivera-demo" not in res.redacted_text
    assert "742 Evergreen Terrace" not in res.redacted_text
    assert "Jordan Rivera" not in res.redacted_text
    assert "Project Nighthawk" not in res.redacted_text
    # Location preserved when keep_location=True
    assert "Austin" in res.redacted_text
    # Employers preserved when hide_employers_and_schools=False
    assert "Acme Cloud Systems" in res.redacted_text
    assert "[EMAIL_1]" in res.redacted_text
    assert "[PHONE_1]" in res.redacted_text
    assert "[CUSTOM_1]" in res.redacted_text

    # Full anonymization: hide employers, schools, and locations
    res_anon = redactor.redact(
        SYNTHETIC_RESUME,
        keep_location=False,
        hide_employers_and_schools=True,
        use_rampart=False,
    )
    assert "Acme Cloud Systems" not in res_anon.redacted_text
    assert "State University" not in res_anon.redacted_text
    assert "78701" not in res_anon.redacted_text

    # Unredacting a specific value brings it back cleanly (even with irregular spacing)
    res_unredacted = redactor.redact(
        SYNTHETIC_RESUME,
        keep_location=True,
        custom_terms=["Project Nighthawk"],
        unredacted_values=["  Project   Nighthawk  "],
        use_rampart=False,
    )
    assert "Project Nighthawk" in res_unredacted.redacted_text
    assert "jordan.rivera@example.org" not in res_unredacted.redacted_text


def test_redactor_rampart_onnx_if_cached() -> None:
    redactor = RampartRedactor()
    if not redactor.is_downloaded():
        pytest.skip("Rampart ONNX model not downloaded in local cache")
    res = redactor.redact(
        SYNTHETIC_RESUME,
        keep_location=True,
        hide_employers_and_schools=False,
        use_rampart=True,
    )
    assert res.engine == "rampart+rules"
    assert "Jordan Rivera" not in res.redacted_text
    assert "jordan.rivera@example.org" not in res.redacted_text
    assert "Python" in res.redacted_text
    assert "TypeScript" in res.redacted_text

    # Regression: smart scrubber must never miss the candidate's header name when Rampart
    # emits an early PERSON_NAME (e.g. "Francisco" inside "San Francisco, CA" or a partial/body hit),
    # must keep "San Francisco, CA" intact when keep_location=True, and must not mis-tag
    # decimal availability percentages or lowercase+product phrases.
    sf_resume = (
        "Jordan Vance\n"
        "San Francisco, CA | jordan.vance@finscale.io | (415) 555-0194 | https://github.com/jordanvance\n\n"
        "EXPERIENCE\n"
        "Staff Backend Engineer, FinScale Payments (2022–Present)\n"
        "- Architected a multi-region double-entry ledger handling 48,000 TPS at 99.995% availability.\n"
        "- Led zero-downtime migration from monolithic Postgres settlement tables to sharded Spanner partitions.\n"
    )
    sf_res = redactor.redact(
        sf_resume,
        keep_location=True,
        hide_employers_and_schools=False,
        use_rampart=True,
    )
    assert sf_res.engine == "rampart+rules"
    assert "Jordan Vance" not in sf_res.redacted_text
    assert sf_res.redacted_text.startswith("[PERSON_NAME_1]\n")
    assert "San Francisco, CA" in sf_res.redacted_text
    assert "sharded Spanner" in sf_res.redacted_text
    assert "99.995%" in sf_res.redacted_text

    # Regression: single-line header contact, Markdown heading, newline boundary, and repeated bio name
    for sample, must_hide, must_keep in [
        (
            "Alex Rivera | alex@example.com | (206) 555-0111\n\nEXPERIENCE\nSoftware Engineer",
            ["Alex", "Rivera", "alex@example.com"],
            ["Software Engineer"],
        ),
        (
            "Elena Rostova\nSeattle, WA | elena@example.com\n\nEXPERIENCE\nStaff Engineer",
            ["Elena", "Rostova", "elena@example.com"],
            ["Seattle, WA"],
        ),
        (
            "# Jordan Rivera\nAustin, TX | jordan@example.org\n\nSUMMARY\nJordan Rivera is a Staff Engineer.",
            ["Jordan", "Rivera", "jordan@example.org"],
            ["Austin, TX", "Staff Engineer"],
        ),
    ]:
        out = redactor.redact(sample, keep_location=True, use_rampart=True)
        for secret in must_hide:
            assert secret not in out.redacted_text, f"{secret!r} leaked in {out.redacted_text!r}"
        for keep in must_keep:
            assert keep in out.redacted_text, f"{keep!r} unexpectedly scrubbed in {out.redacted_text!r}"


def test_redactor_header_name_not_suppressed_by_early_rampart_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even without cached ONNX weights, verify early Rampart hits never suppress header-name redaction."""
    from mockery.redactor import DetectedSpan

    redactor = RampartRedactor()
    sample = (
        "Jordan Vance\n"
        "San Francisco, CA | jordan.vance@finscale.io | (415) 555-0194\n\n"
        "SUMMARY\n"
        "Jordan Vance builds payment ledgers.\n"
    )
    monkeypatch.setattr(redactor, "is_downloaded", lambda: True)
    monkeypatch.setattr(
        redactor,
        "_run_rampart_onnx",
        lambda _text: [DetectedSpan(17, 26, "PERSON_NAME", "rampart")],  # "Francisco"
    )
    redactor._session = object()  # mark engine as rampart+rules for test

    res = redactor.redact(sample, keep_location=True, use_rampart=True)
    assert res.engine == "rampart+rules"
    assert "Jordan Vance" not in res.redacted_text
    assert res.redacted_text.count("[PERSON_NAME_1]") == 2
    assert "San Francisco, CA" in res.redacted_text


def test_jd_cleaner_and_ssrf_guard() -> None:
    cleaned = clean_job_listing_text(SYNTHETIC_JD)
    assert "Build responsive TypeScript" in cleaned
    assert "Equal Opportunity Employer" not in cleaned

    title, company = parse_job_title_and_company(
        "Job Application for Senior Software Engineer, Full-stack at ExampleCo",
        url="https://job-boards.greenhouse.io/exampleco/jobs/12345",
        text=cleaned,
    )
    assert "Senior Software Engineer" in title
    assert company.lower() == "exampleco"

    for bad_url in [
        "file:///etc/passwd",
        "http://127.0.0.1:8765/api/status",
        "http://localhost:8080/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/secret",
    ]:
        with pytest.raises(ValueError):
            asyncio.run(fetch_job_listing(bad_url))


def test_question_banks_and_jd_ranking(tmp_path: Path) -> None:
    # Also add a user-contributed bank file to verify custom loading
    custom_dir = tmp_path / "banks"
    custom_dir.mkdir(parents=True)
    (custom_dir / "custom_test.json").write_text(
        json.dumps(
            {
                "pack_id": "custom_test",
                "title": "Custom Test Pack",
                "category": "behavioral",
                "questions": [
                    {
                        "id": "custom_01",
                        "competency": "Ownership",
                        "levels": ["L5", "L6"],
                        "prompt": "Tell me about a time you built a streaming inference UI.",
                        "jd_tags": ["typescript", "react", "streaming", "inference"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = QuestionBankStore(user_banks_dir=custom_dir)
    packs = store.list_packs()
    pack_ids = {p["pack_id"] for p in packs}
    assert {"behavioral_leadership_core", "system_design_core", "coding_practical_core", "custom_test"} <= pack_ids

    beh = store.list_questions(category="behavioral")
    sd = store.list_questions(category="system_design")
    coding = store.list_questions(category="coding")
    assert len(beh) == 21  # 20 built-in + 1 custom
    assert len(sd) == 10
    assert len(coding) == 8

    # JD ranking boosts questions matching TypeScript/React/streaming/rate limiting
    ranked = store.list_questions(category="system_design", jd_text=SYNTHETIC_JD)
    assert ranked[0]["jd_relevance_score"] > 0

    # Ladder codes from older user packs and saved profiles fold onto role-neutral seniority
    lead_only = store.list_questions(level="Lead")
    assert lead_only and all("Lead" in q.get("levels", []) for q in lead_only)
    assert [q["id"] for q in store.list_questions(level="L6")] == [q["id"] for q in lead_only]
    custom = next(q for q in lead_only if q["id"] == "custom_01")
    assert custom["levels"] == ["Senior", "Lead"]

    # Every bundled question is told apart by its own title in the picker and the bank browser
    bundled_titles = [q["title"] for q in beh + sd + coding if q["id"] != "custom_01"]
    assert all(bundled_titles) and len(set(bundled_titles)) == len(bundled_titles)
    assert not any(code in q.get("levels", []) for q in beh + sd + coding for code in ("L4", "L5", "L6", "L7", "EM"))


def test_profile_store_permissions_and_llm_safety(tmp_path: Path) -> None:
    store = CandidateProfileStore(tmp_path / "profiles", RampartRedactor())
    prof = store.set_resume(
        filename="jordan.txt",
        raw_text=SYNTHETIC_RESUME,
        prefs=ResumeRedactionPrefs(use_rampart=False),
    )
    assert prof.redacted_count >= 4
    llm_resume = store.resume_for_llm()
    assert "jordan.rivera@example.org" not in llm_resume
    assert "Jordan Rivera" not in llm_resume
    assert "Acme Cloud Systems" in llm_resume

    # public_dict never leaks raw_text
    pub = store.public_dict()
    assert pub["resume"] is not None
    assert "raw_text" not in pub["resume"]

    # File permissions are 0600 on POSIX
    mode = stat.S_IMODE(os.stat(store.profile_path).st_mode)
    assert mode == 0o600


def test_milestone1_rest_api_endpoints(tmp_path: Path) -> None:
    app = create_app(project_root=tmp_path, preload_asr=False)
    with TestClient(app) as client:
        # 1. Status includes addons and profile_summary
        st = client.get("/api/status").json()
        assert "addons" in st
        assert "profile_summary" in st
        assert st["profile_summary"]["has_resume"] is False
        assert st["addons"]["modules_map"]["practice_mode"] is True

        # 2. Upload resume via file (DOCX)
        docx_bytes = _make_synthetic_docx(
            [
                "Jordan Rivera",
                "Austin, TX | jordan.rivera@example.org | (415) 555-0192",
                "Senior Software Engineer | Acme Cloud Systems (2020 - Present)",
            ]
        )
        up = client.post(
            "/api/profile/resume/file",
            files={"file": ("jordan_resume.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert up.status_code == 200
        up_data = up.json()
        assert up_data["resume"]["filename"] == "jordan_resume.docx"
        assert "raw_text" not in up_data["resume"]
        assert "jordan.rivera@example.org" not in up_data["resume"]["redacted_text"]

        # 3. Update redaction preferences (hide employers & add custom term)
        red = client.post(
            "/api/profile/resume/redact",
            json={
                "keep_location": True,
                "hide_employers_and_schools": True,
                "custom_terms": ["Senior"],
                "unredacted_values": [],
                "use_rampart": False,
            },
        )
        assert red.status_code == 200
        red_data = red.json()
        assert "Acme Cloud Systems" not in red_data["resume"]["redacted_text"]
        assert "[CUSTOM_1]" in red_data["resume"]["redacted_text"]

        # 4. Import Job Listing via pasted text + reject SSRF URL
        bad_jd = client.post("/api/profile/jd", json={"url": "http://127.0.0.1/admin"})
        assert bad_jd.status_code == 400

        jd_res = client.post(
            "/api/profile/jd",
            json={
                "title": "Senior Software Engineer, Full-stack",
                "company": "ExampleCo",
                "text": SYNTHETIC_JD,
                "target_level": "L5",
            },
        )
        assert jd_res.status_code == 200
        jd_data = jd_res.json()
        jd_id = jd_data["imported_jd"]["id"]
        assert jd_data["active_jd_id"] == jd_id
        assert "Equal Opportunity Employer" not in jd_data["imported_jd"]["text"]

        # 5. Question banks API ranks by active JD
        banks = client.get("/api/banks?category=system_design&match_active_jd=true").json()
        assert banks["matched_jd_id"] == jd_id
        assert len(banks["questions"]) == 10

        # 6. Add-Ons module toggle persists
        mod_res = client.post(
            "/api/addons/modules",
            json={"modules": {"coding_studio": False}},
        )
        assert mod_res.status_code == 200
        assert mod_res.json()["modules_map"]["coding_studio"] is False

        reloaded = AddonManager(tmp_path, RampartRedactor())
        assert reloaded.status_dict()["modules_map"]["coding_studio"] is False

        # 7. Delete JD & Resume
        assert client.delete(f"/api/profile/jd/{jd_id}").status_code == 200
        assert client.delete("/api/profile/resume").status_code == 200
        final_prof = client.get("/api/profile").json()
        assert final_prof["resume"] is None
        assert final_prof["job_listings"] == []


def test_codex_security_and_correctness_regressions(tmp_path: Path) -> None:
    # 1. Overlapping custom term inside an email must NEVER expose the email local part
    redactor = RampartRedactor()
    res = redactor.redact(
        SYNTHETIC_RESUME,
        keep_location=False,
        custom_terms=["example.org"],
        use_rampart=False,
    )
    assert "jordan.rivera@" not in res.redacted_text
    assert "example.org" not in res.redacted_text
    # 2. keep_location=False scrubs City, ST even in rules-only mode
    assert "Austin, TX" not in res.redacted_text

    # 3. public_dict masks hidden PII and never exposes raw original strings for active spans
    store = CandidateProfileStore(tmp_path / "profiles", redactor)
    store.set_resume(
        filename="jordan.txt",
        raw_text=SYNTHETIC_RESUME,
        prefs=ResumeRedactionPrefs(use_rampart=False),
    )
    pub = store.public_dict()
    spans = pub["resume"]["spans"]
    assert spans
    for sp in spans:
        assert "original" not in sp
        assert "display_value" in sp
    raw_json = json.dumps(pub)
    assert "jordan.rivera@example.org" not in raw_json

    # 4. Server-side span toggle works without exposing hidden PII
    email_span = next(sp for sp in spans if sp["label"] == "EMAIL")
    toggled = store.toggle_span_redaction(email_span["id"])
    assert toggled is not None
    assert "jordan.rivera@example.org" in toggled.redacted_text
    toggled_back = store.toggle_span_redaction(email_span["id"])
    assert toggled_back is not None
    assert "jordan.rivera@example.org" not in toggled_back.redacted_text

    # 5. Superseded redaction cannot resurrect a deleted resume
    orig_redact = store.redactor.redact

    def _delete_mid_redact(text: str, **kwargs):
        store.clear_resume()
        return orig_redact(text, **kwargs)

    store.redactor.redact = _delete_mid_redact  # type: ignore[method-assign]
    try:
        store.update_redaction_prefs(ResumeRedactionPrefs(hide_employers_and_schools=True, use_rampart=False))
    finally:
        store.redactor.redact = orig_redact  # type: ignore[method-assign]
    assert store.state.resume is None

    # 6. Loopback Host header validation rejects foreign Host (inbound DNS rebinding defense)
    app = create_app(project_root=tmp_path / "app", preload_asr=False)
    with TestClient(app) as client:
        evil = client.get("/api/profile", headers={"host": "attacker.example.com"})
        assert evil.status_code == 403
        ok = client.get("/api/profile", headers={"host": "127.0.0.1:8765"})
        assert ok.status_code == 200

