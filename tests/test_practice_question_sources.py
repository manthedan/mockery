"""Where a Practice session's questions come from: the format's own bank, the level
picked on Home, and (optionally) one question the candidate wrote themselves."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mockery.levels import normalize_level
from mockery.practice import PracticeConfig, PracticeQuestionItem, PracticeSession
from mockery.practice_conversation import _format_question_plan
from mockery.server import create_app


def _offline_client(tmp_path: Path) -> TestClient:
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.llm_config.provider = "none"
    app.state.runtime.llm_config.api_key = ""
    return TestClient(app)


def test_ladder_codes_fold_onto_role_neutral_seniority() -> None:
    assert [normalize_level(code) for code in ("L4", "l5", "L6", "L7", "EM")] == [
        "Mid",
        "Senior",
        "Lead",
        "Lead",
        "Manager",
    ]
    assert normalize_level(" senior ") == "Senior"
    assert normalize_level("Senior (L5)") == "Senior (L5)"
    assert PracticeConfig(target_level="L6").target_level == "Lead"
    assert PracticeConfig(target_level="").target_level == "Senior"


def test_general_sessions_draw_only_from_the_general_bank(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)

    general = client.get("/api/banks?category=general&match_active_jd=false").json()["questions"]
    general_ids = {q["id"] for q in general}
    assert {"gen_company_01", "gen_future_01"} <= general_ids
    assert all(q["category"] == "general" for q in general)

    sess = client.post(
        "/api/practice/start",
        json={"interview_type": "general", "question_source": "bank_only", "question_count": 4},
    ).json()["session"]
    assert len(sess["questions"]) == 4
    assert {q["bank_question_id"] for q in sess["questions"]} <= general_ids
    assert all(q["category"] == "general" for q in sess["questions"])
    assert not any(q["starter_code"] or q["core_requirements"] for q in sess["questions"])


def test_blank_role_and_level_start_role_neutral(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    data = client.post("/api/practice/start", json={"interview_type": "behavioral"}).json()
    sess = data["session"]
    assert sess["role_title"] == "General role"
    assert sess["config"]["target_level"] == "Senior"
    opening = data["opening_turn"]["text"]
    assert "at the Senior level" in opening
    assert "General role" not in opening and "Software Engineer" not in opening

    legacy = client.post(
        "/api/practice/start",
        json={"role_title": "Charge Nurse", "interview_type": "behavioral", "target_level": "EM"},
    ).json()
    assert legacy["session"]["config"]["target_level"] == "Manager"
    assert "for Charge Nurse (Manager)" in legacy["opening_turn"]["text"]


def test_custom_question_opens_the_session_verbatim(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    own = "Design a ticketing system for a 60,000 seat stadium.\nAssume a 10 minute on-sale spike."

    design = client.post(
        "/api/practice/start",
        json={
            "interview_type": "system_design",
            "selected_question_id": "sd_feed_01",
            "custom_question": own,
        },
    ).json()
    questions = design["session"]["questions"]
    assert len(questions) == 1
    q0 = questions[0]
    assert q0["source"] == "custom" and q0["bank_question_id"] is None
    assert q0["prompt"] == own
    assert q0["title"] == "Design a ticketing system for a 60,000 seat stadium."
    assert q0["category"] == "system_design"
    assert own in design["opening_turn"]["text"]

    behavioral = client.post(
        "/api/practice/start",
        json={
            "interview_type": "behavioral",
            "question_source": "bank_only",
            "question_count": 3,
            "custom_question": "Tell me about the time you closed the Q3 books two days early.",
        },
    ).json()["session"]["questions"]
    assert [q["source"] for q in behavioral] == ["custom", "bank", "bank"]

    too_long = client.post(
        "/api/practice/start",
        json={"interview_type": "coding", "custom_question": "x" * 6001},
    )
    assert too_long.status_code == 422


def test_custom_question_loses_contact_details_and_nothing_else(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    own = (
        "Priya Raman at Northwind asked me this in the Lisbon onsite: design a rate limiter for "
        "10.0.0.1 behind https://api.example.com handling 1,000,000 requests a day.\n"
        "Follow up with her at priya.raman@example.com or (555) 010-0199. "
        "Reference 123-45-6789, card 4111 1111 1111 1111."
    )
    started = client.post(
        "/api/practice/start",
        json={"interview_type": "system_design", "custom_question": own},
    ).json()
    prompt = started["session"]["questions"][0]["prompt"]

    for gone in ("priya.raman@example.com", "(555) 010-0199", "123-45-6789", "4111 1111 1111 1111"):
        assert gone not in prompt
        assert gone not in started["opening_turn"]["text"]
    for kept in ("Priya Raman", "Northwind", "Lisbon", "10.0.0.1", "https://api.example.com", "1,000,000"):
        assert kept in prompt
    assert "[email address removed]" in prompt and "[phone number removed]" in prompt
    assert started["removed_details"] == [
        "email address",
        "phone number",
        "Social Security number",
        "card number",
    ]

    plain = "Tell me about the time you cut 2024 onboarding from 14 days to 9 for 300 new hires."
    untouched = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "bank_only", "custom_question": plain},
    ).json()
    assert untouched["session"]["questions"][0]["prompt"] == plain
    assert untouched["removed_details"] == []


def test_system_design_level_bar_reaches_the_interviewer_plan(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)

    def start(level: str) -> dict:
        return client.post(
            "/api/practice/start",
            json={
                "interview_type": "system_design",
                "question_source": "bank_only",
                "target_level": level,
                "selected_question_id": "sd_feed_01",
            },
        ).json()["session"]["questions"][0]

    mid, lead = start("Mid"), start("Lead")
    assert mid["level_bar"] and lead["level_bar"] and mid["level_bar"] != lead["level_bar"]

    session = PracticeSession(
        config=PracticeConfig(interview_type="system_design", target_level="Lead"),
        questions=[PracticeQuestionItem.model_validate(lead)],
    )
    assert f"Depth expected at Lead level: {lead['level_bar']}" in _format_question_plan(session)


def _spy_on_llm(client: TestClient) -> list[str]:
    """Configure an LLM and record every prompt it is asked to complete."""
    llm = client.app.state.runtime.llm
    llm.config.provider = "openai"
    llm.config.api_key = "test-key"
    assert llm.config.is_configured
    asked: list[str] = []

    async def _fake_complete(*, system: str, user: str, **_: object) -> str:
        asked.append(user)
        return (
            '{"opening": "Welcome.", "questions": [{"title": "Ledger Migration", '
            '"competency": "Ownership", "prompt": "Tell me about the ledger migration."}]}'
        )

    llm._complete_text = _fake_complete  # type: ignore[method-assign]
    return asked


def test_nothing_to_tailor_means_no_llm_call_and_bank_questions(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    asked = _spy_on_llm(client)

    for source in ("mixed", "resume_jd_tailored"):
        res = client.post(
            "/api/practice/start",
            json={"role_title": "Charge Nurse", "interview_type": "behavioral", "question_source": source},
        )
        assert res.status_code == 200, res.text
        questions = res.json()["session"]["questions"]
        assert questions and {q["source"] for q in questions} == {"bank"}

    assert asked == []


def test_a_job_listing_is_enough_to_tailor(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    asked = _spy_on_llm(client)
    saved = client.post(
        "/api/profile/jd",
        json={
            "title": "Charge Nurse",
            "company": "Example Health",
            "text": "Lead a 24-bed surgical unit and coordinate staffing across shifts.",
        },
    )
    assert saved.status_code == 200, saved.text

    res = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "resume_jd_tailored"},
    )
    assert res.status_code == 200, res.text
    assert len(asked) == 1 and "surgical unit" in asked[0]
    assert "tailored" in {q["source"] for q in res.json()["session"]["questions"]}


def test_a_question_picked_under_another_format_is_not_carried_over(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    design_id = client.get("/api/banks?category=system_design&match_active_jd=false").json()["questions"][0]["id"]

    res = client.post(
        "/api/practice/start",
        json={"interview_type": "general", "question_source": "bank_only", "selected_question_id": design_id},
    )
    assert res.status_code == 200, res.text
    session = res.json()["session"]
    assert {q["category"] for q in session["questions"]} == {"general"}
    assert session["config"]["selected_question_id"] is None
