"""My questions: the candidate's own saved questions, one hand-editable pack per interview format."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from fastapi.testclient import TestClient

from mockery.banks import MAX_MY_QUESTIONS, split_pasted_questions
from mockery.server import create_app


def _offline_client(tmp_path: Path) -> TestClient:
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.llm_config.provider = "none"
    app.state.runtime.llm_config.api_key = ""
    return TestClient(app)


def _mine(client: TestClient, category: str) -> list[dict]:
    listed = client.get(f"/api/banks?category={category}&match_active_jd=false").json()["questions"]
    return [q for q in listed if q["mine"]]


def test_a_pasted_list_splits_per_line_or_per_blank_line() -> None:
    assert split_pasted_questions("") == []
    assert split_pasted_questions(
        "1. Tell me about a missed deadline.\n- Why this role?\n• Describe a conflict.\r\n2) What is 3.5 times 2?"
    ) == [
        "Tell me about a missed deadline.",
        "Why this role?",
        "Describe a conflict.",
        "What is 3.5 times 2?",
    ]
    assert split_pasted_questions(
        "Design a ticketing system.\nAssume a 10 minute on-sale spike.\n\n \nDesign a URL shortener."
    ) == [
        "Design a ticketing system.\nAssume a 10 minute on-sale spike.",
        "Design a URL shortener.",
    ]
    assert split_pasted_questions("1. First?\n2. Second?\n\n3. Third?\n   Keep this line with it.\n\n-\n- \n***") == [
        "First?",
        "Second?",
        "Third?\nKeep this line with it.",
    ]


def test_saved_questions_are_listed_by_format_and_skip_repeats(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    kept = client.post(
        "/api/banks/mine",
        json={"category": "behavioral", "text": "Tell me about closing the Q3 books two days early.\nBe specific."},
    ).json()
    assert kept["already_saved"] == 0 and len(kept["added"]) == 1
    assert kept["added"][0]["title"] == "Tell me about closing the Q3 books two days early."
    assert kept["added"][0]["prompt"] == "Tell me about closing the Q3 books two days early.\nBe specific."

    pasted = client.post(
        "/api/banks/mine",
        json={
            "category": "behavioral",
            "as_list": True,
            "text": "Describe a conflict with a peer.\n\nTell me about closing the Q3 books   two days early.\nBe specific.",
        },
    ).json()
    assert [q["prompt"] for q in pasted["added"]] == ["Describe a conflict with a peer."]
    assert pasted["already_saved"] == 1

    again = client.post(
        "/api/banks/mine",
        json={
            "category": "behavioral",
            "as_list": True,
            "text": "Describe a  conflict with a peer.\nA new one?\nWhat does print(\"A\") output?\nWhat does print(\"a\") output?",
        },
    ).json()
    assert [q["prompt"] for q in again["added"]] == [
        "A new one?",
        'What does print("A") output?',
        'What does print("a") output?',
    ]
    assert again["already_saved"] == 1

    assert len(_mine(client, "behavioral")) == 5
    assert _mine(client, "general") == []
    bundled = [q for q in client.get("/api/banks?category=behavioral").json()["questions"] if not q["mine"]]
    assert bundled

    pack_path = tmp_path / "banks" / "my_behavioral.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    assert pack["title"] == "My questions" and pack["category"] == "behavioral"
    assert stat.S_IMODE(pack_path.stat().st_mode) == 0o600


def test_saved_questions_lose_contact_details_only(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    saved = client.post(
        "/api/banks/mine",
        json={
            "category": "general",
            "as_list": True,
            "text": "Why did Priya Raman at Northwind say to email priya.raman@example.com?\nCall (555) 010-0199 first?",
        },
    ).json()
    prompts = [q["prompt"] for q in saved["added"]]
    assert prompts == [
        "Why did Priya Raman at Northwind say to email [email address removed]?",
        "Call [phone number removed] first?",
    ]
    assert saved["removed_details"] == ["email address", "phone number"]
    assert "example.com" not in (tmp_path / "banks" / "my_general.json").read_text(encoding="utf-8")


def test_a_saved_question_opens_the_session_as_written_and_is_never_auto_selected(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    own = "Walk me through the Harbor Clinic scheduling rollout.\nWhat would you change?"
    saved_id = client.post("/api/banks/mine", json={"category": "behavioral", "text": own}).json()["added"][0]["id"]

    picked = client.post(
        "/api/practice/start",
        json={
            "interview_type": "behavioral",
            "question_source": "bank_only",
            "question_count": 3,
            "selected_question_id": saved_id,
        },
    ).json()
    questions = picked["session"]["questions"]
    assert questions[0]["source"] == "custom" and questions[0]["prompt"] == own
    assert [q["source"] for q in questions[1:]] == ["bank", "bank"]
    assert own in picked["opening_turn"]["text"]

    automatic = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "bank_only", "question_count": 6},
    ).json()["session"]["questions"]
    assert len(automatic) == 6
    assert all(q["source"] == "bank" and q["prompt"] != own for q in automatic)

    other_format = client.post(
        "/api/practice/start",
        json={"interview_type": "general", "question_source": "bank_only", "selected_question_id": saved_id},
    ).json()["session"]["questions"]
    assert all(q["prompt"] != own for q in other_format)


def test_removing_a_saved_question(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    added = client.post(
        "/api/banks/mine",
        json={"category": "coding", "as_list": True, "text": "Reverse a linked list.\nMerge two sorted arrays."},
    ).json()["added"]

    assert client.delete(f"/api/banks/mine/general/{added[0]['id']}").status_code == 404
    assert client.delete(f"/api/banks/mine/coding/{added[0]['id']}").json() == {"ok": True}
    assert [q["prompt"] for q in _mine(client, "coding")] == ["Merge two sorted arrays."]
    assert client.delete(f"/api/banks/mine/coding/{added[0]['id']}").status_code == 404
    assert client.delete("/api/banks/mine/behavioral/beh_conflict_01").status_code == 404
    assert client.delete(f"/api/banks/mine/poetry/{added[1]['id']}").status_code == 422


def test_requests_that_cannot_be_saved_change_nothing(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    assert client.post("/api/banks/mine", json={"category": "behavioral", "text": "  \n "}).status_code == 422
    assert client.post("/api/banks/mine", json={"category": "poetry", "text": "Why?"}).status_code == 422
    too_long = client.post("/api/banks/mine", json={"category": "behavioral", "text": "x" * 6001})
    assert too_long.status_code == 422 and "6,000" in too_long.json()["detail"]

    full = "\n".join(f"Question number {n}?" for n in range(MAX_MY_QUESTIONS))
    assert len(
        client.post("/api/banks/mine", json={"category": "general", "as_list": True, "text": full}).json()["added"]
    ) == MAX_MY_QUESTIONS
    over = client.post("/api/banks/mine", json={"category": "general", "text": "One more?"})
    assert over.status_code == 422 and "Remove some" in over.json()["detail"]
    assert len(_mine(client, "general")) == MAX_MY_QUESTIONS

    broken = tmp_path / "banks" / "my_coding.json"
    broken.write_text("{ not json", encoding="utf-8")
    refused = client.post("/api/banks/mine", json={"category": "coding", "text": "Reverse a linked list."})
    assert refused.status_code == 422 and "left untouched" in refused.json()["detail"]
    assert broken.read_text(encoding="utf-8") == "{ not json"
