"""Regressions found in review of custom questions, My questions, and thinking-model handling."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from mockery.banks import QuestionBankStore
from mockery.practice import (
    PracticeConfig,
    PracticeQuestionItem,
    PracticeSession,
    PracticeTurn,
    estimate_carl_heuristic,
    said_too_little,
)
from mockery.redactor import RampartRedactor
from mockery.rubrics import CODING, GENERAL
from mockery.server import create_app

ROOT = Path(__file__).resolve().parents[1]
EMAIL = "jordan.lee@example.com"
PHONE = "(555) 010-0142"


def _client(tmp_path: Path, *, configured: bool = False) -> TestClient:
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.llm_config.provider = "openai" if configured else "none"
    app.state.runtime.llm_config.api_key = "test-key" if configured else ""
    return TestClient(app)


class _RecordingModel:
    """Stands in for the provider: records every message sent and answers with fixed JSON."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = json.dumps(reply)
        self.sent: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.sent.extend(str(message.get("content", "")) for message in kwargs["messages"])
        message = SimpleNamespace(content=self.reply, model_extra={})
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop", message=message)])


def _connect(client: TestClient, reply: dict[str, Any]) -> _RecordingModel:
    model = _RecordingModel(reply)
    client.app.state.runtime.llm._client = lambda: model  # type: ignore[attr-defined]
    return model


INTERVIEWER_REPLY = {
    "interviewer_reply": "What did you do first?",
    "should_advance": False,
    "carl_signals": {"context": "strong", "action": "partial", "result": "strong", "learning": "partial"},
    "coach_nudge": {"tone": "tip", "message": "Name the first thing you did."},
}


def test_phone_and_id_numbers_written_without_separators_are_removed() -> None:
    strip = RampartRedactor().strip_contact_details
    cleaned, removed = strip("Call 2025550100 or +44 20 7946 0958. Synthetic SSN 123456789. Card 4111111111111111.")
    assert cleaned == (
        "Call [phone number removed] or [phone number removed]. "
        "Synthetic SSN [Social Security number removed]. Card [card number removed]."
    )
    assert removed == ["phone number", "Social Security number", "card number"]


def test_numbers_and_links_in_a_technical_question_are_left_alone() -> None:
    strip = RampartRedactor().strip_contact_details
    for question in (
        "What is 1000000000009 modulo 11?",
        "Read https://example.com/releases/123.123.1234 and summarise the change.",
        "Design for 1000000000 users with n up to 2147483647 behind 10.0.0.1 by 2026-11-23.",
        "Sort 4 5 3 9 1 2 0 0 0 0 0 0 0 0 0 0 in place.",
    ):
        assert strip(question) == (question, [])
    assert strip("Pay with 4111 1111 1111 1111 today")[0] == "Pay with [card number removed] today"


def test_a_saved_questions_file_stays_private_whatever_its_contents_say(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post(
        "/api/profile/jd",
        json={
            "title": "Harbor Clinic Scheduling Lead",
            "company": "Harbor Clinic",
            "text": "Own the Harbor Clinic scheduling rollout across twelve sites and the nurse rota migration.",
        },
    )
    own = f"Walk me through the Harbor Clinic scheduling rollout and nurse rota migration. Ask {EMAIL} first?"
    pack = {
        "pack_id": "interview_notes",
        "category": "general",
        "questions": [
            {
                "id": "note_1",
                "prompt": own,
                "jd_tags": ["scheduling", "rollout", "rota", "migration", "harbor", "clinic"],
            }
        ],
    }
    banks = tmp_path / "banks"
    banks.mkdir(exist_ok=True)
    (banks / "my_behavioral.json").write_text(json.dumps(pack), encoding="utf-8")

    listed = client.get("/api/banks?category=behavioral").json()["questions"]
    assert listed[0]["id"] == "note_1" and listed[0]["mine"] is True
    assert all(q["id"] != "note_1" for q in client.get("/api/banks?category=general").json()["questions"])

    automatic = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "bank_only", "question_count": 6},
    ).json()
    assert all("Harbor Clinic scheduling rollout" not in q["prompt"] for q in automatic["session"]["questions"])

    picked = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "bank_only", "selected_question_id": "note_1"},
    ).json()
    first = picked["session"]["questions"][0]
    assert first["source"] == "custom" and EMAIL not in first["prompt"]
    assert picked["removed_details"] == ["email address"]


def test_an_edit_made_while_saving_is_kept(tmp_path: Path) -> None:
    store = QuestionBankStore(tmp_path)
    store.add_my_questions("behavioral", ["Tell me about a missed deadline."])
    path = tmp_path / "my_behavioral.json"
    edits: list[str] = []

    def add_after_a_hand_edit(data: dict[str, Any]) -> int:
        if not edits:
            by_hand = json.loads(path.read_text(encoding="utf-8"))
            by_hand["questions"].append({"id": "by_hand", "prompt": "Typed straight into the file?"})
            path.write_text(json.dumps(by_hand), encoding="utf-8")
            edits.append("done")
        data["questions"].append({"id": "by_app", "prompt": "Added through Mockery?"})
        return len(data["questions"])

    assert store._change_my_pack("behavioral", add_after_a_hand_edit) == 3
    saved = [q["id"] for q in json.loads(path.read_text(encoding="utf-8"))["questions"]]
    assert saved[1:] == ["by_hand", "by_app"]
    assert not list(tmp_path.glob(".my_behavioral.*.tmp"))

    path.write_text("{ unfinished", encoding="utf-8")
    assert not store.list_questions(category="behavioral")[-1].get("mine")


def test_older_sessions_and_spoken_answers_reach_the_model_without_contact_details(tmp_path: Path) -> None:
    client = _client(tmp_path, configured=True)
    model = _connect(client, INTERVIEWER_REPLY)
    question = PracticeQuestionItem(
        source="custom",
        category="behavioral",
        title="Rollout",
        competency="Your question",
        prompt=f"Tell me about the rollout. Loop in {EMAIL} if needed.",
        status="active",
    )
    session = PracticeSession(
        role_title=f"Scheduling Lead ({PHONE})",
        config=PracticeConfig(interview_type="behavioral"),
        phase="active",
        questions=[question],
        turns=[PracticeTurn(role="interviewer", question_id=question.id, text=question.prompt)],
    )
    client.app.state.runtime.practice_store.save(session)  # type: ignore[attr-defined]

    turn = client.post(
        f"/api/practice/{session.id}/turn", json={"text": f"I led it and my number is {PHONE}, or {EMAIL}."}
    )
    assert turn.status_code == 200, turn.text
    assert client.post(f"/api/practice/{session.id}/hint", json={}).status_code == 200
    assert client.post(f"/api/practice/{session.id}/debrief", json={}).status_code == 200
    client.post(f"/api/practice/{session.id}/retry", json={"answer_text": f"Shorter: I led it. Reach me at {EMAIL}."})

    assert len(model.sent) >= 6
    assert not [text for text in model.sent if EMAIL in text or PHONE in text]
    assert any("[email address removed]" in text for text in model.sent)

    started = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "bank_only", "role_title": f"Lead, call {PHONE}"},
    ).json()["session"]
    assert PHONE not in started["role_title"]


def test_saved_questions_that_share_an_id_are_picked_and_removed_by_format(tmp_path: Path) -> None:
    client = _client(tmp_path)
    banks = tmp_path / "banks"
    banks.mkdir(exist_ok=True)

    def pack(prompts: list[str]) -> str:
        return json.dumps({"questions": [{"id": "shared", "prompt": prompt} for prompt in prompts]})

    (banks / "my_behavioral.json").write_text(pack(["Behavioral one?"]), encoding="utf-8")
    (banks / "my_coding.json").write_text(pack(["Coding one?", "Coding two?"]), encoding="utf-8")

    coding = client.post(
        "/api/practice/start", json={"interview_type": "coding", "selected_question_id": "shared"}
    ).json()["session"]["questions"][0]
    assert coding["source"] == "custom" and coding["prompt"] == "Coding one?"

    assert client.delete("/api/banks/mine/coding/shared").json() == {"ok": True}
    assert [q["prompt"] for q in json.loads((banks / "my_coding.json").read_text())["questions"]] == ["Coding two?"]
    assert json.loads((banks / "my_behavioral.json").read_text())["questions"][0]["prompt"] == "Behavioral one?"

    (banks / "my_behavioral.json").write_text("{ unfinished", encoding="utf-8")
    assert client.delete("/api/banks/mine/coding/shared").json() == {"ok": True}
    assert (banks / "my_behavioral.json").read_text() == "{ unfinished"


def test_ungraded_sessions_never_ask_for_store_or_replay_grades(tmp_path: Path) -> None:
    client = _client(tmp_path, configured=True)
    model = _connect(client, INTERVIEWER_REPLY)
    session = client.post(
        "/api/practice/start",
        json={"interview_type": "behavioral", "question_source": "bank_only", "answer_framework": "none"},
    ).json()["session"]
    answer = "When I joined, our team had an outage every quarter and I led the migration that fixed it."
    for _ in range(2):
        assert client.post(f"/api/practice/{session['id']}/turn", json={"text": answer}).status_code == 200

    stored = client.app.state.runtime.practice_store.get(session["id"])  # type: ignore[attr-defined]
    replies = [m.content for m in stored.llm_conversation.messages if m.role == "assistant"]
    assert replies and all('"strong"' not in reply and '"partial"' not in reply for reply in replies)
    assert all(turn.carl_signals is None for turn in stored.turns)

    model.sent.clear()
    client.post(f"/api/practice/{session['id']}/debrief", json={})
    client.post(f"/api/practice/{session['id']}/retry", json={"answer_text": "We cut outages by 80% in a quarter."})
    assert model.sent
    assert all("strong|partial|missing" not in text and '"strong"' not in text for text in model.sent)


def test_terse_answers_count_and_denials_earn_nothing_offline() -> None:
    assert said_too_little("Fully remote") is False
    assert said_too_little("O(n)") is False
    assert said_too_little("Um, hmm. Sorry, pass.") is True
    assert estimate_carl_heuristic("Fully remote", rubric=GENERAL)["context"] != "missing"
    assert estimate_carl_heuristic("O(n)", rubric=CODING)["result"] == "strong"

    denied = "I did not check empty inputs or duplicates, and I do not know whether it works or passes"
    assert set(estimate_carl_heuristic(denied, rubric=CODING).values()) == {"missing"}
    affirmed = "I did not sort it, but it handles empty input and duplicates and passes the sample."
    assert estimate_carl_heuristic(affirmed, rubric=CODING)["learning"] == "strong"


def test_inline_code_is_not_cut_in_two_for_speech() -> None:
    script = """
import { splitForSpeech } from "./src/mockery/static/voice_clip.js";
const tick = String.fromCharCode(96);
const text = "word ".repeat(158) + "do " + tick + "foo(a, b)" + tick + " here";
const pieces = splitForSpeech(text, 800);
console.log(JSON.stringify({
  unbalanced: pieces.filter((p) => (p.split(tick).length - 1) % 2 === 1).length,
  spanWhole: pieces.some((p) => p.includes(tick + "foo(a, b)" + tick)),
  longest: Math.max(...pieces.map((p) => p.length)),
  words: pieces.join(" ").split(" ").length,
}));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"unbalanced": 0, "spanWhole": True, "longest": 792, "words": 162}


def test_asking_about_something_else_is_not_asking_again() -> None:
    from evals import metrics

    rows = metrics.load_jsonl_dataset(ROOT / "evals" / "datasets" / "practice_interviewer.jsonl")
    non_answers = [row for row in rows if "non_answer" in row["id"]]
    assert len(non_answers) == 3
    for example in non_answers:
        assert metrics.score_example("practice_interviewer", example["reference_response"], example)[0] == 1.0
        off_topic = json.loads(example["reference_response"])
        off_topic["interviewer_reply"] = (
            "I have not heard an answer yet, and that is fine. Let me ask something else: what is your favorite color?"
        )
        assert metrics.score_example("practice_interviewer", json.dumps(off_topic), example)[0] < 1.0
