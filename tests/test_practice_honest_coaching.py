"""With no model connected, the coach and interviewer praise only what an answer earned."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mockery.practice import build_heuristic_nudge, estimate_carl_heuristic
from mockery.rubrics import CARL, GENERAL, NO_STRUCTURE, STAR
from mockery.server import create_app

NON_ANSWERS = ["Yeah, great question, uhhh uhhh", "uh oh", "farts"]
FILLER = "Yeah, um that's the kind of thing I did, you know, just like all the time."
PRAISE_WORDS = ("Good", "Great", "Strong", "Solid", "Nice", "Thanks for walking", "clear picture", "thoughtful")
REAL_ANSWER = (
    "When I joined, our team's checkout failed during an outage every quarter. "
    "I led the migration, I wrote the retry logic, and I convinced two other teams to adopt idempotency keys. "
    "We reduced duplicate charges by 96% in six weeks."
)


def _offline_client(tmp_path: Path) -> TestClient:
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.llm_config.provider = "none"
    app.state.runtime.llm_config.api_key = ""
    return TestClient(app)


def _start(client: TestClient, **body: object) -> dict:
    res = client.post("/api/practice/start", json={"question_source": "bank_only", **body})
    assert res.status_code == 200, res.text
    return res.json()["session"]


@pytest.mark.parametrize("framework", ["carl", "star", "none"])
def test_saying_nothing_earns_no_praise_anywhere(tmp_path: Path, framework: str) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="behavioral", answer_framework=framework)
    question = session["questions"][0]["prompt"]

    said_by_app: list[str] = []
    for non_answer in NON_ANSWERS:
        turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": non_answer}).json()
        said_by_app.append(turn["interviewer_turn"]["text"])
        said_by_app.append((turn["session"]["live_nudge"] or {}).get("message", ""))

    assert question in said_by_app[0]  # the question is asked again rather than followed up
    assert "leave that one there" in said_by_app[4]  # after three tries it moves on without thanks
    for line in said_by_app:
        assert not any(word in line for word in PRAISE_WORDS), line

    debrief = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]["debrief"]
    assert debrief["top_strengths"] == []
    assert "too short to coach" in debrief["overall_encouragement"]
    for review in debrief["question_reviews"]:
        assert review["strength"] == ""
        assert not any(word in review["coaching_tip"] for word in PRAISE_WORDS)


def test_filler_is_not_mistaken_for_setting_the_scene() -> None:
    for rubric in (CARL, STAR):
        signals = estimate_carl_heuristic(FILLER, rubric=rubric)
        assert set(signals.values()) == {"missing"}
        nudge = build_heuristic_nudge(signals, "behavioral", rubric, FILLER)
        assert nudge["tone"] == "tip"
        assert not any(word in nudge["message"] for word in PRAISE_WORDS)
    unstructured = build_heuristic_nudge(estimate_carl_heuristic(FILLER, rubric=NO_STRUCTURE), "behavioral", NO_STRUCTURE, FILLER)
    assert not any(word in unstructured["message"] for word in PRAISE_WORDS)


def test_praise_follows_only_from_the_signal_that_earned_it() -> None:
    setting_only = "When I joined, our team had a legacy project with an outage every quarter and a hard deadline."
    earned = build_heuristic_nudge(estimate_carl_heuristic(setting_only), "behavioral", CARL, setting_only)
    assert earned["message"].startswith("Good setup on the context!")

    result_only = "Honestly the main thing is that we reduced costs by 20% across the whole organisation that year."
    unearned = build_heuristic_nudge(estimate_carl_heuristic(result_only), "behavioral", CARL, result_only)
    assert unearned["message"] == CARL.signal("action").nudge

    short_general = "Because money."
    assert build_heuristic_nudge(
        estimate_carl_heuristic(short_general, rubric=GENERAL), "general", GENERAL, short_general
    )["message"] == GENERAL.nothing_yet


def test_a_real_answer_is_still_thanked_and_credited(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="behavioral")

    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": REAL_ANSWER}).json()
    assert turn["interviewer_turn"]["text"].startswith("Thanks for walking through that.")
    assert turn["session"]["live_nudge"]["message"].startswith("Solid C-A-R foundation!")

    debrief = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]["debrief"]
    assert debrief["top_strengths"] and "Context, Action and Result came through clearly" in debrief["top_strengths"][0]
    assert debrief["overall_encouragement"].startswith("Good rep.")


def test_short_remarks_are_normal_in_a_coding_round(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="coding")
    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": "Can I use a heap?"}).json()
    assert "I'll ask again" not in turn["interviewer_turn"]["text"]
