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
    thin = build_heuristic_nudge(
        estimate_carl_heuristic(short_general, rubric=GENERAL), "general", GENERAL, short_general
    )
    assert thin == {"tone": "tip", "message": GENERAL.signal("result").nudge}

    silence = "Um, hmm."
    assert build_heuristic_nudge(
        estimate_carl_heuristic(silence, rubric=GENERAL), "general", GENERAL, silence
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


def _client_with_failing_model(tmp_path: Path, error: Exception) -> TestClient:
    client = _offline_client(tmp_path)
    llm = client.app.state.runtime.llm
    llm.config.provider = "openai"
    llm.config.api_key = "test-key"

    async def _fail(*_: object, **__: object) -> str:
        raise error

    llm._complete_text = _fail  # type: ignore[method-assign]
    llm._complete_messages = _fail  # type: ignore[method-assign]
    return client


def test_a_built_in_reply_says_the_connected_ai_failed_and_why(tmp_path: Path) -> None:
    client = _client_with_failing_model(tmp_path, RuntimeError("Error code: 401 - Invalid API key"))
    session = _start(client, interview_type="behavioral")

    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": REAL_ANSWER}).json()
    reason = turn["interviewer_turn"]["fallback_reason"]
    assert reason.startswith("Your AI did not respond.") and "API key" in reason
    assert turn["session"]["turns"][-1]["fallback_reason"] == reason  # saved with the session

    debrief = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]["debrief"]
    assert "API key" in debrief["fallback_reason"]

    retry = client.post(f"/api/practice/{session['id']}/retry", json={"answer_text": REAL_ANSWER}).json()
    assert "API key" in retry["session"]["retry_history"][-1]["feedback"]["fallback_reason"]


def test_a_garbled_model_reply_is_reported_as_unusable(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    llm = client.app.state.runtime.llm
    llm.config.provider = "openai"
    llm.config.api_key = "test-key"

    async def _garble(*_: object, **__: object) -> str:
        return "Sure! Here is my feedback, but not as JSON."

    llm._complete_text = _garble  # type: ignore[method-assign]
    llm._complete_messages = _garble  # type: ignore[method-assign]
    session = _start(client, interview_type="behavioral")
    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": REAL_ANSWER}).json()
    assert "not in a form Mockery could use" in turn["interviewer_turn"]["fallback_reason"]


def test_no_reason_is_given_when_no_ai_is_connected(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="behavioral")
    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": REAL_ANSWER}).json()
    assert turn["interviewer_turn"]["fallback_reason"] == ""
    debrief = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]["debrief"]
    assert "fallback_reason" not in debrief


def test_the_eval_scores_invented_praise_below_an_honest_reply() -> None:
    import json

    from evals import metrics

    datasets = Path(__file__).resolve().parents[1] / "evals" / "datasets"
    debriefs = {row["id"]: row for row in metrics.load_jsonl_dataset(datasets / "practice_debrief.jsonl")}
    example = debriefs["pd_train_07_non_answer"]
    honest, _ = metrics.score_example("practice_debrief", example["reference_response"], example)
    flattering = json.loads(example["reference_response"])
    flattering["top_strengths"] = ["You showed up and engaged with the question.", "You sounded confident."]
    flattering["question_reviews"][0]["strength"] = "You engaged with the prompt."
    inflated, info = metrics.score_example("practice_debrief", json.dumps(flattering), example)
    assert honest == 1.0 and inflated < 0.8
    assert any("Invented strengths" in failure for failure in info["failures"])

    replies = {row["id"]: row for row in metrics.load_jsonl_dataset(datasets / "practice_interviewer.jsonl")}
    example = replies["pi_13_behavioral_filler_non_answer"]
    honest, _ = metrics.score_example("practice_interviewer", example["reference_response"], example)
    thanking = json.loads(example["reference_response"])
    thanking["interviewer_reply"] = "Thanks for engaging with that. Could you name one specific project?"
    thanked, info = metrics.score_example("practice_interviewer", json.dumps(thanking), example)
    assert honest == 1.0 and thanked < honest
    assert any("said nothing" in failure for failure in info["failures"])


def test_a_model_that_lists_no_strengths_is_taken_at_its_word(tmp_path: Path) -> None:
    import json

    client = _offline_client(tmp_path)
    llm = client.app.state.runtime.llm
    llm.config.provider = "openai"
    llm.config.api_key = "test-key"
    session = _start(client, interview_type="behavioral")
    question = session["questions"][0]

    async def _reply(*_: object, **__: object) -> str:
        return json.dumps(
            {
                "interviewer_reply": "I have not heard an answer yet. Could you pick one real example?",
                "should_advance": False,
                "carl_signals": {"context": "missing", "action": "missing", "result": "missing", "learning": "missing"},
                "coach_nudge": {"tone": "tip", "message": "Pick one real example and start with the situation."},
            }
        )

    async def _debrief(*_: object, **__: object) -> str:
        return json.dumps(
            {
                "overall_encouragement": "There was not enough here to coach yet. Pick one real example next time.",
                "top_strengths": [],
                "carl_primer": "CARL gives an answer its shape.",
                "question_reviews": [
                    {
                        "question_id": question["id"],
                        "title": question["title"],
                        "prompt": question["prompt"],
                        "carl": {"context": "missing", "action": "missing", "result": "missing", "learning": "missing"},
                        "strength": "",
                        "coaching_tip": "Choose one real project before you start speaking.",
                        "carl_reframe": "Nobody owned it, so I did X, and Y changed.",
                    }
                ],
                "retry_drill": {"question_id": question["id"], "focus": "One real story", "prompt": "Tell one real story."},
            }
        )

    llm._complete_messages = _reply  # type: ignore[method-assign]
    llm._complete_text = _debrief  # type: ignore[method-assign]
    client.post(f"/api/practice/{session['id']}/turn", json={"text": "uh oh"})
    debrief = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]["debrief"]
    assert debrief["top_strengths"] == []
    assert debrief["question_reviews"][0]["strength"] == ""
    assert "fallback_reason" not in debrief
