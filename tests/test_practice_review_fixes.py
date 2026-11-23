"""Regressions found in review of the answer-framework and honest-coaching work."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mockery.practice import estimate_carl_heuristic, said_too_little
from mockery.prompts import get_prompt_spec, validate_prompt_template
from mockery.rubrics import CODING, GENERAL, SYSTEM_DESIGN
from mockery.server import create_app

ROOT = Path(__file__).resolve().parents[1]
FILLER = "Yeah, um, that's the kind of thing I did, you know, just like all the time."


def _client(tmp_path: Path, *, configured: bool = False) -> TestClient:
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.llm_config.provider = "openai" if configured else "none"
    app.state.runtime.llm_config.api_key = "test-key" if configured else ""
    return TestClient(app)


def _start(client: TestClient, **body: object) -> dict:
    res = client.post("/api/practice/start", json={"question_source": "bank_only", **body})
    assert res.status_code == 200, res.text
    return res.json()["session"]


def _turn(client: TestClient, session: dict, text: str) -> dict:
    res = client.post(f"/api/practice/{session['id']}/turn", json={"text": text})
    assert res.status_code == 200, res.text
    return res.json()


def test_studio_rounds_are_scored_on_their_own_signals_offline(tmp_path: Path) -> None:
    reflection = "I learned about the format of the interview."
    assert set(estimate_carl_heuristic(reflection, rubric=CODING).values()) == {"missing"}
    assert set(estimate_carl_heuristic(reflection, rubric=SYSTEM_DESIGN).values()) == {"missing"}

    spoken = "I'd use a hash map and a linked list, and I handle the empty cache and duplicate keys in O(1)."
    scored = estimate_carl_heuristic(spoken, rubric=CODING)
    assert scored["context"] == "strong" and scored["result"] == "strong" and scored["learning"] == "strong"

    client = _client(tmp_path)
    session = _start(client, interview_type="coding")
    _turn(client, session, reflection)
    debrief = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]["debrief"]
    assert "Edge cases" not in json.dumps(debrief["top_strengths"])
    assert "Edge cases came through" not in debrief["question_reviews"][0]["strength"]


def test_a_short_factual_answer_is_an_answer(tmp_path: Path) -> None:
    assert not said_too_little("$130k to $150k base")
    assert not said_too_little("Yes, in 2021 at Acme")
    assert said_too_little("Um, hmm") and said_too_little("Yeah, great question, uhhh uhhh")
    assert set(estimate_carl_heuristic("Um, hmm", rubric=GENERAL).values()) == {"missing"}

    client = _client(tmp_path)
    session = _start(client, interview_type="general")
    reply = _turn(client, session, "$130k to $150k base")["interviewer_turn"]["text"]
    assert "I'll ask again" not in reply


def test_wordy_filler_is_not_thanked_offline(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session = _start(client, interview_type="behavioral", question_count=1)
    said = [_turn(client, session, FILLER)["interviewer_turn"]["text"] for _ in range(3)]
    for line in said:
        assert "Thanks for walking" not in line and "thoughtful ground" not in line and "clear picture" not in line
    assert said[0].startswith("Let's make that concrete.")
    assert said[-1].startswith("Let's stop there for today.")


def test_no_structure_sessions_keep_no_grades_anywhere(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session = _start(client, interview_type="behavioral", answer_framework="none")
    _turn(client, session, "When I joined, our team had an outage every quarter and I led the migration that fixed it.")
    done = client.post(f"/api/practice/{session['id']}/debrief", json={}).json()["session"]
    assert all("carl" not in review for review in done["debrief"]["question_reviews"])

    retried = client.post(
        f"/api/practice/{session['id']}/retry", json={"answer_text": "We reduced outages by 80% and I learned to stage rollouts."}
    ).json()["session"]
    assert "carl_after" not in retried["retry_history"][-1]["feedback"]
    assert retried["turns"][-1]["carl_signals"] is None


def test_the_round_does_not_ask_two_questions_when_it_moves_on(tmp_path: Path) -> None:
    client = _client(tmp_path, configured=True)
    llm = client.app.state.runtime.llm
    reask = "I have not heard an answer yet. Could you tell me about one real project?"

    async def _reask(*_: object, **__: object) -> str:
        return json.dumps(
            {
                "interviewer_reply": reask,
                "should_advance": False,
                "carl_signals": {"context": "missing", "action": "missing", "result": "missing", "learning": "missing"},
                "coach_nudge": {"tone": "tip", "message": "Pick one real example and start with the situation."},
            }
        )

    llm._complete_messages = _reask  # type: ignore[method-assign]
    session = _start(client, interview_type="behavioral")
    next_prompt = session["questions"][1]["prompt"]

    replies = [_turn(client, session, "uh oh")["interviewer_turn"]["text"] for _ in range(3)]
    assert replies[0] == reask and replies[1] == reask
    assert reask not in replies[2]
    assert replies[2].startswith("Let's leave that one there and move on.") and next_prompt in replies[2]


def test_hint_and_plan_failures_are_reported(tmp_path: Path) -> None:
    client = _client(tmp_path, configured=True)
    llm = client.app.state.runtime.llm

    async def _fail(*_: object, **__: object) -> str:
        raise RuntimeError("Error code: 401 - Invalid API key")

    llm._complete_text = _fail  # type: ignore[method-assign]
    llm._complete_messages = _fail  # type: ignore[method-assign]
    client.post(
        "/api/profile/jd",
        json={"title": "Charge Nurse", "company": "Example Health", "text": "Lead a 24-bed surgical unit across shifts."},
    )
    res = client.post("/api/practice/start", json={"interview_type": "behavioral", "question_source": "mixed"})
    assert res.status_code == 200, res.text
    assert "API key" in res.json()["opening_turn"]["fallback_reason"]

    hint = client.post(f"/api/practice/{res.json()['session']['id']}/hint", json={}).json()
    assert "API key" in hint["coach_turn"]["fallback_reason"]


def test_a_pinned_question_from_a_mixed_case_pack_is_honoured(tmp_path: Path) -> None:
    banks = tmp_path / "banks"
    banks.mkdir()
    (banks / "mine.json").write_text(
        json.dumps(
            {
                "pack_id": "mine",
                "title": "My pack",
                "category": "Behavioral",
                "version": "1.0",
                "questions": [
                    {"id": "mine_01", "title": "My Own Question", "competency": "Ownership", "prompt": "Tell me about your proudest launch.", "levels": ["Senior"]}
                ],
            }
        )
    )
    client = _client(tmp_path)
    session = _start(client, interview_type="behavioral", selected_question_id="mine_01")
    assert session["questions"][0]["bank_question_id"] == "mine_01"
    assert session["config"]["selected_question_id"] == "mine_01"


def test_a_placeholder_against_an_extra_brace_is_rejected() -> None:
    template = get_prompt_spec("practice_retry_eval").default_text
    assert validate_prompt_template("practice_retry_eval", template) == []
    assert validate_prompt_template("practice_retry_eval", template.replace("{signal_schema} }", "{signal_schema}}}"))


def test_code_blocks_are_never_split_into_speech() -> None:
    script = """
import { splitForSpeech } from "./src/mockery/static/voice_clip.js";
const filler = "Read the problem statement carefully before you begin. ".repeat(14);
const fence = String.fromCharCode(96).repeat(3);
const text = filler + "\\n" + fence + "python\\n" + "def solve(items):\\n    return sorted(items)\\n".repeat(20) + fence + "\\nNow call `self.cache.get(key, now_ms)` once. Then say the complexity.";
const pieces = splitForSpeech(text, 800);
console.log(JSON.stringify({
  code: pieces.some((p) => p.includes("def solve") || p.includes("sorted(items)")),
  unbalanced: pieces.filter((p) => (p.split(String.fromCharCode(96)).length - 1) % 2 === 1).length,
  spanWhole: pieces.some((p) => p.includes("`self.cache.get(key, now_ms)`")),
  longest: Math.max(...pieces.map((p) => p.length)),
}));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    res: dict[str, Any] = json.loads(proc.stdout)
    assert res["code"] is False
    assert res["unbalanced"] == 0 and res["spanWhole"] is True
    assert res["longest"] <= 800


def test_the_eval_no_longer_scores_flattery_as_perfect() -> None:
    from evals import metrics

    datasets = ROOT / "evals" / "datasets"
    replies = [r for r in metrics.load_jsonl_dataset(datasets / "practice_interviewer.jsonl") if "non_answer" in r["id"]]
    assert len(replies) == 3 and all(r["interview_type"] == "behavioral" for r in replies)
    for example in replies:
        flattering = json.loads(example["reference_response"])
        flattering["interviewer_reply"] = "You sounded very confident and composed. We have some time left in the interview."
        assert metrics.score_example("practice_interviewer", json.dumps(flattering), example)[0] < 1.0

    debriefs = {r["id"]: r for r in metrics.load_jsonl_dataset(datasets / "practice_debrief.jsonl")}
    example = debriefs["pd_train_07_non_answer"]
    flattering = json.loads(example["reference_response"])
    flattering["overall_encouragement"] = "You sounded very confident and composed throughout. That is a great start."
    assert metrics.score_example("practice_debrief", json.dumps(flattering), example)[0] < 1.0
    omitted = json.loads(example["reference_response"])
    del omitted["top_strengths"]
    assert metrics.score_example("practice_debrief", json.dumps(omitted), example)[0] < 1.0
