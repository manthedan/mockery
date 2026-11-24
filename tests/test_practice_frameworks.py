"""Answer frameworks: a Practice session is graded and coached in the rubric it was started with."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mockery.practice import PracticeConfig, build_heuristic_nudge, estimate_carl_heuristic
from mockery.prompts import (
    practice_debrief_rubric_args,
    practice_reply_rubric_args,
    practice_tone_instruction,
    render_prompt,
)
from mockery.rubrics import (
    CARL,
    CODING,
    GENERAL,
    NO_STRUCTURE,
    SIGNAL_SLOTS,
    STAR,
    SYSTEM_DESIGN,
    framework_for,
    rubric_for,
)
from mockery.server import create_app

STAR_ANSWER = (
    "When I joined, our team's checkout failed during an outage every quarter. "
    "My job was to get the error rate down before the holiday deadline. "
    "I led the migration, I wrote the retry logic, and I convinced two other teams to adopt idempotency keys. "
    "We reduced duplicate charges by 96% in six weeks."
)


def _offline_client(tmp_path: Path) -> TestClient:
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.llm_config.provider = "none"
    app.state.runtime.llm_config.api_key = ""
    return TestClient(app)


def _start(client: TestClient, **body: object) -> dict:
    res = client.post("/api/practice/start", json=body)
    assert res.status_code == 200, res.text
    return res.json()["session"]


@pytest.mark.parametrize("rubric", [CARL, STAR, GENERAL, SYSTEM_DESIGN, CODING, NO_STRUCTURE])
def test_every_rubric_fills_the_four_slots_in_order(rubric) -> None:
    assert tuple(signal.slot for signal in rubric.signals) == SIGNAL_SLOTS
    assert len(set(rubric.signal_keys)) == 4
    assert set(rubric.drill_order) == set(SIGNAL_SLOTS)
    for slot in rubric.nudge_order:
        assert rubric.signal(slot).nudge


@pytest.mark.parametrize("rubric", [CARL, STAR, GENERAL, SYSTEM_DESIGN, CODING])
def test_the_grader_is_told_what_each_signal_means(rubric) -> None:
    args = practice_reply_rubric_args(rubric.grader, rubric.signal_keys)
    for signal in rubric.signals:
        assert f'"{signal.key}": "strong|partial|missing"' in args["signal_schema"]
        assert f"{signal.label} = " in args["signal_definitions"]


@pytest.mark.parametrize("rubric", [SYSTEM_DESIGN, CODING])
def test_system_design_and_coding_rubrics_avoid_carl_bleedthrough(rubric) -> None:
    assert "CARL" not in rubric.focus_label
    assert "CARL" not in rubric.debrief_name

    reply_prompt = render_prompt(
        "practice_interviewer_reply",
        interview_type_label=rubric.id.replace("_", " "),
        role_title="Senior Engineer",
        target_level="Senior",
        tone_instruction=practice_tone_instruction("supportive"),
        **practice_reply_rubric_args(rubric.grader, rubric.signal_keys),
    )
    debrief_prompt = render_prompt(
        "practice_debrief",
        **practice_debrief_rubric_args(rubric.grader, rubric.signal_keys),
    )
    assert "teaching the CARL framework" not in debrief_prompt
    assert "CARL format" not in debrief_prompt
    assert "Learning = " not in reply_prompt
    for signal in rubric.signals:
        assert f'"{signal.key}": "strong|partial|missing"' in reply_prompt
        assert f'"{signal.key}": "strong|partial|missing"' in debrief_prompt
        assert f"{signal.label} = " in reply_prompt

    graded = rubric.signals_from_grader(
        {
            rubric.signals[0].key: "strong",
            rubric.signals[1].key: "partial",
            rubric.signals[2].key: "strong",
            rubric.signals[3].key: "missing",
        }
    )
    assert graded == {
        "context": "strong",
        "action": "partial",
        "result": "strong",
        "learning": "missing",
    }


def test_the_format_decides_which_frameworks_apply() -> None:
    assert framework_for("behavioral", "star") == "star"
    assert framework_for("behavioral", "none") == "none"
    assert framework_for("behavioral", "carl") == "carl"
    assert framework_for("general", "star") == "general"
    assert framework_for("general", "none") == "general"
    assert rubric_for("system_design", "star") is SYSTEM_DESIGN
    assert rubric_for("coding", "none") is CODING


def test_sessions_saved_before_frameworks_existed_stay_carl() -> None:
    for interview_type in ("behavioral", "general"):
        legacy = PracticeConfig.model_validate({"interview_type": interview_type, "target_level": "L5"})
        assert legacy.rubric is CARL


def test_star_session_is_scored_and_coached_in_star(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="behavioral", answer_framework="star", question_source="bank_only")

    assert session["config"]["answer_framework"] == "star"
    assert [s["label"] for s in session["rubric"]["signals"]] == ["Situation", "Task", "Action", "Result"]
    assert session["rubric"]["meter_title"] == "Live STAR Meter"

    setup_only = "When I joined, our team's checkout failed during an outage every quarter, close to a deadline."
    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": setup_only}).json()
    assert "yours to do" in turn["session"]["live_nudge"]["message"]

    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": STAR_ANSWER}).json()
    scored = turn["session"]["turns"][-2]["carl_signals"]
    assert scored == {"context": "strong", "action": "strong", "result": "strong", "learning": "strong"}

    client.post(f"/api/practice/{session['id']}/debrief", json={})
    report = client.get(f"/api/practice/{session['id']}/markdown").text
    assert "## Question-by-question STAR coaching" in report
    assert "**STAR:** Situation `strong` · Task `strong` · Action `strong` · Result `strong`" in report
    assert "CARL" not in report and "Learning" not in report


def test_the_same_answer_scores_differently_under_each_rubric() -> None:
    carl = estimate_carl_heuristic(STAR_ANSWER, rubric=CARL)
    star = estimate_carl_heuristic(STAR_ANSWER, rubric=STAR)
    assert carl["learning"] == "missing"
    assert star["learning"] == "strong"
    no_task = STAR_ANSWER.replace("My job was to get the error rate down before the holiday deadline. ", "")
    assert estimate_carl_heuristic(no_task, rubric=STAR)["action"] == "missing"
    task = STAR.signal("action")
    assert build_heuristic_nudge(estimate_carl_heuristic(no_task, rubric=STAR), "behavioral", STAR)[
        "message"
    ] == f"{task.earned_praise} {task.nudge}"


def test_general_questions_use_their_own_signals(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="general", answer_framework="none")

    assert session["config"]["answer_framework"] == "general"
    assert [s["label"] for s in session["rubric"]["signals"]] == ["Direct", "Specific", "Fit", "Concise"]

    answer = (
        "I want this role because your team builds tools nurses actually use. "
        "For example, I spent three years on a surgical unit, and this position lets me bring that to your product."
    )
    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": answer}).json()
    assert turn["session"]["turns"][-2]["carl_signals"] == {
        "context": "strong",
        "action": "strong",
        "result": "strong",
        "learning": "strong",
    }
    vague = estimate_carl_heuristic("It seems like a good opportunity for me at this point in time.", rubric=GENERAL)
    assert vague["action"] == "missing" and vague["result"] == "missing"


def test_no_structure_stores_no_signals_and_names_no_framework(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    session = _start(client, interview_type="behavioral", answer_framework="none", question_source="bank_only")
    assert session["rubric"]["coached"] is False

    turn = client.post(f"/api/practice/{session['id']}/turn", json={"text": STAR_ANSWER}).json()
    candidate = turn["session"]["turns"][-2]
    assert candidate["carl_signals"] is None
    assert set(turn["session"]["questions"][0]["carl_signals"].values()) == {"missing"}
    nudge = (turn["session"]["live_nudge"] or {}).get("message", "")
    assert not any(word in nudge for word in ("CARL", "STAR", "Context", "Learning"))

    client.post(f"/api/practice/{session['id']}/debrief", json={})
    report = client.get(f"/api/practice/{session['id']}/markdown").text
    assert "CARL" not in report and "STAR" not in report
    assert "`strong`" not in report and "`missing`" not in report
    listed = client.get("/api/practice/sessions").json()["sessions"]
    assert next(row for row in listed if row["id"] == session["id"])["score"] is None


def test_star_grades_from_the_model_land_in_the_right_slots(tmp_path: Path) -> None:
    client = _offline_client(tmp_path)
    llm = client.app.state.runtime.llm
    llm.config.provider = "openai"
    llm.config.api_key = "test-key"
    seen: dict[str, str] = {}

    async def _fake_complete(*, system: str, user: str, **_: object) -> str:
        seen["system"], seen["user"] = system, user
        return json.dumps(
            {
                "interviewer_reply": "Thanks. What did that change for the team?",
                "should_advance": False,
                "carl_signals": {"situation": "strong", "task": "partial", "action": "strong", "result": "missing"},
                "coach_nudge": {"tone": "tip", "message": "Clear actions; now close with the Result."},
            }
        )

    llm._complete_text = _fake_complete  # type: ignore[method-assign]
    reply = asyncio.run(
        llm.generate_practice_interviewer_reply(
            role_title="Charge Nurse",
            company="",
            interview_type="behavioral",
            target_level="Senior",
            coach_style="supportive",
            current_question={"prompt": "Tell me about a time you fixed a recurring problem.", "carl_focus": ""},
            next_question=None,
            recent_turns=[],
            candidate_answer=STAR_ANSWER,
            follow_up_count=0,
            existing_carl={},
            rubric=STAR,
        )
    )

    assert reply["carl_signals"] == {
        "context": "strong",  # Situation
        "action": "partial",  # Task
        "result": "strong",  # Action
        "learning": "missing",  # Result
    }
    assert '"situation": "strong|partial|missing", "task": "strong|partial|missing"' in seen["system"]
    assert "Task = " in seen["system"] and "Learning = " not in seen["system"]
    assert "STAR Coaching Debrief" in seen["user"]


def test_carl_prompts_read_exactly_as_before_the_frameworks_were_added() -> None:
    reply = render_prompt(
        "practice_interviewer_reply",
        interview_type_label="behavioral",
        role_title="Engineer",
        target_level="Senior",
        tone_instruction=practice_tone_instruction("supportive"),
        **practice_reply_rubric_args(),
    )
    assert (
        '"carl_signals": {"context": "strong|partial|missing", "action": "strong|partial|missing", '
        '"result": "strong|partial|missing", "learning": "strong|partial|missing"}, "coach_nudge"'
    ) in reply
    assert "carry them forward. When the problem or context was clearly stated earlier" in reply
    assert "(e.g., quantify the Result, add a Learning, mention complexity or tests)" in reply

    debrief = render_prompt("practice_debrief", **practice_debrief_rubric_args())
    assert "interview coach teaching the CARL framework (Context, Action, Result, Learning). NEVER be harsh" in debrief
    no_framework = render_prompt(
        "practice_debrief", **practice_debrief_rubric_args(NO_STRUCTURE.grader, NO_STRUCTURE.signal_keys, coached=False)
    )
    assert "teaching the" not in no_framework and "never name or teach one" in no_framework
