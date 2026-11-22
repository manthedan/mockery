from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient


def test_export_machine_specs_includes_all_backend_machines(tmp_path: Path) -> None:
    from mockery.server import create_app
    from mockery.state_machines import export_machine_specs

    specs = export_machine_specs()
    machines = specs["machines"]
    expected = {
        "session_phase",
        "card_probe",
        "vad_turn",
        "asr_model",
        "utterance",
        "practice_phase",
        "addon_pack",
        "diarizer_enrollment",
        "resume_redaction",
        "tts_backend",
    }
    assert expected.issubset(machines.keys())
    assert "stateDiagram-v2" in machines["practice_phase"]["mermaid"]
    assert "stateDiagram-v2" in machines["addon_pack"]["mermaid"]

    app = create_app(project_root=tmp_path, preload_asr=False)
    with TestClient(app) as client:
        res = client.get("/api/machines")
        assert res.status_code == 200
        body = res.json()
        assert expected.issubset(body["machines"].keys())


def test_practice_phase_lifecycle_live_debriefing_rollback_and_invariants(tmp_path: Path) -> None:
    from mockery.practice import PracticeQuestionItem, PracticeSession, PracticeTurn
    from mockery.server import create_app
    from mockery.state_machines import validate_practice_invariants

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime

    async def fake_plan(**kwargs: Any):
        qs = [
            {
                "id": "pq_one",
                "bank_question_id": "beh_1",
                "source": "bank",
                "category": "behavioral",
                "title": "Cross-team conflict",
                "competency": "Collaboration",
                "prompt": "Tell me about a time you navigated a cross-team disagreement.",
                "carl_focus": "Context, Action, Result, Learning",
            },
            {
                "id": "pq_two",
                "bank_question_id": "beh_2",
                "source": "bank",
                "category": "behavioral",
                "title": "High-stakes incident",
                "competency": "Ownership",
                "prompt": "Tell me about a production incident you owned end-to-end.",
                "carl_focus": "Root cause and lasting fix",
            },
        ]
        return qs, "Welcome to your mock interview."

    fail_next_turn = {"flag": False}

    async def fake_reply(**kwargs: Any):
        if fail_next_turn["flag"]:
            fail_next_turn["flag"] = False
            raise RuntimeError("Simulated LLM outage on turn")
        return {
            "interviewer_reply": "Can you quantify the exact impact of your approach?",
            "should_advance": False,
            "carl_signals": {"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
            "coach_nudge": {"tone": "tip", "message": "Add a concrete metric for Result."},
        }

    async def fake_hint(**kwargs: Any):
        return "Frame your Result with a before/after latency or revenue metric."

    debrief_started = threading.Event()
    debrief_release = threading.Event()
    fail_debrief = {"flag": False}

    async def fake_debrief(**kwargs: Any):
        if fail_debrief["flag"]:
            fail_debrief["flag"] = False
            raise RuntimeError("Simulated debrief LLM timeout")
        debrief_started.set()
        await asyncio.to_thread(debrief_release.wait, 5.0)
        return {
            "overall_encouragement": "Strong ownership and clear narrative!",
            "top_strengths": ["Clear personal actions"],
            "question_reviews": [],
            "retry_drill": {
                "question_id": "pq_one",
                "focus": "Quantify Result & Learning",
                "prompt": "Re-deliver the result and what changed going forward in 45 seconds.",
            },
        }

    async def fake_retry(**kwargs: Any):
        return {
            "encouragement": "Much sharper!",
            "what_improved": "Added concrete 40% reduction metric.",
            "one_more_polish": "State the durable process change.",
            "carl_after": {"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
        }

    rt.llm.generate_practice_plan = fake_plan  # type: ignore[method-assign]
    rt.llm.generate_practice_interviewer_reply = fake_reply  # type: ignore[method-assign]
    rt.llm.generate_practice_hint = fake_hint  # type: ignore[method-assign]
    rt.llm.generate_practice_debrief = fake_debrief  # type: ignore[method-assign]
    rt.llm.evaluate_practice_retry = fake_retry  # type: ignore[method-assign]

    with TestClient(app) as client:
        # 1. Start session -> active
        r_start = client.post(
            "/api/practice/start",
            json={"interview_type": "behavioral", "question_count": 2},
        )
        assert r_start.status_code == 200
        sess = r_start.json()["session"]
        sid = sess["id"]
        sm = sess["state_machines"]
        assert sm["practice"]["phase"] == "active"
        assert sm["invariants_ok"] is True
        assert sm["jobs"]["debrief"] == "idle"
        assert set(sm["practice"]["allowed_events"]) == {
            "submit_turn",
            "request_hint",
            "advance_question",
            "update_workspace",
            "generate_debrief",
        }
        assert "submit_retry" in sm["practice"]["blocked_reasons"]

        # 2. Illegal retry in active -> structured 409 transition_rejected
        r_early_retry = client.post(f"/api/practice/{sid}/retry", json={"answer_text": "Too early"})
        assert r_early_retry.status_code == 409
        detail = r_early_retry.json()["detail"]
        assert detail["type"] == "transition_rejected"
        assert detail["machine"] == "practice_phase"
        assert detail["state"] == "active"
        assert detail["event"] == "submit_retry"
        assert "submit_turn" in detail["allowed_events"]
        assert detail["message"] == detail["remediation"]

        # 3. Failed LLM turn leaves session atomic & invariants intact (no orphaned candidate turn)
        turn_count_before = len(sess["turns"])
        fail_next_turn["flag"] = True
        try:
            r_fail = client.post(f"/api/practice/{sid}/turn", json={"text": "Answer during outage"})
            assert r_fail.status_code >= 500
        except RuntimeError:
            pass
        r_after_fail = client.get(f"/api/practice/sessions/{sid}")
        assert r_after_fail.status_code == 200
        s_after_fail = r_after_fail.json()["session"]
        assert len(s_after_fail["turns"]) == turn_count_before
        assert s_after_fail["state_machines"]["invariants_ok"] is True

        # 4. Expected question_id mismatch returns structured 409 transition_rejected
        r_mismatch = client.post(
            f"/api/practice/{sid}/turn",
            json={"text": "Stale question", "expected_question_id": "pq_two"},
        )
        assert r_mismatch.status_code == 409
        assert r_mismatch.json()["detail"]["type"] == "transition_rejected"
        assert r_mismatch.json()["detail"]["machine"] == "practice_phase"

        # 5. Failed debrief preserves active question status (rollback safety!)
        fail_debrief["flag"] = True
        try:
            r_deb_fail = client.post(f"/api/practice/{sid}/debrief")
            assert r_deb_fail.status_code >= 500
        except RuntimeError:
            pass
        s_after_deb_fail = client.get(f"/api/practice/sessions/{sid}").json()["session"]
        assert s_after_deb_fail["phase"] == "active"
        assert s_after_deb_fail["questions"][0]["status"] == "active"
        assert s_after_deb_fail["state_machines"]["invariants_ok"] is True
        assert s_after_deb_fail["state_machines"]["jobs"]["debrief"] == "idle"

        # 6. Normal turn + advance through both questions -> awaiting_debrief
        r_turn = client.post(
            f"/api/practice/{sid}/turn",
            json={"text": "I aligned both teams around shared SLOs.", "expected_question_id": "pq_one"},
        )
        assert r_turn.status_code == 200
        assert r_turn.json()["session"]["state_machines"]["invariants_ok"] is True

        r_next1 = client.post(f"/api/practice/{sid}/next", json={"expected_question_id": "pq_one"})
        assert r_next1.status_code == 200
        assert r_next1.json()["session"]["state_machines"]["practice"]["phase"] == "active"
        assert r_next1.json()["session"]["state_machines"]["invariants_ok"] is True

        r_next2 = client.post(f"/api/practice/{sid}/next", json={"expected_question_id": "pq_two"})
        assert r_next2.status_code == 200
        s_await = r_next2.json()["session"]
        assert s_await["phase"] == "awaiting_debrief"
        assert s_await["state_machines"]["practice"]["phase"] == "awaiting_debrief"
        assert s_await["state_machines"]["invariants_ok"] is True
        assert set(s_await["state_machines"]["practice"]["allowed_events"]) == {
            "update_workspace",
            "generate_debrief",
        }

        # 7. Live DEBRIEFING state observable mid-flight + fast 409 rejection of concurrent /turn
        debrief_result: dict[str, Any] = {}

        def _run_debrief() -> None:
            debrief_result["res"] = client.post(f"/api/practice/{sid}/debrief")

        t = threading.Thread(target=_run_debrief)
        t.start()
        assert debrief_started.wait(timeout=3.0)

        mid_get = client.get(f"/api/practice/sessions/{sid}").json()["session"]
        assert mid_get["phase"] == "debriefing"
        assert mid_get["state_machines"]["practice"]["phase"] == "debriefing"
        assert mid_get["state_machines"]["jobs"]["debrief"] == "running"
        assert mid_get["state_machines"]["practice"]["allowed_events"] == []

        r_during = client.post(f"/api/practice/{sid}/turn", json={"text": "Concurrent turn"})
        assert r_during.status_code == 409
        d_during = r_during.json()["detail"]
        assert d_during["type"] == "transition_rejected"
        assert d_during["state"] == "debriefing"
        assert d_during["allowed_events"] == []

        debrief_release.set()
        t.join(timeout=5.0)
        assert debrief_result["res"].status_code == 200
        s_done = debrief_result["res"].json()["session"]
        assert s_done["phase"] == "completed"
        assert s_done["state_machines"]["practice"]["phase"] == "completed"
        assert s_done["state_machines"]["jobs"]["debrief"] == "idle"
        assert s_done["state_machines"]["invariants_ok"] is True
        assert set(s_done["state_machines"]["practice"]["allowed_events"]) == {
            "generate_debrief",
            "submit_retry",
        }

        # 8. Retry succeeds in completed; workspace edit rejected with structured 409
        r_ws = client.post(f"/api/practice/{sid}/workspace", json={"workspace_context": {"code": "x=1"}})
        assert r_ws.status_code == 409
        assert r_ws.json()["detail"]["state"] == "completed"
        assert r_ws.json()["detail"]["event"] == "update_workspace"

        r_retry = client.post(f"/api/practice/{sid}/retry", json={"answer_text": "Cut latency by 40%."})
        assert r_retry.status_code == 200
        assert r_retry.json()["session"]["state_machines"]["invariants_ok"] is True

    # 9. Direct unit check of invariant detector on deliberately broken PracticeSession
    broken = PracticeSession(
        phase="completed",
        questions=[
            PracticeQuestionItem(id="q1", prompt="P1", status="active", carl_signals={"context": "bogus"}),
            PracticeQuestionItem(id="q1", prompt="P2", status="active"),
        ],
        current_question_index=5,
        turns=[PracticeTurn(id="t1", role="candidate", question_id="missing_q", text="hi")],
        debrief=None,
    )
    violations = validate_practice_invariants(broken)
    assert any("Duplicate question ID" in v for v in violations)
    assert any("invalid CARL status" in v for v in violations)
    assert any("unknown question_id" in v for v in violations)
    assert any("out of bounds" in v for v in violations)
    assert any("Multiple questions marked active" in v for v in violations)
    assert any("missing debrief" in v for v in violations)


def test_addon_pack_state_machine_uninstalling_and_error(tmp_path: Path) -> None:
    from mockery.addons import AddonManager
    from mockery.redactor import RampartRedactor

    redactor = RampartRedactor()
    mgr = AddonManager(tmp_path, redactor)

    installed = {"value": False}
    should_fail_install = {"value": True}
    uninstall_entered = asyncio.Event()
    uninstall_release = asyncio.Event()

    async def fake_install() -> None:
        if should_fail_install["value"]:
            raise RuntimeError("Network reset")
        installed["value"] = True

    async def fake_uninstall() -> bool:
        uninstall_entered.set()
        await uninstall_release.wait()
        installed["value"] = False
        return True

    mgr.register_pack_handler(
        "pyodide_python",
        is_installed=lambda: installed["value"],
        install=fake_install,
        uninstall=fake_uninstall,
        available=True,
    )

    async def _run() -> None:
        p0 = {p["id"]: p for p in mgr.status_dict()["packs"]}["pyodide_python"]
        assert p0["state"] == "not_installed"
        assert p0["allowed_events"] == ["install"]
        assert "uninstall" in p0["blocked_reasons"]

        # Failed install transitions to explicit error state with install retry allowed
        try:
            await mgr.install_pack("pyodide_python")
            assert False, "Expected install failure"
        except RuntimeError:
            pass
        p_err = {p["id"]: p for p in mgr.status_dict()["packs"]}["pyodide_python"]
        assert p_err["state"] == "error"
        assert p_err["error"] == "Network reset"
        assert p_err["allowed_events"] == ["install"]

        # Retry succeeds -> installed
        should_fail_install["value"] = False
        await mgr.install_pack("pyodide_python")
        p_ok = {p["id"]: p for p in mgr.status_dict()["packs"]}["pyodide_python"]
        assert p_ok["state"] == "installed"
        assert p_ok["allowed_events"] == ["uninstall"]
        assert "install" in p_ok["blocked_reasons"]

        # Uninstall enters explicit uninstalling state with all pack events blocked
        u_task = asyncio.create_task(mgr.uninstall_pack("pyodide_python"))
        await asyncio.wait_for(uninstall_entered.wait(), timeout=2.0)
        p_un = {p["id"]: p for p in mgr.status_dict()["packs"]}["pyodide_python"]
        assert p_un["state"] == "uninstalling"
        assert p_un["allowed_events"] == []
        assert "install" in p_un["blocked_reasons"]
        assert "uninstall" in p_un["blocked_reasons"]

        uninstall_release.set()
        await u_task
        p_done = {p["id"]: p for p in mgr.status_dict()["packs"]}["pyodide_python"]
        assert p_done["state"] == "not_installed"
        assert p_done["allowed_events"] == ["install"]

    asyncio.run(_run())


def test_diarizer_profile_and_tts_state_machines(tmp_path: Path) -> None:
    from mockery.diarizer import OnlineTwoSpeakerDiarizer, WeSpeakerEmbedder
    from mockery.profiles import CandidateProfileStore
    from mockery.redactor import RampartRedactor
    from mockery.state_machines import (
        DiarizerEnrollmentState,
        ResumeRedactionState,
        TTSBackendState,
        derive_tts_backend_state,
    )
    from mockery.tts import LocalTTSEngine

    # 1. DiarizerEnrollmentState progression
    embedder = WeSpeakerEmbedder(tmp_path / "wespeaker")
    diar = OnlineTwoSpeakerDiarizer(embedder)
    st0 = diar.status_dict()
    assert st0["enrollment_state"] == DiarizerEnrollmentState.UNLOADED.value
    assert "swap_roles" in st0["allowed_events"]
    assert "manual_correction" not in st0["allowed_events"]
    assert "manual_correction" in st0["blocked_reasons"]

    # Simulate loaded embedder across cold_start -> single_speaker -> two_speakers -> manual_calibrated
    embedder.status_dict = lambda: {"ready": True, "backend": "stub"}  # type: ignore[method-assign]
    st_cold = diar.status_dict()
    assert st_cold["enrollment_state"] == DiarizerEnrollmentState.COLD_START.value
    assert "swap_roles" in st_cold["allowed_events"]
    assert "manual_correction" not in st_cold["allowed_events"]

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    diar._centroids["interviewer"] = v1
    diar._utt_history["u1"] = (v1, 1.5, "interviewer")
    st_one = diar.status_dict()
    assert st_one["enrollment_state"] == DiarizerEnrollmentState.SINGLE_SPEAKER.value
    assert "swap_roles" in st_one["allowed_events"]
    assert "manual_correction" in st_one["allowed_events"]
    assert st_one["utterance_history_count"] == 1

    diar._centroids["candidate"] = v2
    st_two = diar.status_dict()
    assert st_two["enrollment_state"] == DiarizerEnrollmentState.TWO_SPEAKERS.value
    assert "swap_roles" in st_two["allowed_events"]

    diar.swap_roles(mark_manual=True)
    st_man = diar.status_dict()
    assert st_man["enrollment_state"] == DiarizerEnrollmentState.MANUAL_CALIBRATED.value
    assert st_man["manual_override"] is True

    # 2. CandidateProfileStore invariants + dangling active_jd_id self-healing
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "profile.json").write_text(
        json.dumps(
            {
                "resume": None,
                "job_listings": [{"id": "jd_keep", "title": "Staff Eng", "text": "Build distributed systems"}],
                "active_jd_id": "jd_deleted_dangling",
            }
        ),
        encoding="utf-8",
    )
    store = CandidateProfileStore(profiles_dir, RampartRedactor())
    pub = store.public_dict()
    assert pub["active_jd_id"] == "jd_keep"
    assert pub["state_machines"]["resume_redaction"]["state"] == ResumeRedactionState.NO_RESUME.value
    assert pub["state_machines"]["invariants_ok"] is True

    store.set_resume(filename="cv.txt", raw_text="Jane Doe jane@example.com built services.")
    pub2 = store.public_dict()
    assert pub2["state_machines"]["resume_redaction"]["state"] in (
        ResumeRedactionState.REDACTED_RULES_ONLY.value,
        ResumeRedactionState.REDACTED_RAMPART.value,
    )
    assert pub2["state_machines"]["invariants_ok"] is True

    # 3. TTSBackendState derivation + LocalTTSEngine.status_dict() + runtime fallback consistency
    assert (
        derive_tts_backend_state(
            kokoro_installed=True, kokoro_runtime=True, kokoro_error=None, macos_say_available=True
        )
        == TTSBackendState.KOKORO_READY
    )
    assert (
        derive_tts_backend_state(
            kokoro_installed=True, kokoro_runtime=True, kokoro_error="boom", macos_say_available=True
        )
        == TTSBackendState.KOKORO_FALLBACK_MACOS
    )
    assert (
        derive_tts_backend_state(
            kokoro_installed=False, kokoro_runtime=False, kokoro_error=None, macos_say_available=True
        )
        == TTSBackendState.MACOS_SAY
    )
    assert (
        derive_tts_backend_state(
            kokoro_installed=False, kokoro_runtime=False, kokoro_error=None, macos_say_available=False
        )
        == TTSBackendState.BROWSER_FALLBACK
    )
    tts = LocalTTSEngine(tmp_path / "models")
    tts_st = tts.status_dict()
    assert tts_st["backend_state"] in {s.value for s in TTSBackendState}

    # Simulate Kokoro installed + runtime present, then Kokoro.create() failing at synthesis time
    class _BrokenKokoro:
        def create(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("ONNX runtime crash")

    tts.is_kokoro_installed = lambda: True  # type: ignore[method-assign]
    tts._has_kokoro_runtime = lambda: True  # type: ignore[method-assign]
    tts._kokoro = _BrokenKokoro()
    tts._ensure_kokoro_loaded = lambda: True  # type: ignore[method-assign]

    wav, eng = asyncio.run(tts.synthesize_wav("Hello candidate"))
    assert wav.startswith(b"RIFF")
    assert eng == "macos_say"
    assert tts._kokoro is None
    assert tts._kokoro_error and "ONNX runtime crash" in tts._kokoro_error
    st_after = tts.status_dict()
    assert st_after["backend_state"] == TTSBackendState.KOKORO_FALLBACK_MACOS.value
    assert st_after["engine"] == "macos_say"
    assert "af_heart" not in st_after["voices"]


def test_phase1_codex_regressions_cancellation_http409_and_invariants(tmp_path: Path) -> None:
    from mockery.addons import AddonManager
    from mockery.practice import PracticeSession, PracticeTurn
    from mockery.redactor import RampartRedactor
    from mockery.server import create_app
    from mockery.state_machines import (
        AddonPackState,
        compute_allowed_addon_pack_events,
        derive_addon_pack_state,
        validate_practice_invariants,
    )

    # 1. 0-question session with dangling turn question_id must fail invariants
    zero_q = PracticeSession(
        phase="awaiting_debrief",
        questions=[],
        turns=[PracticeTurn(id="t1", role="candidate", question_id="orphan_q", text="hi")],
    )
    assert any("unknown question_id" in v for v in validate_practice_invariants(zero_q))

    # 2. AddonPackState precedence: coming_soon > error > installed
    assert (
        derive_addon_pack_state(installed=False, available=False, error="stale")
        == AddonPackState.COMING_SOON
    )
    assert (
        derive_addon_pack_state(installed=True, available=True, error="uninstall failed")
        == AddonPackState.ERROR
    )
    err_allowed, _ = compute_allowed_addon_pack_events(
        AddonPackState.ERROR, installed=True, available=True
    )
    assert set(err_allowed) == {"install", "uninstall"}
    err_unavail_allowed, _ = compute_allowed_addon_pack_events(
        AddonPackState.ERROR, installed=True, available=False
    )
    assert err_unavail_allowed == ["uninstall"]

    # 3. Cancelled uninstall waits for threaded IO before releasing lock
    mgr = AddonManager(tmp_path / "addon_cancel", RampartRedactor())
    installed = {"value": True}
    io_entered = threading.Event()
    io_release = threading.Event()

    def slow_uninstall_io() -> bool:
        io_entered.set()
        io_release.wait(timeout=5.0)
        installed["value"] = False
        return True

    async def fake_uninstall() -> bool:
        return await asyncio.to_thread(slow_uninstall_io)

    async def fake_install() -> None:
        installed["value"] = True

    mgr.register_pack_handler(
        "pyodide_python",
        is_installed=lambda: installed["value"],
        install=fake_install,
        uninstall=fake_uninstall,
        available=True,
    )

    async def _run_cancel_addon() -> None:
        u_task = asyncio.create_task(mgr.uninstall_pack("pyodide_python"))
        await asyncio.to_thread(io_entered.wait, 2.0)
        u_task.cancel()
        await asyncio.sleep(0.05)
        # Lock must still be held and state still uninstalling until IO finishes
        assert {p["id"]: p for p in mgr.status_dict()["packs"]}["pyodide_python"]["state"] == "uninstalling"
        i_task = asyncio.create_task(mgr.install_pack("pyodide_python"))
        await asyncio.sleep(0.05)
        assert not i_task.done()
        io_release.set()
        try:
            await u_task
        except asyncio.CancelledError:
            pass
        await i_task
        assert installed["value"] is True

    asyncio.run(_run_cancel_addon())

    # 4. HTTP 409 transition_rejected on conflicting addon install/uninstall & unavailable packs
    app = create_app(project_root=tmp_path / "srv", preload_asr=False)
    rt = app.state.runtime
    install_entered = threading.Event()
    install_release = threading.Event()
    pack_installed = {"value": False}

    async def slow_pack_install() -> None:
        install_entered.set()
        await asyncio.to_thread(install_release.wait, 5.0)
        pack_installed["value"] = True

    async def quick_pack_uninstall() -> bool:
        pack_installed["value"] = False
        return True

    rt.addons.register_pack_handler(
        "pyodide_python",
        is_installed=lambda: pack_installed["value"],
        install=slow_pack_install,
        uninstall=quick_pack_uninstall,
        available=True,
    )
    rt.addons.register_pack_handler(
        "kokoro_tts",
        is_installed=lambda: False,
        install=slow_pack_install,
        uninstall=quick_pack_uninstall,
        available=False,
    )

    with TestClient(app) as client:
        # Unavailable pack returns 409 coming_soon and does NOT mutate state to error
        r_unavail = client.post("/api/addons/kokoro_tts/install")
        assert r_unavail.status_code == 409
        assert r_unavail.json()["detail"]["type"] == "transition_rejected"
        assert r_unavail.json()["detail"]["state"] == "coming_soon"
        kok = {p["id"]: p for p in client.get("/api/addons").json()["packs"]}["kokoro_tts"]
        assert kok["state"] == "coming_soon"

        # Start install in background; DELETE during installing returns 409 transition_rejected
        res_holder: dict[str, Any] = {}

        def _do_install() -> None:
            res_holder["res"] = client.post("/api/addons/pyodide_python/install")

        th = threading.Thread(target=_do_install)
        th.start()
        assert install_entered.wait(timeout=3.0)

        r_del_busy = client.delete("/api/addons/pyodide_python")
        assert r_del_busy.status_code == 409
        assert r_del_busy.json()["detail"]["type"] == "transition_rejected"
        assert r_del_busy.json()["detail"]["state"] == "installing"
        assert r_del_busy.json()["detail"]["event"] == "uninstall"

        install_release.set()
        th.join(timeout=5.0)
        assert res_holder["res"].status_code == 200
        assert pack_installed["value"] is True

        # 5. Cancelled debrief during disk save keeps DEBRIEFING & lock until save finishes
        async def quick_plan(**kwargs: Any):
            return (
                [
                    {
                        "id": "pq_a",
                        "title": "Design cache",
                        "category": "system_design",
                        "prompt": "Design a distributed cache.",
                    }
                ],
                "Let's begin.",
            )

        async def quick_debrief(**kwargs: Any):
            return {
                "overall_encouragement": "Great work!",
                "top_strengths": ["Clear trade-offs"],
                "question_reviews": [],
                "retry_drill": {"question_id": "pq_a", "focus": "Consistency", "prompt": "Retry."},
            }

        rt.llm.generate_practice_plan = quick_plan  # type: ignore[method-assign]
        rt.llm.generate_practice_debrief = quick_debrief  # type: ignore[method-assign]

        r_st = client.post("/api/practice/start", json={"interview_type": "system_design", "question_count": 1})
        assert r_st.status_code == 200
        sid = r_st.json()["session"]["id"]

        orig_save = rt.practice_store.save
        save_entered = threading.Event()
        save_release = threading.Event()
        intercept_once = {"flag": True}

        def slow_save(sess: PracticeSession) -> None:
            if intercept_once["flag"] and sess.phase == "completed":
                intercept_once["flag"] = False
                save_entered.set()
                save_release.wait(timeout=5.0)
            orig_save(sess)

        rt.practice_store.save = slow_save  # type: ignore[method-assign]

        # Run debrief coroutine directly on a fresh event loop and cancel it mid-save
        async def _verify_debrief_save_lock() -> None:
            def _iter_routes(routes: Any):
                for r in routes:
                    orig = getattr(r, "original_router", None)
                    if orig is not None:
                        yield from _iter_routes(orig.routes)
                    else:
                        sub = getattr(r, "routes", None)
                        if sub is not None and not hasattr(r, "endpoint"):
                            yield from _iter_routes(sub)
                        else:
                            yield r

            endpoint = next(
                r.endpoint
                for r in _iter_routes(app.routes)
                if getattr(r, "path", None) == "/api/practice/{session_id}/debrief"
            )
            deb_task = asyncio.create_task(endpoint(sid))
            await asyncio.to_thread(save_entered.wait, 3.0)
            deb_task.cancel()
            await asyncio.sleep(0.05)
            # Still debriefing while save thread hasn't finished!
            assert rt.practice_debrief_jobs.get(sid) is not None
            save_release.set()
            try:
                await deb_task
            except asyncio.CancelledError:
                pass
            assert rt.practice_debrief_jobs.get(sid) is None
            persisted = rt.practice_store.get(sid)
            assert persisted is not None
            assert persisted.phase == "completed"

        asyncio.run(_verify_debrief_save_lock())

