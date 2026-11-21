from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mock_helper.grouper import QAGrouper
from mock_helper.llm import _fast_extra_body
from mock_helper.models import InterviewSession, SessionConfig, Utterance
from mock_helper.server import create_app
from mock_helper.sources import (
    CHANNEL_CANDIDATE,
    CHANNEL_INTERVIEWER,
    CrossChannelBleedFilter,
    SessionAudioRouter,
)
from mock_helper.vad import SileroVADRuntime


def _synthesize_pcm16(text: str, *, voice: str | None = None, trailing_silence_s: float = 0.9) -> bytes:
    """Use macOS native `say` + `ffmpeg` to generate genuine 16kHz spoken int16 PCM."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff_path = Path(tmp) / "speech.aiff"
        cmd = ["say"]
        if voice:
            cmd.extend(["-v", voice])
        cmd.extend(["-o", str(aiff_path), text])
        subprocess.run(cmd, check=True)
        raw_pcm = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-i",
                str(aiff_path),
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
    if trailing_silence_s > 0:
        silence = np.zeros(int(16000 * trailing_silence_s), dtype=np.int16).tobytes()
        return raw_pcm + silence
    return raw_pcm


def test_dual_channel_vad_and_stable_utt_revisions() -> None:
    vad_rt = SileroVADRuntime()
    router = SessionAudioRouter(vad_rt, mode="remote_meet")

    long_speech = _synthesize_pcm16(
        "Tell me about a complex distributed storage system you designed and operated at scale."
    )

    events = []
    for i in range(0, len(long_speech), 3200):
        frame = bytes([CHANNEL_INTERVIEWER]) + long_speech[i : i + 3200]
        events.extend(router.ingest_binary_frame(frame))

    finals = [e for e in events if e.kind == "final"]
    assert len(finals) >= 1
    assert all(e.speaker == "interviewer" for e in events)

    # Verify monotonic `rev` per `utterance_id` across partial -> final lifecycle
    by_id: dict[str, list[int]] = {}
    for ev in events:
        by_id.setdefault(ev.utterance_id, []).append(ev.rev)
    for revs in by_id.values():
        assert revs == sorted(revs)
        assert len(set(revs)) == len(revs)


def test_upsert_revisions_phase2_speaker_flip_and_regroup() -> None:
    session = InterviewSession()
    grouper = QAGrouper()

    # 1. Warmup greeting stays out of Q1
    d0 = grouper.ingest_utterance(
        session, speaker="interviewer", text="Hey, can you hear me okay?", start_ms=0, end_ms=1500
    )
    assert d0.qa_id is None
    assert len(session.warmup_utterance_ids) == 1

    # 2. Q1 opens
    d1 = grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Tell me about a time you resolved a severe production latency spike.",
        start_ms=2000,
        end_ms=6500,
        utt_id="utt_q1",
    )
    assert d1.opened_new_qa is True

    # 3. Live in-progress candidate answer (`final=False`, `rev=1`) is visible inside Q1 immediately!
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text="I noticed p99 latency jump during a flash sale...",
        start_ms=7000,
        end_ms=10000,
        utt_id="utt_ans1",
        rev=1,
        final=False,
    )
    q1 = session.qa_groups[0]
    assert len(session.utterances_for_card(q1, include_drafts=True)) == 2
    assert session.utterances_for_card(q1)[1].final is False

    # 4. Finalization (`rev=2, final=True`) updates `utt_ans1` in place without duplicating!
    d_fin = grouper.ingest_utterance(
        session,
        speaker="candidate",
        text=(
            "I noticed p99 checkout latency jump from 80ms to 1.4s during flash sales because hot inventory rows "
            "caused lock contention, so I moved reservations to Redis Lua scripts."
        ),
        start_ms=7000,
        end_ms=19000,
        utt_id="utt_ans1",
        rev=2,
        final=True,
    )
    assert d_fin.should_speculate_probes is True
    assert len(session.utterances) == 3  # warmup + q1 + ans1
    assert session.utterances_for_card(q1)[1].final is True

    # 5. Stale revision (`rev=1` arriving out of order) is ignored
    session.upsert_utterance(
        Utterance(id="utt_ans1", rev=1, final=False, speaker="candidate", text="STALE", start_ms=7000, end_ms=8000)
    )
    assert "Redis Lua" in session.utterances_for_card(q1)[1].text

    # 6. Phase 2 delayed diarization speaker-label correction (`rev=3` flips speaker cleanly in place)
    grouper.ingest_utterance(
        session,
        speaker="interviewer",  # initially misattributed by online diarizer
        text="And we verified zero lost orders across two million checkouts.",
        start_ms=20000,
        end_ms=24000,
        utt_id="utt_diar_flip",
        rev=1,
        final=True,
    )
    grouper.ingest_utterance(
        session,
        speaker="candidate",  # corrected at rev=2!
        text="And we verified zero lost orders across two million checkouts.",
        start_ms=20000,
        end_ms=24000,
        utt_id="utt_diar_flip",
        rev=2,
        final=True,
    )
    assert session.utterance_map()["utt_diar_flip"].speaker == "candidate"

    # 7. Full-context non-destructive regrouping over `utterance_ids`
    grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Tell me about a disagreement with a senior engineer.",
        start_ms=30000,
        end_ms=34000,
        utt_id="utt_q2",
    )
    ok = grouper.apply_regroup_boundaries(
        session,
        [
            {"start_utterance_id": "utt_q1", "title": "Flash Sale Lock Contention"},
            {"start_utterance_id": "utt_q2", "title": "Engineering Disagreement"},
        ],
    )
    assert ok is True
    assert len(session.qa_groups) == 2
    assert session.qa_groups[0].title == "Flash Sale Lock Contention"
    assert session.qa_groups[1].utterance_ids == ["utt_q2"]


def test_system_design_phase_cards_and_echo_bleed_filter() -> None:
    session = InterviewSession(config=SessionConfig(interview_type="system_design"))
    grouper = QAGrouper()

    grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Let's design a global rate limiter for our API gateway.",
        start_ms=1000,
        end_ms=5000,
    )
    assert session.qa_groups[0].card_kind == "phase"
    assert session.qa_groups[0].badge_prefix() == "P"

    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text=(
            "First let's clarify functional and non-functional requirements: we need per-tenant sliding window limits "
            "under five milliseconds p99 across three regions with eventual quota sync."
        ),
        start_ms=6000,
        end_ms=16000,
    )
    d_p2 = grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Sounds good, let's walk through your high-level design and data model.",
        start_ms=17000,
        end_ms=21000,
    )
    assert d_p2.opened_new_qa is True
    assert len(session.qa_groups) == 2

    # Echo bleed suppression test (speakerphone bleed from Meet tab into laptop mic)
    bleed = CrossChannelBleedFilter()
    bleed.record_candidate_text(10000, 15000, "We sharded the token buckets using consistent hashing.")
    assert bleed.is_echo_bleed(10800, 15400, "sharded the token buckets using consistent hashing") is True
    assert bleed.is_echo_bleed(16000, 19000, "What happens when a Redis node fails over?") is False

    # DeepSeek V4.1 Flash non-thinking parameter check
    assert _fast_extra_body("deepseek-flash") == {"thinking": {"type": "disabled"}}
    assert _fast_extra_body("gemini-2.5-flash") is None


def test_websocket_blind_calibration_jsonl_and_dual_exports(tmp_path: Path) -> None:
    app = create_app(project_root=tmp_path, preload_asr=False)
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws:
        _ = ws.receive_json()  # initial session_state
        _ = ws.receive_json()  # initial asr_status

        ws.send_json(
            {
                "type": "configure_session",
                "config": {"save_raw_audio": True, "mode": "remote_meet"},
            }
        )
        _ = ws.receive_json()

        # Stream a binary audio frame to verify opt-in WAV archiving (gated only by `save_raw_audio`)
        ws.send_bytes(bytes([CHANNEL_INTERVIEWER]) + np.zeros(1600, dtype=np.int16).tobytes())

        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_ask1",
                "speaker": "interviewer",
                "text": "Walk me through how your team migrated the billing monolith.",
                "start_ms": 1000,
                "end_ms": 5000,
            }
        )
        msg1 = ws.receive_json()
        qa_id = msg1["session"]["qa_groups_resolved"][0]["id"]

        # Inject in-progress candidate answer (`final=False`) and trigger Dive Deeper mid-answer!
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_live_ans",
                "rev": 1,
                "final": False,
                "speaker": "candidate",
                "text": "We strangled the monolith over two quarters and our team reduced deploy risk",
                "start_ms": 6000,
                "end_ms": 13000,
            }
        )
        _ = ws.receive_json()

        ws.send_json({"type": "dive_deeper", "qa_id": qa_id})
        assert ws.receive_json()["type"] == "dive_deeper_loading"
        probe_res = ws.receive_json()
        assert probe_res["type"] == "dive_deeper_result"
        assert len(probe_res["result"]["probes"]) == 3
        _ = ws.receive_json()

        # Submit Blind Self-Rating + Run Calibration
        ws.send_json(
            {
                "type": "generate_calibration",
                "rating": {
                    "hire_signal": "lean_hire",
                    "perceived_level": "L5 Senior",
                    "gut_rationale": "Clear migration sequencing",
                    "biggest_concern": "Heavy use of 'we' without clear individual ownership",
                },
            }
        )
        assert ws.receive_json()["type"] == "calibration_loading"
        cal_ready = ws.receive_json()
        assert cal_ready["type"] == "calibration_ready"
        assert "lean_hire" in cal_ready["markdown"]
        final_state = ws.receive_json()
        session_id = final_state["session"]["id"]

        ws.send_json({"type": "flush_and_end"})
        _ = ws.receive_json()

    # Verify append-only JSONL durability + opt-in WAV file + both Interviewer & Candidate exports
    sessions_dir = tmp_path / "sessions"
    assert (sessions_dir / f"{session_id}.events.jsonl").exists()
    assert (sessions_dir / f"{session_id}_interviewer.wav").exists()

    int_md = client.get(f"/api/sessions/{session_id}/markdown?audience=interviewer").text
    cand_md = client.get(f"/api/sessions/{session_id}/markdown?audience=candidate").text
    assert "Interviewer Blind Pre-Calibration" in int_md
    assert "Interviewer Blind Pre-Calibration" not in cand_md
    assert "Transcript & Notes (Candidate Copy)" in cand_md


def test_codex_review_regressions(tmp_path: Path, monkeypatch) -> None:
    import wave

    import pytest
    from starlette.websockets import WebSocketDisconnect

    from mock_helper.llm import LLMRuntimeConfig
    from mock_helper.models import DiveDeeperResponse, ProbeSuggestion
    from mock_helper.storage import SessionAudioArchive

    # 1. Provider-scoped API key fallback + persisted PARAKEET_MODEL on startup
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    (tmp_path / ".env.local").write_text(
        "LLM_PROVIDER=openai\nPARAKEET_MODEL=mlx-community/parakeet-tdt-0.6b-v2\n",
        encoding="utf-8",
    )
    cfg = LLMRuntimeConfig(tmp_path)
    assert cfg.api_key == "sk-openai-secret"
    assert cfg.parakeet_model == "mlx-community/parakeet-tdt-0.6b-v2"

    app = create_app(project_root=tmp_path, preload_asr=False)
    assert app.state.runtime.asr.model_id == "mlx-community/parakeet-tdt-0.6b-v2"
    client = TestClient(app)

    # 2. Reject untrusted cross-site WebSocket & HTTP POST Origins
    bad_post = client.post(
        "/api/settings/llm",
        headers={"Origin": "https://attacker.example"},
        json={
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "fast_model": "gpt-4.1-mini",
            "calibration_model": "gpt-4.1",
        },
    )
    assert bad_post.status_code == 403

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/session", headers={"Origin": "https://attacker.example"}):
            pass

    # 3. Resumable WAV archive does NOT truncate prior audio on close -> reopen
    arch1 = SessionAudioArchive(tmp_path, "sess_wav")
    pcm_chunk = np.ones(1600, dtype=np.int16).tobytes()
    arch1.append_frame(bytes([CHANNEL_INTERVIEWER]) + pcm_chunk)
    arch1.close()

    arch2 = SessionAudioArchive(tmp_path, "sess_wav")
    arch2.append_frame(bytes([CHANNEL_INTERVIEWER]) + pcm_chunk)
    arch2.close()

    with wave.open(str(tmp_path / "sess_wav_interviewer.wav"), "rb") as wf:
        assert wf.getnframes() == 3200
        assert wf.getframerate() == 16000

    # 4. Draft -> Final interviewer pivot splits card + Regroup preserves notes from merged cards
    session = InterviewSession()
    grouper = QAGrouper()
    grouper.ingest_utterance(
        session, speaker="interviewer", text="Tell me about a system outage you led.", start_ms=0, end_ms=3000, utt_id="u1"
    )
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text="I debugged a BGP route leak across two edge PoPs that dropped fifteen percent of ingress traffic.",
        start_ms=4000,
        end_ms=12000,
        utt_id="u2",
    )
    # Interviewer draft starts before pivot phrase is spoken
    d_draft = grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Awesome thanks so",
        start_ms=13000,
        end_ms=14000,
        utt_id="u3",
        rev=1,
        final=False,
    )
    assert d_draft.opened_new_qa is False
    assert len(session.qa_groups) == 1

    # Final revision contains explicit topic pivot -> splits into Q2!
    d_final = grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Awesome thanks, moving on to my next question: tell me about mentoring a junior engineer.",
        start_ms=13000,
        end_ms=18000,
        utt_id="u3",
        rev=2,
        final=True,
    )
    assert d_final.opened_new_qa is True
    assert len(session.qa_groups) == 2

    session.qa_groups[0].notes = "Note from Q1"
    session.qa_groups[1].notes = "Note from Q2"
    session.qa_groups[1].dive_deeper_history.append(
        DiveDeeperResponse(
            qa_id=session.qa_groups[1].id,
            quick_read="Read",
            probes=[ProbeSuggestion(angle="Scope", question="How?", why="Why")],
        )
    )
    # Regroup merges Q1 & Q2 into one card -> BOTH notes & probe history survive!
    grouper.apply_regroup_boundaries(session, [{"start_utterance_id": "u1", "title": "Combined"}])
    assert len(session.qa_groups) == 1
    assert "Note from Q1" in session.qa_groups[0].notes
    assert "Note from Q2" in session.qa_groups[0].notes
    assert len(session.qa_groups[0].dive_deeper_history) == 1

    # 5. Echo bleed detected on final revision removes earlier mic draft + 409 guard while capturing
    with client.websocket_connect("/ws/session") as ws:
        init_msg = ws.receive_json()
        active_id = init_msg["session"]["id"]
        _ = ws.receive_json()  # asr_status
        app.state.runtime.store.save(app.state.runtime.active_session)

        # Mic draft arrives slightly before candidate tab transcript
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_echo",
                "rev": 1,
                "final": False,
                "speaker": "interviewer",
                "text": "we partitioned the kafka topic",
                "start_ms": 5000,
                "end_ms": 7000,
            }
        )
        s1 = ws.receive_json()
        assert any(u["id"] == "utt_echo" for u in s1["session"]["utterances"])

        # Candidate tab transcript arrives
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_cand_real",
                "rev": 1,
                "final": True,
                "speaker": "candidate",
                "text": "We partitioned the Kafka topic by tenant ID to isolate noisy neighbors.",
                "start_ms": 4800,
                "end_ms": 9500,
            }
        )
        _ = ws.receive_json()

        # Mic final revision matches candidate tab audio -> invalidates and removes `utt_echo`!
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_echo",
                "rev": 2,
                "final": True,
                "speaker": "interviewer",
                "text": "partitioned the Kafka topic by tenant ID",
                "start_ms": 5000,
                "end_ms": 8500,
            }
        )
        s3 = ws.receive_json()
        assert not any(u["id"] == "utt_echo" for u in s3["session"]["utterances"])
        assert s3["session"]["echo_bleed_dropped"] == 1

        # Simulate active binary capture and verify historical switch is blocked with 409
        ws.send_bytes(bytes([CHANNEL_INTERVIEWER]) + np.zeros(1600, dtype=np.int16).tobytes())
        conflict = client.get(f"/api/sessions/{active_id}")
        assert conflict.status_code == 409


def test_explicit_state_machines_ux_dx_ax(tmp_path: Path) -> None:
    import json

    from mock_helper.models import DiveDeeperResponse, ProbeSuggestion
    from mock_helper.state_machines import (
        CardProbeState,
        VADTurnState,
        derive_card_probe_state,
        validate_session_invariants,
    )
    from mock_helper.vad import ChannelVADSegmenter

    app = create_app(project_root=tmp_path, preload_asr=False)
    client = TestClient(app)

    # 1. AX/DX Introspection via `/api/machines` & `--dump-machines` CLI
    specs = client.get("/api/machines").json()
    assert "session_phase" in specs["machines"]
    assert "stateDiagram-v2" in specs["machines"]["session_phase"]["mermaid"]

    cli_out = subprocess.run(
        ["uv", "run", "mock-helper", "--dump-machines"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "card_probe" in json.loads(cli_out)["machines"]

    # 2. VADTurnState explicit finite state machine
    seg = ChannelVADSegmenter(SileroVADRuntime())
    assert seg.state == VADTurnState.SILENCE
    spoken = _synthesize_pcm16("Testing explicit voice activity state transitions.")
    seg.push_pcm_int16(spoken)
    assert seg.state == VADTurnState.SILENCE

    # 3. Full SessionPhase lifecycle + transition_rejected AX contract + CardProbeState accuracy
    with client.websocket_connect("/ws/session") as ws:
        s0 = ws.receive_json()
        _ = ws.receive_json()  # asr_status
        # A fresh session is immediately `ready`: live capture needs no consent gate or prior configuration
        assert s0["session"]["state_machines"]["session"]["phase"] == "ready"
        assert s0["session"]["state_machines"]["invariants_ok"] is True
        assert "start_capture" in s0["session"]["state_machines"]["session"]["allowed_events"]
        assert "start_capture" not in s0["session"]["state_machines"]["session"]["blocked_reasons"]

        # `start_capture` -> transitions `ready` -> `capturing` straight away
        ws.send_json({"type": "start_capture"})
        s_cap = ws.receive_json()
        assert s_cap["type"] == "session_state"
        assert s_cap["session"]["state_machines"]["session"]["phase"] == "capturing"
        assert "load_historical_session" not in s_cap["session"]["state_machines"]["session"]["allowed_events"]
        assert "start_capture" not in s_cap["session"]["state_machines"]["session"]["allowed_events"]

        # Starting capture while already capturing is still an invalid transition -> structured `transition_rejected`
        ws.send_json({"type": "start_capture"})
        rej = ws.receive_json()
        assert rej["type"] == "transition_rejected"
        assert rej["machine"] == "session_phase"
        assert rej["state"] == "capturing"
        assert rej["event"] == "start_capture"
        assert "already active" in rej["remediation"]

        # Add Q&A turns
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_sm_1",
                "speaker": "interviewer",
                "text": "Tell me about a time you simplified a stateful system.",
                "start_ms": 1000,
                "end_ms": 4000,
            }
        )
        s_q1 = ws.receive_json()
        qa_id = s_q1["session"]["qa_groups_resolved"][0]["id"]
        assert s_q1["session"]["qa_groups_resolved"][0]["badge"] == "Q1"

        # Switching interview_type dynamically updates existing card_kind (`Q1` -> `P1`)
        ws.send_json({"type": "configure_session", "config": {"interview_type": "system_design"}})
        s_phase = ws.receive_json()
        assert s_phase["session"]["qa_groups_resolved"][0]["badge"] == "P1"

        # `flush_and_end` -> transitions `capturing` -> `awaiting_self_rating`
        ws.send_json({"type": "flush_and_end"})
        s_paused = ws.receive_json()
        assert s_paused["session"]["state_machines"]["session"]["phase"] == "awaiting_self_rating"
        assert s_paused["session"]["ended_epoch_ms"] is not None

        # Resuming capture clears stale `ended_epoch_ms` and re-enters `capturing`!
        ws.send_json({"type": "start_capture"})
        s_resumed = ws.receive_json()
        assert s_resumed["session"]["state_machines"]["session"]["phase"] == "capturing"
        assert s_resumed["session"]["ended_epoch_ms"] is None

        ws.send_json({"type": "flush_and_end"})
        _ = ws.receive_json()

        # Submit blind self-rating -> transitions `awaiting_self_rating` -> `rated`
        ws.send_json(
            {
                "type": "submit_self_rating",
                "rating": {"hire_signal": "strong_hire", "perceived_level": "L6"},
            }
        )
        s_rated = ws.receive_json()
        assert s_rated["session"]["state_machines"]["session"]["phase"] == "rated"
        assert s_rated["session"]["state_machines"]["invariants_ok"] is True

    # 4. Verify CardProbeState distinguishes fresh `ready_cached` vs `stale_cached`
    sess = app.state.runtime.active_session
    qa = sess.qa_groups[0]
    qa.speculative_cache = DiveDeeperResponse(
        qa_id=qa.id,
        quick_read="Fresh",
        probes=[ProbeSuggestion(angle="Scope", question="Why?", why="Check")],
        based_on_utt_count=1,
    )
    assert (
        derive_card_probe_state(sess, qa, warming_ids=set(), loading_ids=set())
        == CardProbeState.READY_CACHED
    )
    # Add a new candidate utterance -> cache is now honestly reported as `STALE_CACHED`, not instant!
    app.state.runtime.grouper.ingest_utterance(
        sess,
        speaker="candidate",
        text="We replaced six scattered boolean flags with a single explicit state enum.",
        start_ms=5000,
        end_ms=9000,
    )
    assert (
        derive_card_probe_state(sess, qa, warming_ids=set(), loading_ids=set())
        == CardProbeState.STALE_CACHED
    )
    assert validate_session_invariants(sess) == []

    # 5. Verify invariant checker catches orphaned/dangling corruption immediately
    sess.qa_groups[0].utterance_ids.append("utt_nonexistent")
    violations = validate_session_invariants(sess)
    assert any("Dangling utterance reference" in v for v in violations)

    # 6. Consent no longer gates the session lifecycle: `ready` is the entry state and only an
    #    already-active capture or a running calibration can block `start_capture`.
    from mock_helper.state_machines import (
        AsyncJobState,
        SessionPhase,
        compute_allowed_session_events,
        derive_session_phase,
    )

    assert "setup" not in specs["machines"]["session_phase"]["states"]
    assert "[*] --> ready" in specs["machines"]["session_phase"]["mermaid"]
    assert "consent" not in specs["machines"]["session_phase"]["mermaid"]

    # Older saved session JSON with `consent_confirmed: false` (or no such key) still deserializes
    # and is treated exactly like any other fresh session.
    legacy = InterviewSession.model_validate({"config": {"consent_confirmed": False, "save_raw_audio": True}})
    assert SessionConfig().consent_confirmed is True
    assert InterviewSession.model_validate({"config": {}}).config.consent_confirmed is True
    legacy_phase = derive_session_phase(legacy, is_capturing=False, calibration_job=AsyncJobState.IDLE)
    assert legacy_phase == SessionPhase.READY
    allowed_idle, blocked_idle = compute_allowed_session_events(
        legacy, phase=legacy_phase, regroup_job=AsyncJobState.IDLE
    )
    assert "start_capture" in allowed_idle and "start_capture" not in blocked_idle

    for busy_phase, needle in (
        (SessionPhase.CAPTURING, "already active"),
        (SessionPhase.CALIBRATING, "calibration to finish"),
    ):
        allowed_busy, blocked_busy = compute_allowed_session_events(
            legacy, phase=busy_phase, regroup_job=AsyncJobState.IDLE
        )
        assert "start_capture" not in allowed_busy
        assert needle in blocked_busy["start_capture"]

    # While calibration is running the server rejects an explicit `start_capture` command
    runtime = app.state.runtime
    runtime.calibration_job = AsyncJobState.RUNNING
    try:
        with client.websocket_connect("/ws/session") as ws:
            _ = ws.receive_json()  # session_state
            _ = ws.receive_json()  # asr_status
            ws.send_json({"type": "start_capture"})
            rej_cal = ws.receive_json()
            assert rej_cal["type"] == "transition_rejected"
            assert rej_cal["state"] == "calibrating"
            assert rej_cal["event"] == "start_capture"
            assert runtime.is_capturing is False
    finally:
        runtime.calibration_job = AsyncJobState.IDLE


def test_audit_correctness_regressions(tmp_path: Path, monkeypatch) -> None:
    import json
    import stat

    from mock_helper.grouper import make_concise_title
    from mock_helper.llm import LLMRuntimeConfig, _extract_json_object
    from mock_helper.models import DiveDeeperResponse, ProbeSuggestion
    from mock_helper.sources import CHANNEL_ROOM_MIC, SessionAudioRouter
    from mock_helper.state_machines import CardProbeState, derive_card_probe_state
    from mock_helper.storage import SessionStore

    # 0. Multi-sentence interviewer transition picks the actual question sentence for card titles
    t_title = make_concise_title(
        "Cool. That's awesome to hear. Great. I want to move on to a different question. "
        "So I want to ask you about describing a complex technical problem you solved end-to-end."
    )
    assert "describing a complex technical problem" in t_title.lower() or "ask you about describing" in t_title.lower()
    assert not t_title.lower().startswith("cool")

    # 1. Warmup small-talk draft -> final does NOT promote to Q1; chronological order & merge/split cache invalidation
    session = InterviewSession()
    grouper = QAGrouper()
    d1 = grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Can you hear me",
        start_ms=0,
        end_ms=1000,
        utt_id="u_warm",
        rev=1,
        final=False,
    )
    assert d1.qa_id is None
    assert session.warmup_utterance_ids == ["u_warm"]

    d2 = grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Can you hear me okay?",
        start_ms=0,
        end_ms=1800,
        utt_id="u_warm",
        rev=2,
        final=True,
    )
    assert d2.qa_id is None
    assert len(session.qa_groups) == 0
    assert session.warmup_utterance_ids == ["u_warm"]

    # Real Q1 + out-of-order late arrival stays chronologically sorted
    grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Tell me about a production incident you debugged.",
        start_ms=5000,
        end_ms=8000,
        utt_id="u_q1",
    )
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text="We had a memory leak in our Go worker pool.",
        start_ms=12000,
        end_ms=16000,
        utt_id="u_c2",
    )
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text="First I checked heap profiles.",
        start_ms=9000,
        end_ms=11500,
        utt_id="u_c1",
    )
    qa1 = session.qa_groups[0]
    assert qa1.utterance_ids == ["u_q1", "u_c1", "u_c2"]

    # Revision-aware CardProbeState detects in-place draft->final updates
    qa1.speculative_cache = DiveDeeperResponse(
        qa_id=qa1.id,
        quick_read="Initial",
        probes=[ProbeSuggestion(angle="A", question="Q?", why="W")],
        based_on_utt_count=len(qa1.utterance_ids),
        based_on_rev_sum=session.card_revision_sum(qa1),
    )
    assert (
        derive_card_probe_state(session, qa1, warming_ids=set(), loading_ids=set())
        == CardProbeState.READY_CACHED
    )
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text="We had a goroutine leak in our Go worker pool caused by unclosed channels.",
        start_ms=12000,
        end_ms=17000,
        utt_id="u_c2",
        rev=2,
        final=True,
    )
    assert (
        derive_card_probe_state(session, qa1, warming_ids=set(), loading_ids=set())
        == CardProbeState.STALE_CACHED
    )

    # Split invalidates cache; merge invalidates cache & re-keys dive_deeper_history
    qa2 = grouper.split_utterance_to_new_card(session, qa1.id, "u_c2")
    assert qa2 is not None
    assert qa1.speculative_cache is None
    qa2.dive_deeper_history.append(
        DiveDeeperResponse(
            qa_id=qa2.id,
            quick_read="From Q2",
            probes=[ProbeSuggestion(angle="B", question="Q2?", why="W2")],
        )
    )
    qa1.speculative_cache = DiveDeeperResponse(
        qa_id=qa1.id,
        quick_read="Temp",
        probes=[ProbeSuggestion(angle="A", question="Q?", why="W")],
        based_on_utt_count=len(qa1.utterance_ids),
        based_on_rev_sum=session.card_revision_sum(qa1),
    )
    merged = grouper.merge_with_previous(session, qa2.id)
    assert merged is not None
    assert merged.speculative_cache is None
    assert len(merged.dive_deeper_history) == 1
    assert merged.dive_deeper_history[0].qa_id == merged.id

    # 2. _extract_json_object strips <think>...</think> blocks containing braces
    parsed = _extract_json_object(
        '<think>Let me consider {"draft": 1} before deciding.</think>\n{"ok": true, "items": [1, 2]}'
    )
    assert parsed == {"ok": True, "items": [1, 2]}

    # 3. LLMRuntimeConfig preserves unrelated .env.local keys, sets 0o600, and doesn't leak old provider's key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-12345678")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-87654321")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("CUSTOM_USER_VAR=keep_me\n", encoding="utf-8")
    cfg = LLMRuntimeConfig(tmp_path)
    cfg.update_and_save(
        provider="openai",
        base_url="https://api.openai.com/v1",
        fast_model="gpt-4.1-mini",
        calibration_model="gpt-4.1",
        api_key="sk-custom-openai-key\nINJECTED=bad",
    )
    assert cfg.api_key == "sk-custom-openai-keyINJECTED=bad"
    assert "CUSTOM_USER_VAR=keep_me" in env_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

    # Switching provider without passing api_key must NOT carry over the OpenAI key
    cfg.update_and_save(
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        fast_model="deepseek-chat",
        calibration_model="deepseek-reasoner",
        api_key=None,
    )
    assert cfg.api_key == "sk-deepseek-87654321"

    # 4. SessionStore recovers from .events.jsonl if .json is corrupted
    store = SessionStore(tmp_path / "store_recovery")
    rec_sess = InterviewSession()
    rec_utt = Utterance(
        id="utt_rec_1",
        rev=1,
        final=True,
        speaker="interviewer",
        text="Tell me about a tough architecture tradeoff.",
        start_ms=1000,
        end_ms=4000,
    )
    store.append_utterance_event(rec_sess.id, rec_utt)
    (store.sessions_dir / f"{rec_sess.id}.json").write_text("{corrupt json", encoding="utf-8")
    recovered = store.load(rec_sess.id)
    assert recovered is not None
    assert recovered.id == rec_sess.id
    assert any(u.id == "utt_rec_1" for u in recovered.utterances)

    # 5. SessionAudioRouter initial_offset_ms & set_manual_speaker flush under previous speaker
    router = SessionAudioRouter(SileroVADRuntime(), mode="in_person", initial_offset_ms=60_000)
    router.set_manual_speaker("interviewer")
    spoken = _synthesize_pcm16("Quick room mic question before tab toggle.")
    # Feed only the voiced portion (trim trailing silence so speech stays active)
    half_even = (int(len(spoken) * 0.45) // 2) * 2
    voiced_prefix = spoken[:half_even]
    flushed_before = router.ingest_binary_frame(bytes([CHANNEL_ROOM_MIC]) + voiced_prefix)
    flushed_on_switch = router.set_manual_speaker("candidate")
    all_emitted = flushed_before + flushed_on_switch
    assert len(all_emitted) >= 1
    assert all(seg.speaker == "interviewer" for seg in all_emitted)
    assert all(seg.start_ms >= 60_000 for seg in all_emitted)

    # 6. Server-level WS & HTTP regressions: stale rev, empty dive_deeper, new_session readiness, GET vs POST /load, ASR exception
    app = create_app(project_root=tmp_path / "srv", preload_asr=False)
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws:
        init_msg = ws.receive_json()
        sess1_id = init_msg["session"]["id"]
        _ = ws.receive_json()  # asr_status

        # Stale revision does NOT duplicate JSONL events or overwrite newer text
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_rev_test",
                "rev": 2,
                "final": True,
                "speaker": "interviewer",
                "text": "Tell me about a tough production outage you debugged?",
                "start_ms": 1000,
                "end_ms": 4000,
            }
        )
        _ = ws.receive_json()
        jsonl_path = tmp_path / "srv" / "sessions" / f"{sess1_id}.events.jsonl"
        lines_before = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines_before) == 1

        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_rev_test",
                "rev": 1,
                "final": False,
                "speaker": "interviewer",
                "text": "Stale older draft",
                "start_ms": 1000,
                "end_ms": 2000,
            }
        )
        # Send another command right after to synchronize & prove stale rev was ignored cleanly
        ws.send_json({"type": "force_new_card"})
        s_new_card = ws.receive_json()
        lines_after = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines_after) == 1
        assert s_new_card["session"]["utterances"][0]["text"] == "Tell me about a tough production outage you debugged?"

        # dive_deeper on the newly forced empty Q2 card returns structured transition_rejected from card_probe
        empty_qa_id = s_new_card["session"]["qa_groups_resolved"][-1]["id"]
        ws.send_json({"type": "dive_deeper", "qa_id": empty_qa_id})
        rej = ws.receive_json()
        assert rej["type"] == "transition_rejected"
        assert rej["machine"] == "card_probe"
        assert rej["event"] == "dive_deeper"

        # new_session yields a fresh session that is immediately ready to capture. A legacy client that
        # still sends `consent_confirmed: false` is tolerated and never blocks `start_capture`.
        ws.send_json({"type": "new_session", "config": {"role_title": "Staff SWE", "consent_confirmed": False}})
        s2 = ws.receive_json()
        sess2_id = s2["session"]["id"]
        assert sess2_id != sess1_id
        assert s2["session"]["config"]["role_title"] == "Staff SWE"
        assert s2["session"]["state_machines"]["session"]["phase"] == "ready"
        assert "start_capture" in s2["session"]["state_machines"]["session"]["allowed_events"]
        assert "start_capture" not in s2["session"]["state_machines"]["session"]["blocked_reasons"]

    # GET /api/sessions/{sess1_id} is read-only (does NOT switch active_session away from sess2)
    get_res = client.get(f"/api/sessions/{sess1_id}")
    assert get_res.status_code == 200
    assert get_res.json()["session"]["id"] == sess1_id
    assert app.state.runtime.active_session.id == sess2_id

    # POST /api/sessions/{sess1_id}/load rejects untrusted Origin and switches active_session when trusted
    bad_load = client.post(
        f"/api/sessions/{sess1_id}/load",
        headers={"Origin": "https://evil.example"},
    )
    assert bad_load.status_code == 403

    good_load = client.post(f"/api/sessions/{sess1_id}/load")
    assert good_load.status_code == 200
    assert app.state.runtime.active_session.id == sess1_id

    # 7. final_segment_worker survives an ASR exception without deadlocking flush_and_end
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated MLX transcription failure")

    monkeypatch.setattr(app.state.runtime.asr, "transcribe_pcm", _boom)
    with client.websocket_connect("/ws/session") as ws:
        _ = ws.receive_json()  # session_state
        _ = ws.receive_json()  # asr_status
        ws.send_json({"type": "start_capture"})
        _ = ws.receive_json()
        # Push real speech PCM so Silero VAD enqueues a final segment that hits _boom
        spoken_pcm = _synthesize_pcm16("Testing worker survival on ASR exception.")
        ws.send_bytes(bytes([CHANNEL_INTERVIEWER]) + spoken_pcm)
        ws.send_json({"type": "flush_and_end"})
        # Should receive an error event and still cleanly receive session_ended without hanging
        seen_ended = False
        seen_error = False
        for _ in range(6):
            msg = ws.receive_json()
            if msg.get("type") == "error":
                seen_error = True
            if msg.get("type") == "session_state" and msg.get("session_ended"):
                seen_ended = True
                break
        assert seen_error is True
        assert seen_ended is True


def test_scratch_transfer_cli_providers_and_turn_triage(tmp_path: Path, monkeypatch) -> None:
    from mock_helper.llm import LLMRuntimeConfig, _parse_agy_model_spec
    from mock_helper.models import TurnSegment, TurnTriagePlan
    from mock_helper.state_machines import validate_session_invariants

    # 1. Local agy / pi CLI config & effort parsing
    assert _parse_agy_model_spec("gemini-4-hydrogen:low") == ("gemini-4-hydrogen", "low")
    assert _parse_agy_model_spec("gemini-4-hydrogen") == ("gemini-4-hydrogen", None)
    assert _parse_agy_model_spec("agy/gemini-4-hydrogen:high") == ("gemini-4-hydrogen", "high")

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}" if name in ("agy", "pi") else None)
    cfg = LLMRuntimeConfig(tmp_path / "cli_cfg")
    cfg.update_and_save(
        provider="agy",
        base_url="cli://agy",
        fast_model="gemini-4-hydrogen:low",
        calibration_model="gemini-4-hydrogen:high",
    )
    assert cfg.cli_backend == "agy"
    assert cfg.is_configured is True
    assert cfg.public_dict()["masked_api_key"] == "local agy CLI"

    cfg.update_and_save(
        provider="pi",
        base_url="cli://pi",
        fast_model="openai-codex/gpt-6-luna:low",
        calibration_model="openai-codex/gpt-6.2-sol:high",
    )
    assert cfg.cli_backend == "pi"
    assert cfg.is_configured is True
    assert cfg.public_dict()["masked_api_key"] == "local pi CLI"

    # 2. Confidence-gated Turn Triage applier (high-conf mixed split, STT overlap guard, low-conf review queue)
    session = InterviewSession()
    grouper = QAGrouper()
    grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Tell me about a time you pushed back on a client.",
        start_ms=1000,
        end_ms=4000,
        utt_id="u1",
    )
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text="At my previous role as a for deployed engineer I scoped down a custom request.",
        start_ms=5000,
        end_ms=12000,
        utt_id="u2",
    )

    # 2a. Light STT cleanup (overlap >= 0.65) is accepted
    r_clean = grouper.apply_turn_triage(
        session,
        TurnTriagePlan(
            utterance_id="u2",
            segments=[
                TurnSegment(
                    speaker="candidate",
                    text="At my previous role as a forward deployed engineer I scoped down a custom request.",
                )
            ],
            card_action="keep_in_current",
            confidence="high",
        ),
    )
    assert r_clean.mutated is True
    assert "forward deployed engineer" in session.utterance_map()["u2"].text
    assert session.utterance_map()["u2"].rev == 2

    # 2b. Hallucinated paraphrase (overlap < 0.65) is rejected; original text stays intact
    r_halluc = grouper.apply_turn_triage(
        session,
        TurnTriagePlan(
            utterance_id="u2",
            segments=[TurnSegment(speaker="candidate", text="Completely unrelated fabricated summary text.")],
            card_action="keep_in_current",
            confidence="high",
        ),
    )
    assert r_halluc.mutated is False
    assert "forward deployed engineer" in session.utterance_map()["u2"].text

    # 2c. High-confidence mixed-speaker handoff splits into 2 utterances & opens Q2 at interviewer pivot
    grouper.ingest_utterance(
        session,
        speaker="candidate",
        text=(
            "So that saved us two weeks of rework. "
            "Awesome, moving on to a complex technical problem you solved end to end?"
        ),
        start_ms=13000,
        end_ms=21000,
        utt_id="u3",
    )
    r_split = grouper.apply_turn_triage(
        session,
        TurnTriagePlan(
            utterance_id="u3",
            segments=[
                TurnSegment(speaker="candidate", text="So that saved us two weeks of rework."),
                TurnSegment(
                    speaker="interviewer",
                    text="Awesome, moving on to a complex technical problem you solved end to end?",
                ),
            ],
            card_action="open_new_card",
            suggested_title="Complex Technical Problem Solved End-to-End",
            confidence="high",
            rationale="Candidate concludes answer and interviewer pivots to Q2.",
        ),
    )
    assert r_split.mutated is True
    assert len(r_split.split_utterance_ids) == 2
    assert r_split.opened_new_qa_id is not None
    assert len(session.qa_groups) == 2
    assert session.qa_groups[1].title == "Complex Technical Problem Solved End-to-End"
    assert session.utterance_map()["u3"].speaker == "candidate"
    child_id = r_split.split_utterance_ids[1]
    assert session.utterance_map()[child_id].speaker == "interviewer"
    assert child_id == session.qa_groups[1].utterance_ids[0]
    assert validate_session_invariants(session) == []

    # 2d. Medium-confidence speaker flip queues PendingReviewItem without mutating; accepting applies it
    grouper.ingest_utterance(
        session,
        speaker="interviewer",
        text="Sure, let me walk through the distributed cache migration I led.",
        start_ms=22000,
        end_ms=27000,
        utt_id="u4",
    )
    r_med = grouper.apply_turn_triage(
        session,
        TurnTriagePlan(
            utterance_id="u4",
            segments=[
                TurnSegment(
                    speaker="candidate",
                    text="Sure, let me walk through the distributed cache migration I led.",
                )
            ],
            card_action="keep_in_current",
            confidence="medium",
            rationale="First-person project walk-through sounds like candidate.",
        ),
    )
    assert r_med.mutated is False
    assert r_med.queued_review_id is not None
    assert session.utterance_map()["u4"].speaker == "interviewer"
    assert len(session.pending_reviews) == 1

    r_accept = grouper.resolve_pending_review(session, r_med.queued_review_id, accept=True)
    assert r_accept.mutated is True
    assert session.utterance_map()["u4"].speaker == "candidate"
    assert session.pending_reviews == []
    assert validate_session_invariants(session) == []

    # 3. WebSocket `flip_utterance_speaker` and `resolve_review` end-to-end
    app = create_app(project_root=tmp_path / "ws_triage", preload_asr=False)
    client = TestClient(app)
    with client.websocket_connect("/ws/session") as ws:
        _ = ws.receive_json()  # session_state
        _ = ws.receive_json()  # asr_status
        ws.send_json(
            {
                "type": "inject_utterance",
                "id": "utt_ws_flip",
                "speaker": "interviewer",
                "text": "Tell me about a time you collaborated across teams.",
                "start_ms": 1000,
                "end_ms": 4000,
            }
        )
        _ = ws.receive_json()

        # Flip speaker via WS
        ws.send_json({"type": "flip_utterance_speaker", "utterance_id": "utt_ws_flip"})
        s_flipped = ws.receive_json()
        assert s_flipped["session"]["utterances"][0]["speaker"] == "candidate"
        assert s_flipped["session"]["utterances"][0]["rev"] == 2
        assert s_flipped["session"]["state_machines"]["invariants_ok"] is True

        # Queue a review directly and dismiss via WS
        rev_res = app.state.runtime.grouper.apply_turn_triage(
            app.state.runtime.active_session,
            TurnTriagePlan(
                utterance_id="utt_ws_flip",
                segments=[TurnSegment(speaker="interviewer", text="Tell me about a time you collaborated across teams.")],
                confidence="low",
                rationale="Question phrasing looks like interviewer.",
            ),
        )
        assert rev_res.queued_review_id is not None
        ws.send_json({"type": "resolve_review", "review_id": rev_res.queued_review_id, "accept": True})
        s_resolved = ws.receive_json()
        assert s_resolved["session"]["utterances"][0]["speaker"] == "interviewer"
        assert s_resolved["session"]["pending_reviews"] == []
        assert s_resolved["session"]["state_machines"]["invariants_ok"] is True


def test_acoustic_wespeaker_single_mic_diarizer() -> None:
    from mock_helper.diarizer import OnlineTwoSpeakerDiarizer, WeSpeakerEmbedder

    embedder = WeSpeakerEmbedder()
    embedder.ensure_loaded()
    diarizer = OnlineTwoSpeakerDiarizer(embedder)

    def _to_f32(pcm_bytes: bytes) -> np.ndarray:
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Two distinct voices (Daniel = Speaker 1 / You, Samantha = Speaker 2 / Friend or Candidate)
    v1_a = _to_f32(
        _synthesize_pcm16(
            "Hey, I built a local mock interview transcription tool and I want to see if it tells us apart.",
            voice="Daniel",
            trailing_silence_s=0.0,
        )
    )
    v2_a = _to_f32(
        _synthesize_pcm16(
            "Oh nice, do you actually expect it to do real voice recognition instead of guessing from text?",
            voice="Samantha",
            trailing_silence_s=0.0,
        )
    )
    v1_b = _to_f32(
        _synthesize_pcm16(
            "Yeah, it runs a local neural voice embedding model directly on Apple Silicon.",
            voice="Daniel",
            trailing_silence_s=0.0,
        )
    )
    v2_b = _to_f32(
        _synthesize_pcm16(
            "That makes way more sense, especially when I ask you questions or jump in randomly.",
            voice="Samantha",
            trailing_silence_s=0.0,
        )
    )

    s1 = diarizer.diarize_segment(v1_a, 0, int(len(v1_a) / 16))
    assert len(s1) == 1 and s1[0].speaker == "interviewer"
    diarizer.bind_utterance("u1", s1[0])

    s2 = diarizer.diarize_segment(v2_a, 6000, 6000 + int(len(v2_a) / 16))
    assert len(s2) == 1 and s2[0].speaker == "candidate"
    diarizer.bind_utterance("u2", s2[0])

    # Even when Speaker 2 asks a question or Speaker 1 answers, pure acoustics stays 100% accurate
    s3 = diarizer.diarize_segment(v1_b, 12000, 12000 + int(len(v1_b) / 16))
    assert len(s3) == 1 and s3[0].speaker == "interviewer"

    s4 = diarizer.diarize_segment(v2_b, 18000, 18000 + int(len(v2_b) / 16))
    assert len(s4) == 1 and s4[0].speaker == "candidate"

    # Zero-pause rapid mid-segment handoff (Daniel -> Samantha concatenated back-to-back with only 150ms gap)
    tiny_gap = np.zeros(int(16000 * 0.15), dtype=np.float32)
    mixed = np.concatenate([v1_b, tiny_gap, v2_b])
    s_mix = diarizer.diarize_segment(mixed, 25000, 25000 + int(len(mixed) / 16))
    assert len(s_mix) == 2
    assert s_mix[0].speaker == "interviewer"
    assert s_mix[1].speaker == "candidate"
    assert s_mix[0].end_ms == s_mix[1].start_ms

    # Manual correction feedback: flipping u1 & u2 swaps centroids cleanly
    assert diarizer.record_manual_correction("u1", "candidate") is True
    assert diarizer.record_manual_correction("u2", "interviewer") is True
    s_after_swap = diarizer.diarize_segment(v2_a, 40000, 40000 + int(len(v2_a) / 16))
    assert s_after_swap[0].speaker == "interviewer"


def test_diarizer_sparse_cold_start_and_self_healing() -> None:
    from mock_helper.diarizer import OnlineTwoSpeakerDiarizer, WeSpeakerEmbedder

    embedder = WeSpeakerEmbedder()
    embedder.ensure_loaded()

    def _to_f32(pcm_bytes: bytes) -> np.ndarray:
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    lets_word = _to_f32(_synthesize_pcm16("Let's", voice="Daniel", trailing_silence_s=0.0))
    # Embed a short 1-word clip inside a sparse 1.5s lead-in window (mostly silence)
    sparse_lead = np.concatenate(
        [
            np.zeros(int(16000 * 0.45), dtype=np.float32),
            lets_word[: int(16000 * 0.30)],
            np.zeros(int(16000 * 0.85), dtype=np.float32),
        ]
    )
    v1_a = _to_f32(
        _synthesize_pcm16(
            "Testing, are we talking right now? Yeah we are talking. Are you talking too?",
            voice="Daniel",
            trailing_silence_s=0.0,
        )
    )
    v2_a = _to_f32(
        _synthesize_pcm16(
            "Yes, I am talking right now and testing whether it separates our voices.",
            voice="Samantha",
            trailing_silence_s=0.0,
        )
    )
    v1_b = _to_f32(
        _synthesize_pcm16(
            "Okay great, now let's keep going back and forth so we can verify stability.",
            voice="Daniel",
            trailing_silence_s=0.0,
        )
    )
    v2_b = _to_f32(
        _synthesize_pcm16(
            "Sounds good, I am speaking again as the second person in the room.",
            voice="Samantha",
            trailing_silence_s=0.0,
        )
    )

    # 1. Exact reproduction of 20261120_121200_b858: sparse "Let's" lead-in on a long first chunk
    d1 = OnlineTwoSpeakerDiarizer(embedder)
    first_chunk = np.concatenate([sparse_lead, v1_a, np.zeros(3200, dtype=np.float32), v2_a])
    res_first = d1.diarize_segment(first_chunk, 1000, 1000 + int(len(first_chunk) / 16))
    assert len(res_first) >= 1

    # Subsequent turns by both speakers MUST NOT get stuck on candidate!
    r_v1 = d1.diarize_segment(v1_b, 15000, 15000 + int(len(v1_b) / 16))
    r_v2 = d1.diarize_segment(v2_b, 22000, 22000 + int(len(v2_b) / 16))
    assert r_v1[0].speaker == "interviewer"
    assert r_v2[0].speaker == "candidate"

    # 2. Standalone short sparse lead-in ("Let's" + silence) must NOT poison speaker 1 centroid
    d2 = OnlineTwoSpeakerDiarizer(embedder)
    _ = d2.diarize_segment(sparse_lead, 0, int(len(sparse_lead) / 16))
    assert d2.status_dict()["enrolled_speakers"] == []
    assert d2.diarize_segment(v1_a, 2000, 2000 + int(len(v1_a) / 16))[0].speaker == "interviewer"
    assert d2.diarize_segment(v2_a, 8000, 8000 + int(len(v2_a) / 16))[0].speaker == "candidate"
    assert d2.diarize_segment(v1_b, 14000, 14000 + int(len(v1_b) / 16))[0].speaker == "interviewer"

    # 3. Cold-start self-healing: even if interviewer centroid starts poisoned with synthetic noise,
    #    accumulating >= 4 voiced turns from both voices automatically re-clusters cleanly.
    d3 = OnlineTwoSpeakerDiarizer(embedder)
    rng = np.random.default_rng(42)
    poison = rng.normal(size=256).astype(np.float32)
    poison /= np.linalg.norm(poison)
    d3._centroids["interviewer"] = poison
    d3._counts["interviewer"] = 1

    # Feed 4 turns (2 Daniel, 2 Samantha); initially all look far from random poison -> self-heals!
    d3.diarize_segment(v1_a, 0, int(len(v1_a) / 16))
    d3.diarize_segment(v2_a, 6000, 6000 + int(len(v2_a) / 16))
    d3.diarize_segment(v1_b, 12000, 12000 + int(len(v1_b) / 16))
    d3.diarize_segment(v2_b, 18000, 18000 + int(len(v2_b) / 16))

    # After self-healing, Daniel -> interviewer and Samantha -> candidate!
    assert d3.diarize_segment(v1_a, 24000, 24000 + int(len(v1_a) / 16))[0].speaker == "interviewer"
    assert d3.diarize_segment(v2_a, 30000, 30000 + int(len(v2_a) / 16))[0].speaker == "candidate"


def test_diarizer_swap_roles_and_ws_command(tmp_path: Path) -> None:
    from mock_helper.diarizer import DiarizedSubSegment, OnlineTwoSpeakerDiarizer, WeSpeakerEmbedder

    embedder = WeSpeakerEmbedder()
    d = OnlineTwoSpeakerDiarizer(embedder)
    c_a = np.zeros(256, dtype=np.float32)
    c_a[0] = 1.0
    c_b = np.zeros(256, dtype=np.float32)
    c_b[1] = 1.0
    d._centroids["interviewer"] = c_a
    d._centroids["candidate"] = c_b
    d._counts["interviewer"] = 3
    d._counts["candidate"] = 5
    d._last_speaker = "interviewer"
    d._first_speaker = "interviewer"
    d.bind_utterance(
        "utt_1",
        DiarizedSubSegment(
            speaker="interviewer",
            audio=np.zeros(16000, dtype=np.float32),
            start_ms=0,
            end_ms=1000,
            confidence=0.9,
            embedding=c_a,
        ),
    )

    assert d.has_manual_override is False
    assert d.swap_roles(mark_manual=False) is True
    assert d.has_manual_override is False
    assert np.allclose(d._centroids["interviewer"], c_b)
    assert np.allclose(d._centroids["candidate"], c_a)
    assert d._counts == {"interviewer": 5, "candidate": 3}
    assert d._last_speaker == "candidate"
    assert d._utt_history["utt_1"][2] == "candidate"

    assert d.swap_roles(mark_manual=True) is True
    assert d.has_manual_override is True

    # Also test WebSocket swap_all_speakers command
    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    rt.grouper.ingest_utterance(
        rt.active_session,
        speaker="interviewer",
        text="I'm doing good, how are you?",
        start_ms=0,
        end_ms=1500,
        utt_id="u_cand_first",
    )
    rt.grouper.ingest_utterance(
        rt.active_session,
        speaker="candidate",
        text="Doing great! Ready for your Meta interview tomorrow?",
        start_ms=1600,
        end_ms=4200,
        utt_id="u_int_second",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws/session") as ws:
        _ = ws.receive_json()  # initial state
        ws.send_json({"type": "swap_all_speakers"})
        swapped_msg = None
        for _ in range(5):
            msg = ws.receive_json()
            if msg.get("roles_swapped") is True:
                swapped_msg = msg
                break
        assert swapped_msg is not None
    umap = rt.active_session.utterance_map()
    assert umap["u_cand_first"].speaker == "candidate"
    assert umap["u_int_second"].speaker == "interviewer"
    assert rt.diarizer.has_manual_override is True
    assert rt.active_session.id in rt.role_orientation_checked


def test_fast_kaldi_fbank_and_sliding_windows_equivalence() -> None:
    from mock_helper.diarizer import WeSpeakerEmbedder, extract_kaldi_fbank80

    embedder = WeSpeakerEmbedder()
    embedder.ensure_loaded()

    rng = np.random.default_rng(42)
    # 4.23s audio (deliberately non-multiple of 160 so tail window tests fallback branch too)
    audio = (rng.standard_normal(67683) * 0.05).astype(np.float32)
    win_n = 24000  # 1.5s
    hop_n = 9600   # 0.6s (exact multiple of 160)
    starts = list(range(0, len(audio) - win_n + 1, hop_n))
    starts.append(len(audio) - win_n)  # non-aligned tail start

    # Verify individual fbank shape & finite output
    fb = extract_kaldi_fbank80(audio[:win_n])
    assert fb.shape == (148, 80)
    assert np.isfinite(fb).all()

    embs_fast = embedder.embed_sliding_windows(audio, starts, win_n)
    embs_ref = embedder.embed_batch([audio[st : st + win_n] for st in starts])
    assert embs_fast.shape == embs_ref.shape == (len(starts), 256)
    cos_sims = np.sum(embs_fast * embs_ref, axis=1)
    assert float(cos_sims.min()) > 0.9999


def test_parakeet_fast_tdt_decoder_and_optimization_parity() -> None:
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel
    from mock_helper.asr import CONTROL_TOKEN_RE, WHITESPACE_RE, ParakeetEngine

    engine = ParakeetEngine()
    model = engine.ensure_loaded()
    assert engine._fast_decoder is not None
    assert "amx-tdt" in engine._optimization_mode

    pcm_bytes = _synthesize_pcm16(
        "We migrated the checkout queue to Redis and cut p99 latency in half.",
        trailing_silence_s=0.3,
    )
    pcm_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    fast_res = engine.transcribe_pcm(pcm_f32)
    assert fast_res.text

    # Compare FastTDTGreedyDecoder directly against upstream model.generate() on the same padded audio
    pad_n = int(model.preprocessor_config.sample_rate * 0.2)
    padded = np.zeros(pcm_f32.size + 2 * pad_n, dtype=np.float32)
    padded[pad_n : pad_n + pcm_f32.size] = pcm_f32
    mel = get_logmel(mx.array(padded), model.preprocessor_config)
    upstream_raw = model.generate(mel)[0].text
    upstream_clean = WHITESPACE_RE.sub(" ", CONTROL_TOKEN_RE.sub("", upstream_raw)).strip()

    assert fast_res.text == upstream_clean


def test_storage_debounced_wav_and_fast_list_recent_parity(tmp_path: Path) -> None:
    import time as _time
    import wave
    from mock_helper.storage import SessionAudioArchive, SessionStore

    arch = SessionAudioArchive(tmp_path, "sess_debounce", header_sync_interval_s=0.05)
    chunk = np.ones(800, dtype=np.int16).tobytes()
    for _ in range(3):
        arch.append_frame(bytes([CHANNEL_INTERVIEWER]) + chunk)
    _time.sleep(0.06)
    arch.append_frame(bytes([CHANNEL_INTERVIEWER]) + chunk)
    # Periodic sync should have flushed valid header even before close()
    with wave.open(str(tmp_path / "sess_debounce_interviewer.wav"), "rb") as wf:
        assert wf.getnframes() == 3200
    arch.append_frame(bytes([CHANNEL_INTERVIEWER]) + chunk)
    arch.close()
    with wave.open(str(tmp_path / "sess_debounce_interviewer.wav"), "rb") as wf:
        assert wf.getnframes() == 4000

    store = SessionStore(tmp_path)
    sess = InterviewSession(
        id="20261120_999999_perf",
        config=SessionConfig(role_title="Staff Engineer", interview_type="system_design", mode="in_person"),
    )
    grouper = QAGrouper()
    grouper.ingest_utterance(sess, speaker="interviewer", text="Design a rate limiter.", start_ms=1000, end_ms=3500)
    grouper.ingest_utterance(sess, speaker="candidate", text="I would use token buckets.", start_ms=4000, end_ms=9200)
    sess.calibration_markdown = "## Verdict\nStrong Hire"
    store.save(sess)

    recent = store.list_recent()
    assert len(recent) == 1
    item = recent[0]
    metrics = sess.compute_metrics()
    assert item["id"] == sess.id
    assert item["role_title"] == "Staff Engineer"
    assert item["interview_type"] == "system_design"
    assert item["mode"] == "in_person"
    assert item["qa_count"] == metrics.qa_count
    assert item["duration_seconds"] == metrics.duration_seconds
    assert item["has_calibration"] is True




