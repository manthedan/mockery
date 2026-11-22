from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mockery import _format_url_host, _is_loopback_host
from mockery.addons import _hf_repo_cached, _invalidate_hf_repo_cache
from mockery.asr import ParakeetEngine, parakeet_snapshot_is_complete
from mockery.models import InterviewSession, Utterance
from mockery.paths import _is_mockery_checkout, resolve_data_root
from mockery.profiles import CandidateProfileStore, ResumeRedactionPrefs
from mockery.redactor import RampartRedactor
from mockery.storage import SessionStore
from mockery.tts import LocalTTSEngine
from mockery.vad import SileroVADRuntime


def test_silero_vad_runtime_is_lazy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def _boom(*args: Any, **kwargs: Any) -> str:
        nonlocal called
        called = True
        raise RuntimeError("offline")

    monkeypatch.setattr("mockery.vad.hf_hub_download", _boom)
    rt = SileroVADRuntime()
    assert rt.is_ready is False
    assert called is False
    assert rt.status_dict() == {"ready": False, "error": None}


def test_parakeet_stream_cleanup_does_not_block_on_inference_lock() -> None:
    engine = ParakeetEngine(model_id="mlx-community/parakeet-tdt-0.6b-v2")
    with engine._streams_lock:
        engine._streams["utt_1"] = object()  # type: ignore[assignment]
        engine._streams["utt_2"] = object()  # type: ignore[assignment]

    # Hold the inference lock on another thread and verify stream cleanup finishes immediately.
    acquired = threading.Event()
    release = threading.Event()

    def _hold_inference() -> None:
        with engine._lock:
            acquired.set()
            release.wait(timeout=5.0)

    t = threading.Thread(target=_hold_inference, daemon=True)
    t.start()
    assert acquired.wait(timeout=2.0)
    try:
        t0 = time.perf_counter()
        engine.drop_stream("utt_1")
        engine.clear_streams()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.25
        with engine._streams_lock:
            assert len(engine._streams) == 0
    finally:
        release.set()
        t.join(timeout=2.0)


def test_parakeet_completeness_rejects_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hub"
    repo_id = "mlx-community/parakeet-tdt-0.6b-v2"
    snap = hub / "models--mlx-community--parakeet-tdt-0.6b-v2" / "snapshots" / "rev123"
    refs = hub / "models--mlx-community--parakeet-tdt-0.6b-v2" / "refs"
    snap.mkdir(parents=True)
    refs.mkdir(parents=True)
    (refs / "main").write_text("rev123", encoding="utf-8")

    # Config-only partial download must NOT count as installed
    (snap / "config.json").write_text('{"model_type":"parakeet"}', encoding="utf-8")
    assert parakeet_snapshot_is_complete(snap) is False

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    _invalidate_hf_repo_cache()
    assert _hf_repo_cached(repo_id, bypass_cache=True, snapshot_validator=parakeet_snapshot_is_complete) is False

    # Adding non-empty weights completes the snapshot
    (snap / "model.safetensors").write_bytes(b"\x01\x02\x03\x04")
    assert parakeet_snapshot_is_complete(snap) is True
    assert _hf_repo_cached(repo_id, bypass_cache=True, snapshot_validator=parakeet_snapshot_is_complete) is True


def test_phonon_stream_startup_cancel_releases_lock(tmp_path: Path) -> None:
    tts = LocalTTSEngine(tmp_path / "models")
    started = threading.Event()
    gate = threading.Event()

    class _FakePhonon:
        sample_rate = 24000

        def stream(self, text: str, voice: str = "default"):
            started.set()
            gate.wait(timeout=3.0)
            for _ in range(12):
                yield np.zeros(1200, dtype=np.float32)

    tts._phonon = _FakePhonon()
    tts._phonon_installed_override = True  # type: ignore[attr-defined]
    tts.is_phonon_installed = lambda: True  # type: ignore[method-assign]
    tts.is_phonon_available = lambda: True  # type: ignore[method-assign]

    async def _run() -> None:
        task = asyncio.create_task(tts._try_phonon_stream("Synthetic test speech.", voice="default"))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    # Producer must observe stop_event and release _phonon_lock promptly (no permanent wedge)
    assert tts._phonon_lock.acquire(timeout=1.5)
    tts._phonon_lock.release()


def test_profile_resume_upload_clears_unredacted_values_and_blocks_stale_prefs(tmp_path: Path) -> None:
    redactor = RampartRedactor()
    store = CandidateProfileStore(tmp_path / "profiles", redactor)

    p1 = store.set_resume(
        filename="resume_a.txt",
        raw_text="Jordan Taylor\nsample@example.invalid\nBuilt distributed caches at Acme Corp.",
    )
    assert p1 is not None
    email_span = next((s for s in p1.spans if s.get("label") == "EMAIL"), None)
    assert email_span is not None

    # User explicitly unmasks email on Resume A
    toggled = store.toggle_span_redaction(email_span["id"])
    assert toggled is not None
    assert "sample@example.invalid" in toggled.redacted_text
    assert toggled.prefs.unredacted_values

    # Uploading Resume B must NOT carry over Resume A's unredacted_values
    p2 = store.set_resume(
        filename="resume_b.txt",
        raw_text="Jordan Taylor\nsample@example.invalid\nLed ML platform migration.",
    )
    assert p2 is not None
    assert p2.prefs.unredacted_values == []
    assert "sample@example.invalid" not in p2.redacted_text

    # Concurrent stale update_redaction_prefs must never revert a newer upload
    entered_b = threading.Event()
    proceed_b = threading.Event()
    entered_prefs = threading.Event()
    proceed_prefs = threading.Event()
    orig_redact = redactor.redact

    def _controlled_redact(text: str, **kwargs: Any):
        if "RESUME_NEW_B" in text:
            entered_b.set()
            proceed_b.wait(timeout=3.0)
        elif "Led ML platform migration." in text:
            entered_prefs.set()
            proceed_prefs.wait(timeout=3.0)
        return orig_redact(text, **kwargs)

    redactor.redact = _controlled_redact  # type: ignore[method-assign]

    t_upload = threading.Thread(
        target=lambda: store.set_resume(filename="b.txt", raw_text="RESUME_NEW_B senior engineer"),
        daemon=True,
    )
    t_upload.start()
    assert entered_b.wait(timeout=2.0)

    # Now start update_redaction_prefs while B's upload has bumped revision but hasn't committed yet
    t_prefs = threading.Thread(
        target=lambda: store.update_redaction_prefs(ResumeRedactionPrefs(keep_location=False)),
        daemon=True,
    )
    t_prefs.start()
    assert entered_prefs.wait(timeout=2.0)

    # Let B commit first, then let stale prefs try to commit
    proceed_b.set()
    t_upload.join(timeout=3.0)
    proceed_prefs.set()
    t_prefs.join(timeout=3.0)

    assert store.state.resume is not None
    assert store.state.resume.filename == "b.txt"
    assert "RESUME_NEW_B" in store.state.resume.raw_text


def test_storage_list_recent_includes_journal_only_and_corrupt_snapshots(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    sid = "20261122_120000_crashrec"
    utt = Utterance(
        id="utt_01",
        rev=1,
        final=True,
        speaker="candidate",
        text="We migrated our queue workers to bounded backpressure.",
        start_ms=1000,
        end_ms=5500,
    )
    store.append_utterance_event(sid, utt)

    # Journal-only session (no .json yet) must appear in list_recent
    recent = store.list_recent()
    assert any(item["id"] == sid and item.get("recovered") is True for item in recent)

    # Corrupt .json snapshot must still recover via journal in list_recent
    (tmp_path / "sessions" / f"{sid}.json").write_text("{corrupt json", encoding="utf-8")
    recent_corrupt = store.list_recent()
    assert any(item["id"] == sid and item.get("recovered") is True for item in recent_corrupt)


def test_resolve_data_root_ignores_unrelated_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_repo = tmp_path / "unrelated_project"
    other_repo.mkdir()
    (other_repo / "pyproject.toml").write_text('[project]\nname = "other-app"\n', encoding="utf-8")
    (other_repo / ".env.local").write_text("SECRET=1\n", encoding="utf-8")

    monkeypatch.delenv("MOCKERY_HOME", raising=False)
    monkeypatch.chdir(other_repo)
    assert _is_mockery_checkout(other_repo) is False
    resolved = resolve_data_root()
    assert resolved != other_repo

    # Actual Mockery checkout IS detected
    fake_mockery = tmp_path / "mockery_checkout"
    (fake_mockery / "src" / "mockery").mkdir(parents=True)
    (fake_mockery / "pyproject.toml").write_text('[project]\nname = "mockery"\n', encoding="utf-8")
    monkeypatch.chdir(fake_mockery)
    assert _is_mockery_checkout(fake_mockery) is True
    assert resolve_data_root() == fake_mockery


def test_cli_loopback_and_ipv6_url_helpers() -> None:
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("127.0.0.2") is True
    assert _is_loopback_host("localhost") is True
    assert _is_loopback_host("::1") is True
    assert _is_loopback_host("[::1]") is True
    assert _is_loopback_host("0.0.0.0") is False
    assert _is_loopback_host("192.168.1.25") is False

    assert _format_url_host("127.0.0.1") == "127.0.0.1"
    assert _format_url_host("::1") == "[::1]"
    assert _format_url_host("0.0.0.0") == "127.0.0.1"
    assert _format_url_host("::") == "127.0.0.1"


def test_switch_session_replaces_diarizer_instance_sharing_embedder(tmp_path: Path) -> None:
    from mockery.server import AppRuntime

    rt = AppRuntime(tmp_path)
    old_diarizer = rt.diarizer
    old_embedder = rt.voice_embedder
    rt.switch_session(InterviewSession())
    assert rt.diarizer is not old_diarizer
    assert rt.diarizer.embedder is old_embedder


def test_rampart_uninstall_preserves_existing_redacted_resume(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    # Seed a resume redacted by Rampart (including a span only Rampart would have masked)
    rt.profiles.set_resume(
        filename="resume.txt",
        raw_text="Alex Rivera worked at Nebulon Labs on telemetry.",
    )
    assert rt.profiles.state.resume is not None
    rt.profiles.state.resume.redacted_text = "[PERSON_1] worked at [ORG_1] on telemetry."
    rt.profiles.state.resume.engine = "rampart_onnx"
    rt.profiles.state.resume.redacted_count = 2

    # Make rampart_pii report installed so DELETE /api/addons/rampart_pii runs uninstall
    rt.addons.get_pack_spec = lambda pid: {  # type: ignore[method-assign]
        "id": pid,
        "state": "installed",
        "allowed_events": ["uninstall"],
        "blocked_reasons": {},
    }

    async def _fake_uninstall(pack_id: str) -> dict[str, Any]:
        return rt.addons.status_dict()

    rt.addons.uninstall_pack = _fake_uninstall  # type: ignore[method-assign]

    client = TestClient(app)
    resp = client.delete("/api/addons/rampart_pii")
    assert resp.status_code == 200
    assert rt.profiles.state.resume is not None
    assert rt.profiles.state.resume.redacted_text == "[PERSON_1] worked at [ORG_1] on telemetry."
    assert rt.profiles.state.resume.engine == "rampart_onnx"


def test_ws_owner_disconnect_flushes_trailing_audio_and_saves(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.asr import ASRTranscription
    from mockery.server import create_app
    from mockery.sources import CHANNEL_INTERVIEWER

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime

    # Deterministic synthetic VAD that always classifies non-zero frames as speech
    rt.vad_runtime._session = object()  # type: ignore[assignment]
    rt.vad_runtime.step = lambda frame, state: (0.95, state)  # type: ignore[method-assign]
    rt.asr.transcribe_pcm = lambda pcm: ASRTranscription(  # type: ignore[method-assign]
        text="Trailing spoken answer captured on disconnect.",
        inference_ms=5.0,
        audio_duration_ms=len(pcm) / 16.0,
    )

    client = TestClient(app)
    # 512ms of non-zero 16kHz int16 audio (enough frames >= min_speech_ms, no trailing silence)
    tone = (np.sin(np.linspace(0, 40 * np.pi, 8192)) * 12000).astype(np.int16).tobytes()
    with client.websocket_connect("/ws/session") as ws:
        init_msg = ws.receive_json()
        sid = init_msg["session"]["id"]
        _ = ws.receive_json()  # asr_status
        ws.send_json({"type": "start_capture"})
        _ = ws.receive_json()  # session_state
        ws.send_bytes(bytes([CHANNEL_INTERVIEWER]) + tone)
        # Ensure the binary frame is ingested by the server loop before simulating an abrupt client disconnect
        ws.send_json({"type": "set_manual_speaker", "speaker": "interviewer"})
        _ = ws.receive_json()
        ws.close()

    # Trailing audio must have been flushed, transcribed, and saved to disk
    loaded = rt.store.load(sid)
    assert loaded is not None
    assert any("Trailing spoken answer" in u.text for u in loaded.utterances)
    assert rt.is_capturing is False


def test_rules_redactor_detects_accented_unicode_header_name() -> None:
    redactor = RampartRedactor()
    res = redactor.redact(
        "María Núñez\nSenior Distributed Systems Engineer\nBuilt low-latency RPC gateways.",
        use_rampart=False,
    )
    assert "María Núñez" not in res.redacted_text
    assert any(sp.get("label") == "PERSON_NAME" for sp in res.spans)


def test_stale_tab_resume_toggle_rejected_with_409(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    client = TestClient(app)

    r1 = client.post(
        "/api/profile/resume/text",
        json={"filename": "a.txt", "text": "Alex Rivera\nfirst@example.invalid\nEngineer"},
    )
    assert r1.status_code == 200
    doc1 = r1.json()["resume"]["doc_id"]
    span1 = r1.json()["resume"]["spans"][0]["id"]

    r2 = client.post(
        "/api/profile/resume/text",
        json={"filename": "b.txt", "text": "Sam Patel\nsecond@example.invalid\nEngineer"},
    )
    assert r2.status_code == 200
    doc2 = r2.json()["resume"]["doc_id"]
    assert doc1 != doc2

    # Stale tab sending doc1 must get 409 and NOT unmask Resume B
    stale = client.post(f"/api/profile/resume/toggle/{span1}?doc_id={doc1}")
    assert stale.status_code == 409
    assert "second@example.invalid" not in app.state.runtime.profiles.resume_for_llm()


def test_loopback_host_127_0_0_2_allowed_by_server(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    client = TestClient(app)
    resp = client.get("/api/machines", headers={"Host": "127.0.0.2:8765"})
    assert resp.status_code == 200
    bad = client.get("/api/machines", headers={"Host": "evil.example:8765"})
    assert bad.status_code == 403


def test_list_recent_isolates_unreadable_recovery_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions")
    good = InterviewSession(id="20261122_120001_good")
    store.save(good)
    bad_journal = tmp_path / "sessions" / "20261122_120002_bad.events.jsonl"
    bad_journal.write_text('{"type":"utterance_upsert"}\n', encoding="utf-8")

    orig_recover = store._recover_from_jsonl

    def _boom_on_bad(sid: str):
        if sid == "20261122_120002_bad":
            raise PermissionError("unreadable journal")
        return orig_recover(sid)

    monkeypatch.setattr(store, "_recover_from_jsonl", _boom_on_bad)
    recent = store.list_recent()
    assert [item["id"] for item in recent] == ["20261122_120001_good"]


def test_parakeet_completeness_requires_exact_model_safetensors(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "other.safetensors").write_bytes(b"\x01\x02")
    assert parakeet_snapshot_is_complete(snap) is False
    (snap / "model.safetensors").write_bytes(b"\x01\x02")
    assert parakeet_snapshot_is_complete(snap) is True


def test_tts_stream_releases_semaphore_on_header_disconnect(tmp_path: Path) -> None:
    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime

    closed_count = 0

    class _FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            return b"\x00\x00" * 100

        async def aclose(self) -> None:
            nonlocal closed_count
            closed_count += 1

    async def _fake_open_stream(*args: Any, **kwargs: Any):
        return ("phonon", 24000, _FakeStream())

    rt.tts.open_pcm_stream = _fake_open_stream  # type: ignore[method-assign]

    async def _run() -> None:
        async def _send_fail(message: dict[str, Any]) -> None:
            raise RuntimeError("client disconnected before response headers")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/tts/stream",
            "raw_path": b"/api/tts/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"127.0.0.1:8765"),
                (b"content-type", b"application/json"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8765),
        }
        for _ in range(3):
            sent_body = False

            async def _receive() -> dict[str, Any]:
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": b'{"text":"hello"}', "more_body": False}
                return {"type": "http.disconnect"}

            with pytest.raises(RuntimeError):
                await app(scope, _receive, _send_fail)
        assert rt._tts_sem._value == 2
        assert closed_count == 3

    asyncio.run(_run())


def test_failed_vad_closes_gate_and_stops_packet_spam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app
    from mockery.sources import CHANNEL_INTERVIEWER

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    calls = 0

    def _fail_vad() -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("offline VAD missing")

    monkeypatch.setattr(rt.vad_runtime, "ensure_loaded", _fail_vad)
    client = TestClient(app)
    pcm = np.zeros(1600, dtype=np.int16).tobytes()
    with client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"type": "start_capture"})
        err_msg = ws.receive_json()
        assert err_msg["type"] == "capture_error"
        # Subsequent PCM frames while gate is closed must NOT re-attempt VAD load
        ws.send_bytes(bytes([CHANNEL_INTERVIEWER]) + pcm)
        ws.send_bytes(bytes([CHANNEL_INTERVIEWER]) + pcm)
        ws.send_json({"type": "set_manual_speaker", "speaker": "interviewer"})
        sync = ws.receive_json()
        assert sync["type"] == "manual_speaker"
    assert calls == 1


def test_non_owner_tab_cannot_hijack_or_cancel_active_capture(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    rt.vad_runtime._session = object()  # type: ignore[assignment]
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws1, client.websocket_connect("/ws/session") as ws2:
        ws1.receive_json()
        ws1.receive_json()
        ws2.receive_json()
        ws2.receive_json()

        ws1.send_json({"type": "start_capture"})
        s1 = ws1.receive_json()
        _ = ws2.receive_json()
        orig_id = s1["session"]["id"]

        # Second tab must be rejected when trying to hijack with new_session or cancel_capture
        ws2.send_json({"type": "new_session", "config": {}})
        rej1 = ws2.receive_json()
        assert rej1["type"] == "transition_rejected"
        assert rt.active_session.id == orig_id
        assert rt.is_capturing is True

        ws2.send_json({"type": "cancel_capture"})
        rej2 = ws2.receive_json()
        assert rej2["type"] == "transition_rejected"
        assert rt.is_capturing is True


def test_practice_debrief_records_final_answer_directly_without_turn_call(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.practice import PracticeConfig, PracticeQuestionItem, PracticeSession
    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    sess = PracticeSession(
        role_title="Senior Engineer",
        config=PracticeConfig(interview_type="behavioral"),
        questions=[
            PracticeQuestionItem(
                id="q_1",
                title="Conflict resolution",
                prompt="Tell me about a tough disagreement.",
                status="active",
            )
        ],
    )
    rt.practice_store.save(sess)

    seen_turns: list[dict[str, Any]] = []

    async def _fake_debrief(**kwargs: Any) -> dict[str, Any]:
        seen_turns.extend(kwargs.get("turns") or [])
        return {
            "overall_summary": "Solid answer.",
            "strengths": ["Clear outcome"],
            "focus_areas": ["Add metric"],
            "question_reviews": [
                {
                    "question_id": "q_1",
                    "carl": {"context": "strong", "action": "strong", "result": "partial", "learning": "partial"},
                    "what_worked": "Good framing",
                    "upgrade_tip": "Quantify latency",
                }
            ],
            "retry_drill": None,
        }

    rt.llm.generate_practice_debrief = _fake_debrief  # type: ignore[method-assign]
    client = TestClient(app)
    resp = client.post(
        f"/api/practice/{sess.id}/debrief",
        json={"final_answer": "I aligned both teams around shared SLO latency budgets.", "expected_question_id": "q_1"},
    )
    assert resp.status_code == 200
    assert any("shared SLO latency budgets" in t["text"] for t in seen_turns)
    assert resp.json()["session"]["phase"] == "completed"


def test_rules_redactor_detects_decomposed_nfd_unicode_header_name() -> None:
    import unicodedata

    redactor = RampartRedactor()
    nfc_name = unicodedata.normalize("NFC", "María Núñez")
    nfd_name = unicodedata.normalize("NFD", "María Núñez")
    res = redactor.redact(
        f"{nfd_name}\nStaff Infrastructure Engineer\nSummary: {nfc_name} designed multi-region failover.",
        use_rampart=False,
    )
    assert nfd_name not in res.redacted_text
    assert nfc_name not in res.redacted_text
    assert any(sp.get("label") == "PERSON_NAME" for sp in res.spans)


def test_practice_turn_real_asgi_disconnect_cancels_and_does_not_commit(tmp_path: Path) -> None:
    import json as _json

    from mockery.practice import PracticeConfig, PracticeQuestionItem, PracticeSession
    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    sess = PracticeSession(
        role_title="Senior Engineer",
        config=PracticeConfig(interview_type="behavioral"),
        questions=[
            PracticeQuestionItem(
                id="q_1",
                title="Ownership",
                prompt="Tell me about an outage.",
                status="active",
            )
        ],
    )
    rt.practice_store.save(sess)

    llm_started = asyncio.Event()
    llm_cancelled = False

    async def _slow_reply(**kwargs: Any) -> dict[str, Any]:
        nonlocal llm_cancelled
        llm_started.set()
        try:
            await asyncio.sleep(5.0)
            return {"interviewer_reply": "Should never commit", "should_advance": False}
        except asyncio.CancelledError:
            llm_cancelled = True
            raise

    rt.llm.generate_practice_interviewer_reply = _slow_reply  # type: ignore[method-assign]

    async def _run() -> None:
        body_bytes = _json.dumps({"text": "Cancelled answer", "expected_question_id": "q_1"}).encode("utf-8")
        body_sent = False
        disconnect_ev = asyncio.Event()
        sent_statuses: list[int] = []

        async def _receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            await disconnect_ev.wait()
            return {"type": "http.disconnect"}

        async def _send(msg: dict[str, Any]) -> None:
            if msg.get("type") == "http.response.start":
                sent_statuses.append(int(msg["status"]))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": f"/api/practice/{sess.id}/turn",
            "raw_path": f"/api/practice/{sess.id}/turn".encode(),
            "query_string": b"",
            "headers": [
                (b"host", b"127.0.0.1:8765"),
                (b"content-type", b"application/json"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8765),
        }

        req_task = asyncio.create_task(app(scope, _receive, _send))
        await asyncio.wait_for(llm_started.wait(), timeout=2.0)
        disconnect_ev.set()
        await asyncio.wait_for(req_task, timeout=2.0)

        assert llm_cancelled is True
        assert sent_statuses == [499]
        reloaded = rt.practice_store.get(sess.id)
        assert reloaded is not None
        assert reloaded.turns == []

    asyncio.run(_run())


def test_practice_debrief_final_answer_rejects_stale_q_and_avoids_503_duplication(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.practice import PracticeConfig, PracticeQuestionItem, PracticeSession
    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    sess = PracticeSession(
        role_title="Senior Engineer",
        config=PracticeConfig(interview_type="behavioral"),
        questions=[
            PracticeQuestionItem(id="q_1", title="Q1", prompt="Prompt 1", status="active"),
        ],
    )
    rt.practice_store.save(sess)

    client = TestClient(app)

    # Mismatched expected_question_id must 409 instead of silently dropping final_answer
    stale = client.post(
        f"/api/practice/{sess.id}/debrief",
        json={"final_answer": "My final answer", "expected_question_id": "q_wrong"},
    )
    assert stale.status_code == 409

    # First debrief call fails with 503; retry succeeds -> final_answer must appear EXACTLY ONCE
    attempts = 0

    async def _flaky_debrief(**kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient provider 503")
        return {
            "overall_summary": "Good.",
            "strengths": ["Clear"],
            "focus_areas": ["Metrics"],
            "question_reviews": [
                {
                    "question_id": "q_1",
                    "carl": {"context": "strong", "action": "strong", "result": "partial", "learning": "partial"},
                    "what_worked": "Solid",
                    "upgrade_tip": "Add numbers",
                }
            ],
            "retry_drill": None,
        }

    rt.llm.generate_practice_debrief = _flaky_debrief  # type: ignore[method-assign]
    r1 = client.post(
        f"/api/practice/{sess.id}/debrief",
        json={"final_answer": "My final answer", "expected_question_id": "q_1"},
    )
    assert r1.status_code == 503

    r2 = client.post(
        f"/api/practice/{sess.id}/debrief",
        json={"final_answer": "My final answer", "expected_question_id": "q_1"},
    )
    assert r2.status_code == 200
    reloaded = rt.practice_store.get(sess.id)
    assert reloaded is not None
    matching = [t for t in reloaded.turns if t.text == "My final answer"]
    assert len(matching) == 1


def test_non_owner_tab_cannot_flush_and_end_active_capture(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    rt.vad_runtime._session = object()  # type: ignore[assignment]
    client = TestClient(app)

    with client.websocket_connect("/ws/session") as ws1, client.websocket_connect("/ws/session") as ws2:
        ws1.receive_json()
        ws1.receive_json()
        ws2.receive_json()
        ws2.receive_json()

        ws1.send_json({"type": "start_capture"})
        _ = ws1.receive_json()
        _ = ws2.receive_json()

        ws2.send_json({"type": "flush_and_end"})
        rej = ws2.receive_json()
        assert rej["type"] == "transition_rejected"
        assert rt.is_capturing is True


def test_tts_stream_releases_semaphore_even_when_aclose_raises_cancelled(tmp_path: Path) -> None:
    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime

    class _CancelOnCloseStream:
        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            return b"\x00\x00" * 64

        async def aclose(self) -> None:
            raise asyncio.CancelledError()

    async def _fake_open_stream(*args: Any, **kwargs: Any):
        return ("phonon", 24000, _CancelOnCloseStream())

    rt.tts.open_pcm_stream = _fake_open_stream  # type: ignore[method-assign]

    async def _run() -> None:
        async def _send_fail(message: dict[str, Any]) -> None:
            raise RuntimeError("early disconnect")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/tts/stream",
            "raw_path": b"/api/tts/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"127.0.0.1:8765"),
                (b"content-type", b"application/json"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8765),
        }
        for _ in range(3):
            sent_body = False

            async def _receive() -> dict[str, Any]:
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": b'{"text":"hello"}', "more_body": False}
                return {"type": "http.disconnect"}

            with pytest.raises(RuntimeError):
                await app(scope, _receive, _send_fail)
        assert rt._tts_sem._value == 2

    asyncio.run(_run())


def test_completed_debrief_rejects_new_final_answer_with_409(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.practice import PracticeConfig, PracticeQuestionItem, PracticeSession
    from mockery.server import create_app

    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    sess = PracticeSession(
        role_title="Senior Engineer",
        config=PracticeConfig(interview_type="behavioral"),
        phase="completed",
        questions=[PracticeQuestionItem(id="q_1", title="Q1", prompt="P1", status="completed")],
        debrief={"overall_summary": "Done", "question_reviews": []},
    )
    rt.practice_store.save(sess)
    client = TestClient(app)

    # Cached read with no final_answer still returns 200
    r_ok = client.post(f"/api/practice/{sess.id}/debrief", json={})
    assert r_ok.status_code == 200

    # Submitting a new final_answer to an already-completed debrief must 409
    r_conflict = client.post(
        f"/api/practice/{sess.id}/debrief",
        json={"final_answer": "Unsaved answer from another tab", "expected_question_id": "q_1"},
    )
    assert r_conflict.status_code == 409


def test_lan_token_with_semicolon_survives_cookie_roundtrip(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mockery.server import create_app

    token = "abc;def=ghi"
    app = create_app(project_root=tmp_path, auth_token=token, preload_asr=False)
    client = TestClient(app)

    r1 = client.get("/api/machines", params={"token": token})
    assert r1.status_code == 200
    # Follow-up request relies solely on the Set-Cookie issued by middleware
    r2 = client.get("/api/machines")
    assert r2.status_code == 200



