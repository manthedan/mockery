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

