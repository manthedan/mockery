from __future__ import annotations

import io
import os
import stat
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mockery.practice import (
    PracticeConfig,
    PracticeQuestionItem,
    PracticeSession,
    PracticeSessionStore,
    PracticeTurn,
    build_heuristic_nudge,
    estimate_carl_heuristic,
)
from mockery.server import _decode_audio_to_float32_16k, create_app
from mockery.tts import LocalTTSEngine, _clean_text_for_speech


def _make_wav_bytes(samples: np.ndarray, rate: int = 16000, channels: int = 1) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_clean_text_for_speech_strips_markdown_and_clamps():
    raw = "## Question\n**Tell me** about `Redis` and [this link](https://example.com).\n```python\nx = 1\n```"
    cleaned = _clean_text_for_speech(raw)
    assert "```" not in cleaned
    assert "https://example.com" not in cleaned
    assert "##" not in cleaned
    assert "Question Tell me about Redis and this link." == cleaned
    assert len(_clean_text_for_speech("a" * 3000)) <= 1600


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS native `say` synthesis test")
def test_local_tts_macos_wav_and_flag_safety(tmp_path: Path):
    import asyncio

    engine = LocalTTSEngine(tmp_path / "models")
    status = engine.status_dict()
    assert status["macos_say_available"] is True
    assert status["engine"] in ("macos_say", "kokoro")

    # Leading dashes must never be interpreted as `say` CLI flags
    wav_bytes, used = asyncio.run(engine.synthesize_wav("--voice=bad -f /etc/passwd Hello candidate."))
    assert used in ("macos_say", "kokoro")
    assert wav_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 500


def test_kokoro_verified_download_and_synthesis_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import asyncio
    import hashlib
    import httpx
    import mockery.tts as tts_mod

    engine = LocalTTSEngine(tmp_path / "models")
    assert engine.is_kokoro_available() is True
    assert engine.is_kokoro_installed() is False

    fake_model = b"fake-kokoro-fp16-onnx-bytes"
    fake_voices = b"fake-voices-npz-bytes"
    model_sha = hashlib.sha256(fake_model).hexdigest()
    voices_sha = hashlib.sha256(fake_voices).hexdigest()

    monkeypatch.setattr(
        tts_mod,
        "KOKORO_ASSETS",
        (
            ("kokoro-v1.0.fp16.onnx", "https://example.test/kokoro-v1.0.fp16.onnx", model_sha),
            ("voices-v1.0.bin", "https://example.test/voices-v1.0.bin", voices_sha),
        ),
    )

    # 1. Checksum mismatch must fail and leave no partial/corrupt files behind
    bad_transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"tampered"))
    with httpx.Client(transport=bad_transport) as bad_client:
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            engine.install_kokoro(client=bad_client)
    assert engine.is_kokoro_installed() is False
    assert not (engine.models_dir / "kokoro-v1.0.fp16.onnx").exists()
    assert not (engine.models_dir / "kokoro-v1.0.fp16.onnx.tmp").exists()

    # 2. Valid verified download + stubbed Kokoro runtime routes synthesis to `kokoro`
    class _StubKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            self.model_path = model_path
            self.voices_path = voices_path
            self.calls: list[tuple[str, str, float, str]] = []

        def create(self, text: str, *, voice: str, speed: float, lang: str):
            self.calls.append((text, voice, speed, lang))
            return np.zeros(2400, dtype=np.float32), 24000

    stub_holder: dict[str, _StubKokoro] = {}

    def _fake_ensure_loaded() -> bool:
        mp = engine._resolve_model_path()
        assert mp is not None and mp.name == "kokoro-v1.0.fp16.onnx"
        stub = _StubKokoro(str(mp), str(engine.models_dir / "voices-v1.0.bin"))
        engine._kokoro = stub
        engine._kokoro_error = None
        stub_holder["stub"] = stub
        return True

    monkeypatch.setattr(engine, "_ensure_kokoro_loaded", _fake_ensure_loaded)

    def _ok_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("kokoro-v1.0.fp16.onnx"):
            return httpx.Response(200, content=fake_model)
        if req.url.path.endswith("voices-v1.0.bin"):
            return httpx.Response(200, content=fake_voices)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(_ok_handler)) as ok_client:
        assert engine.install_kokoro(client=ok_client) is True

    assert engine.is_kokoro_installed() is True
    st = engine.status_dict()
    assert st["engine"] == "kokoro"
    assert st["backend_state"] == "kokoro_ready"
    assert st["default_voice"] == "af_heart"

    wav_bytes, used = asyncio.run(engine.synthesize_wav("Tell me about scaling Redis.", voice="bf_emma", rate=1.05))
    assert used == "kokoro"
    assert wav_bytes[:4] == b"RIFF"
    assert stub_holder["stub"].calls[-1] == ("Tell me about scaling Redis.", "bf_emma", 1.05, "en-gb")

    # 3. Uninstall cleans up cleanly
    assert engine.uninstall_kokoro() is True
    assert engine.is_kokoro_installed() is False


def test_carl_heuristic_monotonic_and_nudge():
    c1 = estimate_carl_heuristic("When I worked on our legacy payment project, we had an outage.")
    assert c1["context"] in ("partial", "strong")
    assert c1["result"] == "missing"

    c2 = estimate_carl_heuristic(
        "I designed and implemented a circuit breaker that reduced p99 latency by 45% across 2 million users. "
        "What I learned for next time is to always ship load-shedding dashboards first.",
        existing=c1,
    )
    assert c2["action"] in ("partial", "strong")
    assert c2["result"] == "strong"
    assert c2["learning"] == "strong"

    # Never downgrades an already-strong signal on a short follow-up
    c3 = estimate_carl_heuristic("Yes, exactly.", existing=c2)
    assert c3["result"] == "strong"
    assert c3["learning"] == "strong"

    nudge = build_heuristic_nudge(c3, "behavioral")
    assert nudge["tone"] in ("encourage", "tip")
    assert nudge["message"]


def test_practice_session_store_permissions_and_id_validation(tmp_path: Path):
    store = PracticeSessionStore(tmp_path / "practice")
    sess = PracticeSession(
        role_title="Staff Engineer",
        company="Acme",
        config=PracticeConfig(interview_type="behavioral", target_level="L6"),
        questions=[
            PracticeQuestionItem(
                title="Cross-team alignment",
                prompt="Tell me about aligning two teams with conflicting goals.",
                status="active",
            )
        ],
        turns=[PracticeTurn(role="interviewer", text="Welcome! Let's begin.")],
    )
    store.save(sess)
    saved_path = tmp_path / "practice" / f"{sess.id}.json"
    assert saved_path.is_file()
    mode = stat.S_IMODE(os.stat(saved_path).st_mode)
    assert mode == 0o600

    # Path traversal rejected cleanly
    assert store.get("../etc/passwd") is None
    with pytest.raises(ValueError):
        store._safe_path("../../bad")

    # Fresh store reads back cleanly
    store2 = PracticeSessionStore(tmp_path / "practice")
    loaded = store2.get(sess.id)
    assert loaded is not None
    assert loaded.role_title == "Staff Engineer"
    assert len(store2.list_recent()) == 1

    md = loaded.to_markdown()
    assert "Mockery AI Practice Coaching Report" in md
    assert "Staff Engineer" in md


def test_decode_audio_wav_and_raw_pcm():
    # 44.1kHz stereo WAV resampled down to 16kHz mono float32
    t = np.linspace(0, 0.25, int(44100 * 0.25), endpoint=False, dtype=np.float32)
    mono = 0.25 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([mono, mono], axis=1).reshape(-1)
    wav_bytes = _make_wav_bytes(stereo, rate=44100, channels=2)
    decoded = _decode_audio_to_float32_16k(wav_bytes, "audio/wav")
    assert decoded.dtype == np.float32
    assert abs(decoded.size - 4000) <= 5

    # Raw 16kHz int16 mono PCM
    raw_pcm = (np.zeros(3200, dtype=np.float32)).astype("<i2").tobytes()
    decoded_raw = _decode_audio_to_float32_16k(raw_pcm, "application/octet-stream")
    assert decoded_raw.size == 3200


def test_practice_full_lifecycle_privacy_and_asr_routes(tmp_path: Path, monkeypatch):
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    # Force offline deterministic fallback mode so tests never touch external LLMs
    runtime.llm_config.provider = "none"
    runtime.llm_config.api_key = ""

    # Stub ASR transcribe_pcm so we test the HTTP decode + route path without loading 600MB MLX weights
    def _fake_transcribe(pcm: np.ndarray):
        return SimpleNamespace(
            text="I led the migration and reduced latency by 40 percent.",
            inference_ms=12.4,
            audio_duration_ms=(pcm.size / 16000.0) * 1000.0,
        )

    monkeypatch.setattr(runtime.asr, "transcribe_pcm", _fake_transcribe)

    # Spy on generate_practice_plan to guarantee raw resume PII NEVER reaches LLM
    seen_resume_args: list[str] = []
    orig_plan = runtime.llm.generate_practice_plan

    async def _spy_plan(**kwargs):
        seen_resume_args.append(kwargs.get("redacted_resume", ""))
        return await orig_plan(**kwargs)

    monkeypatch.setattr(runtime.llm, "generate_practice_plan", _spy_plan)

    client = TestClient(app)

    # 1. Upload synthetic resume with PII & verify /api/status includes tts
    status = client.get("/api/status").json()
    assert "tts" in status
    assert "engine" in status["tts"]

    r_res = client.post(
        "/api/profile/resume/text",
        json={
            "filename": "synthetic_resume.txt",
            "text": (
                "Jordan Synthetic\n"
                "jordan.synthetic@example.org | (415) 555-0199 | San Francisco, CA\n\n"
                "Senior Software Engineer — Built distributed event pipelines handling 50k QPS."
            ),
        },
    )
    assert r_res.status_code == 200

    # 2. Local ASR transcribe route (raw PCM + WAV + empty rejection)
    raw_audio = np.zeros(4000, dtype="<i2").tobytes()
    asr_res = client.post(
        "/api/asr/transcribe",
        content=raw_audio,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert asr_res.status_code == 200
    assert "reduced latency by 40 percent" in asr_res.json()["text"]

    empty_asr = client.post("/api/asr/transcribe", content=b"")
    assert empty_asr.status_code == 400

    # 3. Module gating: disabling practice_mode blocks /api/practice/start with 403
    client.post("/api/addons/modules", json={"modules": {"practice_mode": False}})
    blocked = client.post("/api/practice/start", json={"interview_type": "behavioral"})
    assert blocked.status_code == 403
    client.post("/api/addons/modules", json={"modules": {"practice_mode": True}})

    # 4. Start practice session
    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Senior Software Engineer, Full-stack",
            "company": "Anthropic",
            "interview_type": "behavioral",
            "question_source": "mixed",
            "target_level": "L5",
            "question_count": 2,
        },
    )
    assert start_res.status_code == 200
    start_data = start_res.json()
    sess = start_data["session"]
    sid = sess["id"]
    assert len(sess["questions"]) == 2
    assert len(sess["turns"]) == 1
    assert sess["turns"][0]["role"] == "interviewer"

    # Strict privacy check: raw email/phone must NEVER appear in resume passed to LLM
    assert seen_resume_args
    assert "jordan.synthetic@example.org" not in seen_resume_args[-1]
    assert "555-0199" not in seen_resume_args[-1]
    assert "distributed event pipelines" in seen_resume_args[-1]

    # 5. Submit a candidate turn
    turn_res = client.post(
        f"/api/practice/{sid}/turn",
        json={
            "text": (
                "When our team faced a major database outage during a product launch, "
                "I designed a read-replica failover queue and led the rollout. "
                "We reduced p99 latency by 60% for 500k users, and I learned to automate chaos drills."
            )
        },
    )
    assert turn_res.status_code == 200
    turn_data = turn_res.json()
    assert turn_data["interviewer_turn"]["role"] == "interviewer"
    q0 = turn_data["session"]["questions"][0]
    assert q0["carl_signals"]["result"] == "strong"
    assert q0["carl_signals"]["learning"] == "strong"

    # 6. Request a supportive hint
    hint_res = client.post(f"/api/practice/{sid}/hint", json={})
    assert hint_res.status_code == 200
    assert hint_res.json()["coach_turn"]["role"] == "coach"

    # 7. Workspace context update + size cap enforcement
    ws_ok = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_summary": "Client -> API -> Postgres"}},
    )
    assert ws_ok.status_code == 200
    ws_huge = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_summary": "x" * 250_000}},
    )
    assert ws_huge.status_code == 400

    # 8. Advance to Q2
    next_res = client.post(f"/api/practice/{sid}/next")
    assert next_res.status_code == 200
    assert next_res.json()["has_more"] is True
    assert next_res.json()["session"]["current_question_index"] == 1

    # 9. Generate Supportive CARL Debrief
    debrief_res = client.post(f"/api/practice/{sid}/debrief")
    assert debrief_res.status_code == 200
    debrief = debrief_res.json()["debrief"]
    assert debrief["overall_encouragement"]
    assert debrief["top_strengths"]
    assert debrief["retry_drill"]["prompt"]
    assert debrief_res.json()["session"]["phase"] == "completed"

    # Completed session rejects new regular turns with 409
    late_turn = client.post(f"/api/practice/{sid}/turn", json={"text": "One more thing"})
    assert late_turn.status_code == 409

    # 10. Submit 45-second Retry Weak Part drill
    retry_res = client.post(
        f"/api/practice/{sid}/retry",
        json={
            "answer_text": (
                "As a result of my failover design, we cut p99 latency by 60% and saved 12 engineer-hours per week. "
                "The big learning I carried forward was baking fault injection into CI."
            )
        },
    )
    assert retry_res.status_code == 200
    fb = retry_res.json()["feedback"]
    assert fb["encouragement"]
    assert len(retry_res.json()["session"]["retry_history"]) == 1

    # 11. Export Markdown coaching report & list sessions
    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert "Retry Drill Progress" in md_res.text
    assert f'filename="{sid}_practice.md"' in md_res.headers["content-disposition"]

    list_res = client.get("/api/practice/sessions")
    assert list_res.status_code == 200
    assert any(s["id"] == sid and s["has_debrief"] for s in list_res.json()["sessions"])
    assert list_res.headers.get("cache-control") == "no-store"


def test_codex_m2_regressions_state_machine_privacy_and_audio(tmp_path: Path, monkeypatch):
    # 1. Punctuation-adjacent percent metric scores strong Result
    sig = estimate_carl_heuristic("I led the migration and cut latency by 45%.")
    assert sig["result"] == "strong"

    # 2. Truncated WAV raises ValueError (HTTP 400, never 500)
    full_wav = _make_wav_bytes(np.zeros(4000, dtype=np.float32), rate=16000, channels=1)
    truncated_wav = full_wav[:80]
    with pytest.raises(ValueError):
        _decode_audio_to_float32_16k(truncated_wav, "audio/wav")

    # 3. PracticeSessionStore returns deep-copy snapshots (uncommitted mutation isolated)
    store = PracticeSessionStore(tmp_path / "practice_iso")
    s0 = PracticeSession(role_title="Original Role")
    store.save(s0)
    snap = store.get(s0.id)
    assert snap is not None
    snap.role_title = "Mutated Unsaved"
    snap2 = store.get(s0.id)
    assert snap2 is not None
    assert snap2.role_title == "Original Role"

    # 4. API state machine invariants, Cache-Control: no-store, expected_question_id, idempotent debrief
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "none"
    runtime.llm_config.api_key = ""
    client = TestClient(app)

    # Truncated WAV via HTTP -> 400 + no-store
    bad_wav_res = client.post(
        "/api/asr/transcribe",
        content=truncated_wav,
        headers={"Content-Type": "audio/wav"},
    )
    assert bad_wav_res.status_code == 400
    assert bad_wav_res.headers.get("cache-control") == "no-store"

    # Profile also sends Cache-Control: no-store
    prof_res = client.get("/api/profile")
    assert prof_res.status_code == 200
    assert prof_res.headers.get("cache-control") == "no-store"

    # Pick an explicit bank question in mixed mode -> must be pinned at index 0
    bank_qs = runtime.banks.list_questions(category="behavioral")
    assert bank_qs
    chosen_id = bank_qs[-1]["id"]
    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Staff SWE",
            "interview_type": "behavioral",
            "question_source": "mixed",
            "question_count": 2,
            "selected_question_id": chosen_id,
        },
    )
    assert start_res.status_code == 200
    sess = start_res.json()["session"]
    sid = sess["id"]
    assert sess["questions"][0]["bank_question_id"] == chosen_id
    q0_id = sess["questions"][0]["id"]

    # Retry rejected on active session (409)
    early_retry = client.post(f"/api/practice/{sid}/retry", json={"answer_text": "Too early"})
    assert early_retry.status_code == 409

    # expected_question_id mismatch rejected with 409 on /turn and /next
    stale_turn = client.post(
        f"/api/practice/{sid}/turn",
        json={"text": "My answer", "expected_question_id": "wrong_qid"},
    )
    assert stale_turn.status_code == 409
    stale_next = client.post(
        f"/api/practice/{sid}/next",
        json={"expected_question_id": "wrong_qid"},
    )
    assert stale_next.status_code == 409

    # Advance through Q1 -> Q2 -> awaiting_debrief
    n1 = client.post(f"/api/practice/{sid}/next", json={"expected_question_id": q0_id})
    assert n1.status_code == 200
    assert n1.json()["has_more"] is True
    q1_id = n1.json()["session"]["questions"][1]["id"]

    n2 = client.post(f"/api/practice/{sid}/next", json={"expected_question_id": q1_id})
    assert n2.status_code == 200
    assert n2.json()["has_more"] is False
    assert n2.json()["session"]["phase"] == "awaiting_debrief"

    # Extra turn and hint rejected with 409 in awaiting_debrief
    assert client.post(f"/api/practice/{sid}/turn", json={"text": "Late turn"}).status_code == 409
    assert client.post(f"/api/practice/{sid}/hint", json={}).status_code == 409

    # Debrief runs once and is idempotent on repeat calls
    debrief_calls = 0
    orig_debrief = runtime.llm.generate_practice_debrief

    async def _counting_debrief(**kwargs):
        nonlocal debrief_calls
        debrief_calls += 1
        return await orig_debrief(**kwargs)

    monkeypatch.setattr(runtime.llm, "generate_practice_debrief", _counting_debrief)
    d1 = client.post(f"/api/practice/{sid}/debrief")
    d2 = client.post(f"/api/practice/{sid}/debrief")
    assert d1.status_code == 200
    assert d2.status_code == 200
    assert debrief_calls == 1
    assert d1.json()["debrief"] == d2.json()["debrief"]

    # Workspace update rejected after completed (409)
    ws_late = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_summary": "Late edit"}},
    )
    assert ws_late.status_code == 409

    # Retry with unknown question_id rejected with 400
    bad_retry = client.post(
        f"/api/practice/{sid}/retry",
        json={"answer_text": "My retry", "question_id": "nonexistent_q"},
    )
    assert bad_retry.status_code == 400

    # Malformed debrief non-dict elements do not crash to_markdown()
    loaded = runtime.practice_store.get(sid)
    assert loaded is not None
    loaded.debrief = {
        "overall_encouragement": "Nice job!",
        "top_strengths": ["Clear structure"],
        "question_reviews": [None, "oops"],
        "retry_drill": "bad",
    }
    md = loaded.to_markdown()
    assert "Nice job!" in md


def test_onboarding_voice_wizard_contracts_and_addon_tts_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app = create_app(project_root=tmp_path)
    runtime = app.state.runtime

    installed_flag = {"on": False}

    def _fake_is_installed() -> bool:
        return installed_flag["on"]

    def _fake_install() -> bool:
        installed_flag["on"] = True
        return True

    def _fake_uninstall() -> bool:
        installed_flag["on"] = False
        return True

    monkeypatch.setattr(runtime.tts, "is_kokoro_installed", _fake_is_installed)
    monkeypatch.setattr(runtime.tts, "install_kokoro", _fake_install)
    monkeypatch.setattr(runtime.tts, "uninstall_kokoro", _fake_uninstall)
    runtime.addons.register_pack_handler(
        "kokoro_tts",
        is_installed=_fake_is_installed,
        install=_fake_install,
        uninstall=_fake_uninstall,
        available=True,
    )

    with TestClient(app) as client:
        r_inst = client.post("/api/addons/kokoro_tts/install")
        assert r_inst.status_code == 200
        body = r_inst.json()
        assert body["tts"]["engine"] == "kokoro"
        assert body["tts"]["kokoro_installed"] is True
        assert "af_heart" in body["tts"]["voices"]

        r_del = client.delete("/api/addons/kokoro_tts")
        assert r_del.status_code == 200
        del_body = r_del.json()
        assert del_body["tts"]["kokoro_installed"] is False
        assert del_body["tts"]["engine"] != "kokoro"

    static_dir = Path(__file__).resolve().parents[1] / "src" / "mockery" / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    practice_js = (static_dir / "practice_studio.js").read_text(encoding="utf-8")

    for required_id in (
        "onboardStepAi",
        "onboardStepPacks",
        "onboardKokoroInstallBtn",
        "onboardKokoroPreviewBtn",
        "onboardOptionalPacksList",
        "practiceKokoroUpgradeRow",
        "practiceInstallKokoroBtn",
        "practiceVoicePickerRow",
        "practiceVoiceSelect",
        "practicePreviewVoiceBtn",
    ):
        assert f'id="{required_id}"' in html, f"Missing DOM element #{required_id} in index.html"

    assert "setOnboardStep" in app_js and "renderOnboardingPacksStep" in app_js
    assert "mockery_onboard_packs_seen" in app_js
    assert "renderVoiceControls" in practice_js and "previewVoice" in practice_js
    assert "mockery_tts_voice" in practice_js

