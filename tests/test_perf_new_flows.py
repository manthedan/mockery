from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mockery.addons import _hf_repo_cached, _invalidate_hf_repo_cache
from mockery.banks import QuestionBankStore
from mockery.practice import (
    PracticeConfig,
    PracticeQuestionItem,
    PracticeSession,
    PracticeSessionStore,
    PracticeTurn,
)
from mockery.redactor import DetectedSpan, RAMPART_FILES, RAMPART_REPO_ID, RampartRedactor
from mockery.server import _decode_audio_to_float32_16k, _lowpass_antialias, create_app
from mockery.tts import LocalTTSEngine, _split_speech_sentences


def _make_wav_bytes(samples: np.ndarray, *, rate: int, channels: int) -> bytes:
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _reference_decode_wav_16k(raw_bytes: bytes) -> np.ndarray:
    """Original baseline WAV decode + full-rate lowpass + np.interp for numerical parity checks."""
    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    if rate != 16000 and pcm.size > 1:
        if rate > 16000:
            pcm = _lowpass_antialias(pcm, rate, 16000)
        target_len = max(1, int(round(pcm.size * (16000.0 / float(rate)))))
        x_old = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return np.ascontiguousarray(pcm, dtype=np.float32)


def test_split_speech_sentences_edge_cases() -> None:
    assert _split_speech_sentences("   ") == []
    # Short intro merges with the next sentence to preserve prosody
    merged = _split_speech_sentences("Great point! Tell me how you partitioned the hot key space under peak load.")
    assert len(merged) == 1
    assert merged[0].startswith("Great point!")

    # Multi-sentence prompt splits into natural streaming chunks
    multi = _split_speech_sentences(
        "Welcome to your system design interview today. "
        "Start by clarifying the functional requirements and latency targets. "
        "Then sketch out the high-level architecture on the whiteboard."
    )
    assert len(multi) == 3

    # Very long single sentence splits cleanly at clause boundaries
    long_clause = (
        "When you migrated the distributed ledger across three regions, "
        "which required coordinating dual writes and backfilling five billion historical rows, "
        "how did you validate consistency without degrading checkout tail latency for merchants?"
    )
    parts = _split_speech_sentences(long_clause, max_chunk_chars=120)
    assert len(parts) >= 2
    assert all(len(p) <= 180 for p in parts)


def test_decode_audio_integer_fir_and_non_integer_parity() -> None:
    rng = np.random.default_rng(42)

    # 1. 48kHz stereo (exact integer factor D=3)
    n48 = 48000 * 2
    t48 = np.linspace(0.0, 2.0, n48, endpoint=False, dtype=np.float32)
    left = 0.35 * np.sin(2.0 * np.pi * 440.0 * t48) + 0.1 * rng.standard_normal(n48).astype(np.float32)
    right = 0.25 * np.cos(2.0 * np.pi * 880.0 * t48) + 0.1 * rng.standard_normal(n48).astype(np.float32)
    stereo48 = np.stack([left, right], axis=1).reshape(-1)
    wav48 = _make_wav_bytes(stereo48, rate=48000, channels=2)
    got48 = _decode_audio_to_float32_16k(wav48, "audio/wav")
    ref48 = _reference_decode_wav_16k(wav48)
    assert got48.shape == ref48.shape == (32000,)
    assert float(np.max(np.abs(got48 - ref48))) < 1e-5

    # 2. 32kHz mono (exact integer factor D=2)
    n32 = 32000
    mono32 = 0.4 * np.sin(np.linspace(0.0, 400.0, n32, dtype=np.float32))
    wav32 = _make_wav_bytes(mono32, rate=32000, channels=1)
    got32 = _decode_audio_to_float32_16k(wav32, "audio/wav")
    ref32 = _reference_decode_wav_16k(wav32)
    assert got32.shape == ref32.shape == (16000,)
    assert float(np.max(np.abs(got32 - ref32))) < 1e-5

    # 3. 44.1kHz stereo (non-integer ratio)
    n44 = 44100
    t44 = np.linspace(0.0, 1.0, n44, endpoint=False, dtype=np.float32)
    s44 = np.stack([0.3 * np.sin(2.0 * np.pi * 523.25 * t44), 0.2 * np.sin(2.0 * np.pi * 659.25 * t44)], axis=1).reshape(-1)
    wav44 = _make_wav_bytes(s44, rate=44100, channels=2)
    got44 = _decode_audio_to_float32_16k(wav44, "audio/wav")
    ref44 = _reference_decode_wav_16k(wav44)
    assert got44.shape == ref44.shape
    assert float(np.max(np.abs(got44 - ref44))) < 1e-5

    # 4. 48kHz mono with non-divisible length (48001 frames -> falls back to general path cleanly)
    mono48_odd = 0.3 * np.sin(np.linspace(0.0, 200.0, 48001, dtype=np.float32))
    wav48_odd = _make_wav_bytes(mono48_odd, rate=48000, channels=1)
    got_odd = _decode_audio_to_float32_16k(wav48_odd, "audio/wav")
    ref_odd = _reference_decode_wav_16k(wav48_odd)
    assert got_odd.shape == ref_odd.shape
    assert float(np.max(np.abs(got_odd - ref_odd))) < 1e-5


def test_tts_streaming_caching_cancellation_and_http_route(tmp_path: Path) -> None:
    tts = LocalTTSEngine(tmp_path / "models")

    # Stub Kokoro backend with call counter
    kokoro_calls: list[str] = []

    class _FakeKokoro:
        def create(self, text: str, voice: str = "af_heart", speed: float = 1.0, lang: str = "en-us"):
            kokoro_calls.append(text)
            samples = np.full(1200, 0.25, dtype=np.float32)
            return samples, 24000

    tts._kokoro = _FakeKokoro()
    tts._ensure_kokoro_loaded = lambda: True  # type: ignore[method-assign]
    tts.is_kokoro_installed = lambda: True  # type: ignore[method-assign]

    prompt = (
        "Welcome to your behavioral practice session today. "
        "Tell me about a time you resolved a high-stakes architecture disagreement."
    )

    async def _run_kokoro_checks() -> None:
        eng, sr, gen = await tts.open_pcm_stream(prompt, voice="kokoro:af_heart", engine="kokoro")
        assert eng == "kokoro"
        assert sr == 24000
        chunks = [ch async for ch in gen]
        assert len(chunks) == 3  # sentence 1 + 140ms silence gap + sentence 2
        assert chunks[1] == b"\x00" * (int(24000 * 0.14) * 2)
        assert len(kokoro_calls) == 2

        # Second call hits bounded PCM16 LRU cache -> 0 new synth calls, identical PCM bytes
        eng2, sr2, gen2 = await tts.open_pcm_stream(prompt, voice="kokoro:af_heart", engine="kokoro")
        assert eng2 == "kokoro" and sr2 == 24000
        cached_chunks = [ch async for ch in gen2]
        assert len(kokoro_calls) == 2
        assert b"".join(cached_chunks) == b"".join(chunks)

        # Early cancellation (barge-in) stops before synthesizing sentence #2 and does not poison cache
        kokoro_calls.clear()
        tts._cache_clear_engine("kokoro")
        _e3, _sr3, gen3 = await tts.open_pcm_stream(prompt, voice="kokoro:af_heart", engine="kokoro")
        first = await anext(gen3)
        assert len(first) > 0
        await gen3.aclose()
        assert len(kokoro_calls) == 1
        assert len(tts._pcm_cache) == 0

    asyncio.run(_run_kokoro_checks())

    # Stub Phonon native stream backend + early barge-in cancel
    phonon_yielded = 0

    class _FakePhonon:
        sample_rate = 24000

        def stream(self, text: str, voice=None):
            nonlocal phonon_yielded
            for _ in range(5):
                phonon_yielded += 1
                yield np.full(480, 0.1, dtype=np.float32)

    tts._phonon = _FakePhonon()
    tts._ensure_phonon_loaded = lambda: True  # type: ignore[method-assign]
    tts.is_phonon_installed = lambda: True  # type: ignore[method-assign]

    async def _run_phonon_checks() -> None:
        nonlocal phonon_yielded
        eng, sr, gen = await tts.open_pcm_stream("Stream me smoothly.", voice="phonon:default", engine="phonon")
        assert eng == "phonon" and sr == 24000
        chunks = [ch async for ch in gen]
        assert len(chunks) == 5
        assert sum(len(c) for c in chunks) == 5 * 480 * 2
        assert phonon_yielded == 5

        # Cache hit on repeat
        eng2, _sr2, gen2 = await tts.open_pcm_stream("Stream me smoothly.", voice="phonon:default", engine="phonon")
        assert eng2 == "phonon"
        assert b"".join([c async for c in gen2]) == b"".join(chunks)
        assert phonon_yielded == 5

    asyncio.run(_run_phonon_checks())

    # Exercise FastAPI POST /api/tts/stream + semaphore cleanup on success and error
    app = create_app(project_root=tmp_path, preload_asr=False)
    app.state.runtime.tts = tts
    with TestClient(app) as client:
        r_ok = client.post("/api/tts/stream", json={"text": "Stream me smoothly.", "engine": "phonon"})
        assert r_ok.status_code == 200
        assert r_ok.headers["x-tts-engine"] == "phonon"
        assert r_ok.headers["x-tts-sample-rate"] == "24000"
        assert r_ok.headers["x-tts-channels"] == "1"
        assert r_ok.headers["x-tts-sample-format"] == "pcm_s16le"
        assert r_ok.headers["cache-control"] == "no-store"
        assert len(r_ok.content) == 5 * 480 * 2

        # Empty text -> 400 and semaphore remains unlocked
        r_empty = client.post("/api/tts/stream", json={"text": "   "})
        assert r_empty.status_code == 400

        # Oversized body -> 413 from middleware
        r_big = client.post(
            "/api/tts/stream",
            content=json.dumps({"text": "x" * 40000}),
            headers={"Content-Type": "application/json"},
        )
        assert r_big.status_code == 413

        # Semaphore must still be free (no leak -> 200, never 429)
        r_ok2 = client.post("/api/tts/stream", json={"text": "Stream me smoothly.", "engine": "phonon"})
        assert r_ok2.status_code == 200


def test_rampart_span_cache_parity_and_fast_snapshot_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    redactor = RampartRedactor()
    assert redactor.is_downloaded() is False

    # Build a synthetic HF snapshot directory structure
    repo_dir = tmp_path / "hub" / f"models--{RAMPART_REPO_ID.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / "abc123"
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    for rel in RAMPART_FILES:
        fp = snap_dir / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"stub")

    # Fast snapshot resolver finds all 3 files without invoking hf_hub_download
    paths = redactor._local_files_paths()
    assert paths is not None
    assert set(paths.keys()) == set(RAMPART_FILES)

    # Stub loaded ONNX session + tokenizer to count inference runs & verify span cache immutability
    run_count = 0
    sample_resume = (
        "Jordan Rivera\n"
        "jordan@example.org | (415) 555-0192 | San Francisco, CA\n"
        "Stripe — Staff Software Engineer | Jan 2022 – Present\n"
        "Stanford University — M.S. Computer Science\n"
    )
    name_end = len("Jordan Rivera")
    city_start = sample_resume.index("San Francisco")
    city_end = city_start + len("San Francisco")

    class _FakeEncoding:
        def __init__(self) -> None:
            self.ids = [101, 200, 201, 202, 102]
            self.attention_mask = [1, 1, 1, 1, 1]
            self.type_ids = [0, 0, 0, 0, 0]
            self.offsets = [(0, 0), (0, 6), (7, name_end), (city_start, city_end), (0, 0)]
            self.overflowing = []

    class _FakeTokenizer:
        def encode(self, _text: str):
            return _FakeEncoding()

    class _FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, _out, _feeds):
            nonlocal run_count
            run_count += 1
            # Labels: 0=O, 1=B-GIVEN_NAME, 2=B-SURNAME, 3=B-CITY
            logits = np.zeros((1, 5, 4), dtype=np.float32)
            logits[0, 0, 0] = 10.0
            logits[0, 1, 1] = 10.0
            logits[0, 2, 2] = 10.0
            logits[0, 3, 3] = 10.0
            logits[0, 4, 0] = 10.0
            return [logits]

    redactor._session = _FakeSession()
    redactor._input_names = {"input_ids", "attention_mask"}
    redactor._tokenizer = _FakeTokenizer()
    redactor._id2label = {0: "O", 1: "B-GIVEN_NAME", 2: "B-SURNAME", 3: "B-CITY"}

    # 1st call runs ONNX once
    r1 = redactor.redact(sample_resume, keep_location=True, hide_employers_and_schools=False)
    assert run_count == 1
    assert "[PERSON_NAME_1]" in r1.redacted_text
    assert "San Francisco" in r1.redacted_text

    # Toggling prefs & unredacted_values hits the span cache (0 extra ONNX runs!)
    r2 = redactor.redact(sample_resume, keep_location=False, hide_employers_and_schools=True)
    assert run_count == 1
    assert "[CITY_1]" in r2.redacted_text
    assert "[COMPANY_1]" in r2.redacted_text
    assert "[SCHOOL_1]" in r2.redacted_text

    r3 = redactor.redact(sample_resume, keep_location=True, unredacted_values=["Jordan Rivera"])
    assert run_count == 1
    assert "Jordan Rivera" in r3.redacted_text

    # Re-running exact r1 options produces bit-exact output (cached spans were never mutated!)
    r1_again = redactor.redact(sample_resume, keep_location=True, hide_employers_and_schools=False)
    assert run_count == 1
    assert r1_again.redacted_text == r1.redacted_text
    assert r1_again.spans == r1.spans
    assert r1_again.entity_counts == r1.entity_counts

    # delete_cache clears span cache and cached paths
    redactor.delete_cache()
    assert len(redactor._span_cache) == 0
    assert redactor._cached_paths is None


def test_addon_fast_hf_repo_cache_and_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    _invalidate_hf_repo_cache()

    repo_id = "mlx-community/parakeet-tdt-0.6b-v2"
    assert _hf_repo_cached(repo_id) is False

    # Create snapshot with a non-empty file
    snap = tmp_path / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "rev1"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "weights.safetensors").write_bytes(b"1234")

    # Within TTL cache still returns False until invalidated
    assert _hf_repo_cached(repo_id) is False
    _invalidate_hf_repo_cache(repo_id)
    assert _hf_repo_cached(repo_id) is True

    # Removing files + invalidating returns False
    (snap / "weights.safetensors").unlink()
    _invalidate_hf_repo_cache(repo_id)
    assert _hf_repo_cached(repo_id) is False


def test_practice_session_store_summary_cache_parity_and_invalidation(tmp_path: Path) -> None:
    store = PracticeSessionStore(tmp_path / "practice")
    sessions = []
    for i in range(5):
        s = PracticeSession(
            id=f"prac_perf_{i:02d}",
            role_title=f"Engineer {i}",
            company="Acme",
            config=PracticeConfig(interview_type="behavioral", target_level="L5"),
            questions=[PracticeQuestionItem(title=f"Q{i}", prompt="Prompt text")],
            turns=[PracticeTurn(role="interviewer", text="Hello")],
        )
        store.save(s)
        sessions.append(s)

    # Compare warm cached store vs brand-new cold store
    cold_store = PracticeSessionStore(tmp_path / "practice")
    warm_list = store.list_recent(limit=10)
    cold_list = cold_store.list_recent(limit=10)
    assert warm_list == cold_list
    assert len(warm_list) == 5

    # Mutating caller's returned dict must not corrupt cache
    warm_list[0]["role_title"] = "MUTATED"
    assert store.list_recent(limit=10)[0]["role_title"] != "MUTATED"

    # External on-disk modification invalidates cache entry via (mtime_ns, size)
    target_file = tmp_path / "practice" / "prac_perf_00.json"
    raw = json.loads(target_file.read_text(encoding="utf-8"))
    raw["role_title"] = "Updated Staff Engineer"
    raw["debrief"] = {"overall_encouragement": "Great work!"}
    target_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    # Bump mtime explicitly so even same-second writes have distinct mtime_ns
    st = target_file.stat()
    os.utime(target_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))

    refreshed = store.list_recent(limit=10)
    by_id = {item["id"]: item for item in refreshed}
    assert by_id["prac_perf_00"]["role_title"] == "Updated Staff Engineer"
    assert by_id["prac_perf_00"]["has_debrief"] is True

    # Deleting a file prunes it from results and summary cache
    (tmp_path / "practice" / "prac_perf_01.json").unlink()
    after_del = store.list_recent(limit=10)
    assert len(after_del) == 4
    assert "prac_perf_01.json" not in store._summary_cache


def test_practice_history_score_matches_debrief_after_retry(tmp_path: Path) -> None:
    """The Home rail must show the same headline score as the Debrief, retries included."""

    def carl(context: str, action: str, result: str, learning: str) -> dict[str, str]:
        return {"context": context, "action": action, "result": result, "learning": learning}

    store = PracticeSessionStore(tmp_path / "practice")
    session = PracticeSession(
        id="prac_retry_score",
        phase="completed",
        questions=[
            PracticeQuestionItem(
                id="q_strong", title="Strong", prompt="p", status="completed",
                carl_signals=carl("strong", "strong", "strong", "strong"),
            ),
            PracticeQuestionItem(
                id="q_weak", title="Weak", prompt="p", status="completed",
                carl_signals=carl("strong", "strong", "missing", "partial"),
            ),
        ],
        debrief={"retry_drill": {"question_id": "q_weak", "prompt": "Retry the result."}},
    )
    store.save(session)
    # Per-signal average across both answers: result lands on "partial" → 3.5 / 4.
    assert store.list_recent()[0]["score"] == 3.5

    # An unscored attempt changes nothing; the latest scored retry replaces the drilled answer.
    session.retry_history = [
        {"candidate_answer": "first", "feedback": {"carl_after": carl("strong", "strong", "partial", "strong")}},
        {"candidate_answer": "second", "feedback": {"carl_after": carl("strong", "strong", "strong", "strong")}},
        {"candidate_answer": "third", "feedback": {}},
    ]
    store.save(session)
    warm = store.list_recent()
    cold = PracticeSessionStore(tmp_path / "practice").list_recent()
    assert warm == cold
    assert warm[0]["score"] == 4.0

    # Without an explicit drill target, the retry stands in for the weakest answer.
    session.debrief = {"overall_encouragement": "Nice."}
    store.save(session)
    assert store.list_recent()[0]["score"] == 4.0


def test_question_bank_cache_jd_score_parity_and_invalidation(tmp_path: Path) -> None:
    user_dir = tmp_path / "banks"
    user_dir.mkdir(parents=True, exist_ok=True)
    custom_pack = user_dir / "custom.json"
    pack_data = {
        "pack_id": "custom_perf",
        "title": "Custom Perf Pack",
        "category": "system_design",
        "version": "1.0",
        "questions": [
            {
                "id": "cust_q1",
                "title": "Redis Distributed Cache Cache",
                "competency": "caching",
                "levels": ["L5", "L6"],
                "jd_tags": ["redis", "distributed systems"],
                "prompt": "Design a distributed redis cache with consistent hashing and redis replication.",
            }
        ],
    }
    custom_pack.write_text(json.dumps(pack_data), encoding="utf-8")

    jd_text = "Staff Engineer working on distributed systems, Redis caching, consistent hashing, and Kubernetes."
    store = QuestionBankStore(user_banks_dir=user_dir)

    # First call (cold) and second call (warm) must match naive reference scoring bit-for-bit
    qs_cold = store.list_questions(category="system_design", level="L5", jd_text=jd_text)
    qs_warm = store.list_questions(category="system_design", level="L5", jd_text=jd_text)
    assert qs_cold == qs_warm

    # Verify exact JD scoring formula including repeated-token multiplicity ("redis" x2, "cache" x2, etc.)
    jd_tokens = set(re.findall(r"[a-z0-9][a-z0-9+#.\-]{1,24}", jd_text.lower()))
    cust_item = next(q for q in qs_warm if q["id"] == "cust_q1")
    expected_score = 0
    for tag in pack_data["questions"][0]["jd_tags"]:
        t = tag.lower()
        if t in jd_text.lower() or t in jd_tokens:
            expected_score += 3
    blob = f"{pack_data['questions'][0]['title']} {pack_data['questions'][0]['competency']} {pack_data['questions'][0]['prompt']}".lower()
    for tok in re.findall(r"[a-z0-9]{4,20}", blob):
        if tok in jd_tokens:
            expected_score += 1
    assert cust_item["jd_relevance_score"] == expected_score
    assert store.get_question("cust_q1")["title"] == "Redis Distributed Cache Cache"

    # Editing pack on disk invalidates cache automatically
    pack_data["questions"][0]["title"] = "Updated Distributed Queue"
    custom_pack.write_text(json.dumps(pack_data), encoding="utf-8")
    st = custom_pack.stat()
    os.utime(custom_pack, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
    assert store.get_question("cust_q1")["title"] == "Updated Distributed Queue"
