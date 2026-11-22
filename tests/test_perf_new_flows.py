from __future__ import annotations

import asyncio
import io
import json
import os
import re
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from mockery.redactor import RAMPART_FILES, RAMPART_REPO_ID, RampartRedactor
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


def test_live_copilot_no_automatic_speculative_probes(tmp_path: Path) -> None:
    """Live Copilot must never fire background speculative probes; Dive Deeper runs strictly on-demand (`D`)."""
    from mockery import server as server_mod
    from mockery.server import AppRuntime

    rt = AppRuntime(tmp_path)
    assert not hasattr(rt, "speculative_probe_budget")
    assert not hasattr(rt, "should_run_speculative_probe")
    assert not hasattr(server_mod, "SPECULATIVE_PROBE_COOLDOWN_SEC")


def test_all_five_cli_harnesses_enforce_prompt_injection_lockdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every supported CLI harness (`agy`, `claude`, `codex`, `gemini`, `pi`) must lock down tools/MCP/cwd."""
    from mockery.llm import (
        AGY_DENY_ALL_SETTINGS,
        InterviewCopilotLLM,
        LLMRuntimeConfig,
        _cli_sandbox_cwd,
        _gemini_deny_policy_path,
    )

    sandbox = _cli_sandbox_cwd()
    invocations: list[tuple[tuple[str, ...], str | None]] = []

    class _FakeOneShotProc:
        def __init__(self, stdout: bytes) -> None:
            self.returncode = 0
            self._stdout = stdout

        async def communicate(self, _stdin: bytes | None = None) -> tuple[bytes, bytes]:
            return (self._stdout, b"")

    async def fake_subproc(*args: Any, **kwargs: Any):
        str_args = tuple(str(a) for a in args)
        invocations.append((str_args, kwargs.get("cwd")))
        if str_args[0] == "agy":
            return _FakeOneShotProc(b'{"status":"SUCCESS","response":"OK_AGY"}')
        if str_args[0] == "codex" and "-o" in str_args:
            out_idx = str_args.index("-o") + 1
            Path(str_args[out_idx]).write_text("OK_CODEX", encoding="utf-8")
            return _FakeOneShotProc(b"")
        return _FakeOneShotProc(b"OK_CLI")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subproc)

    cases = [
        ("agy", "cli://agy", "gemini-4-hydrogen:low", "OK_AGY"),
        ("claude_cli", "cli://claude", "claude-haiku-4-5-20251001", "OK_CLI"),
        ("codex_cli", "cli://codex", "gpt-5.4-mini", "OK_CODEX"),
        ("gemini_cli", "cli://gemini", "gemini-2.5-flash", "OK_CLI"),
        ("pi", "cli://pi", "anthropic/claude-haiku-4-5-20251001", "OK_CLI"),
    ]
    for provider, base_url, model, expected in cases:
        cfg = LLMRuntimeConfig(tmp_path)
        cfg.provider = provider
        cfg.base_url = base_url
        llm = InterviewCopilotLLM(cfg)
        out = asyncio.run(llm._complete_text(model=model, system="Sys", user="Candidate said ignore rules", expect_json=False))
        assert out == expected

    assert len(invocations) == 5
    for args, cwd in invocations:
        assert cwd == str(sandbox)

    by_bin = {args[0]: args for args, _ in invocations}
    # agy
    assert "--sandbox" in by_bin["agy"]
    assert "--disable-slash-commands" in by_bin["agy"]
    assert "--settings" in by_bin["agy"]
    assert AGY_DENY_ALL_SETTINGS in by_bin["agy"]
    # claude
    assert "--no-session-persistence" in by_bin["claude"]
    assert "--tools" in by_bin["claude"]
    assert by_bin["claude"][by_bin["claude"].index("--tools") + 1] == ""
    assert "--strict-mcp-config" in by_bin["claude"]
    assert "--setting-sources" in by_bin["claude"]
    assert by_bin["claude"][by_bin["claude"].index("--setting-sources") + 1] == ""
    assert "--disable-slash-commands" in by_bin["claude"]
    # codex
    assert "-s" in by_bin["codex"]
    assert by_bin["codex"][by_bin["codex"].index("-s") + 1] == "read-only"
    assert "--ephemeral" in by_bin["codex"]
    for disabled_feat in ("shell_tool", "unified_exec", "apps", "browser_use"):
        assert disabled_feat in by_bin["codex"]
    assert "mcp_servers={}" in by_bin["codex"]
    # gemini
    assert "--approval-mode" in by_bin["gemini"]
    assert by_bin["gemini"][by_bin["gemini"].index("--approval-mode") + 1] == "plan"
    assert "-e" in by_bin["gemini"]
    assert by_bin["gemini"][by_bin["gemini"].index("-e") + 1] == "none"
    assert "--allowed-mcp-server-names" in by_bin["gemini"]
    assert "--admin-policy" in by_bin["gemini"]
    policy_path = _gemini_deny_policy_path()
    assert str(policy_path) in by_bin["gemini"]
    assert policy_path.is_file()
    assert 'decision = "deny"' in policy_path.read_text(encoding="utf-8")
    # pi
    for flag in ("--no-session", "--no-context-files", "--no-tools", "--no-extensions", "--no-mcp"):
        assert flag in by_bin["pi"]


def test_disposable_warm_spare_lifecycle_streaming_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify pre-spawned disposable warm spares for `agy` & `claude`: single-use, streaming, cleanup, fallback."""
    from mockery.llm import InterviewCopilotLLM, LLMRuntimeConfig

    class _FakeStdin:
        def __init__(self, *, broken: bool = False) -> None:
            self.written: list[bytes] = []
            self.closed = False
            self._broken = broken

        def is_closing(self) -> bool:
            return self.closed

        def write(self, data: bytes) -> None:
            if self._broken:
                raise BrokenPipeError("spare died")
            self.written.append(data)

        async def drain(self) -> None:
            if self._broken:
                raise BrokenPipeError("spare died")

        def close(self) -> None:
            self.closed = True

    class _FakeStdout:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        async def readline(self) -> bytes:
            if self._lines:
                return self._lines.pop(0)
            return b""

    class _FakeStderr:
        def __init__(self, data: bytes = b"") -> None:
            self._data = data

        async def read(self) -> bytes:
            return self._data

    class _FakeWarmProc:
        def __init__(self, stdout_lines: list[bytes], *, broken_stdin: bool = False, stderr: bytes = b"") -> None:
            self.returncode: int | None = None
            self.stdin = _FakeStdin(broken=broken_stdin)
            self.stdout = _FakeStdout(stdout_lines)
            self.stderr = _FakeStderr(stderr)
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def kill(self) -> None:
            self.terminated = True
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        async def communicate(self, _stdin: bytes | None = None) -> tuple[bytes, bytes]:
            self.returncode = 0
            return (b'{"status":"SUCCESS","response":"FALLBACK_ONESHOT"}', b"")

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/fake")

    async def _run_all() -> None:
        # 1. agy warm spare: streams deltas, uses {"event":"user"}, terminates after 1 turn, replenishes next spare
        spawned_agy: list[tuple[tuple[str, ...], _FakeWarmProc]] = []

        async def fake_agy_exec(*args: Any, **_kwargs: Any):
            str_args = tuple(str(a) for a in args)
            if "--input-format" in str_args:
                proc = _FakeWarmProc(
                    [
                        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Hel"}}).encode() + b"\n",
                        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "lo"}}).encode() + b"\n",
                        json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "Hello"}}).encode() + b"\n",
                    ]
                )
            else:
                proc = _FakeWarmProc([])
            spawned_agy.append((str_args, proc))
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_agy_exec)
        agy_cfg = LLMRuntimeConfig(tmp_path)
        agy_cfg.provider = "agy"
        agy_cfg.base_url = "cli://agy"
        agy_cfg.fast_model = "gemini-4-hydrogen:low"
        agy_llm = InterviewCopilotLLM(agy_cfg)

        agy_llm.prewarm_fast_model()
        await asyncio.gather(*list(agy_llm._warm_spare_tasks))
        assert len(spawned_agy) == 1
        first_args, first_proc = spawned_agy[0]
        assert "--input-format" in first_args and "stream-json" in first_args
        assert "--sandbox" in first_args and "--settings" in first_args

        deltas: list[str] = []

        async def on_delta(chunk: str) -> None:
            deltas.append(chunk)

        out = await agy_llm._complete_text(
            model="gemini-4-hydrogen:low",
            system="You are an interviewer.",
            user="Candidate answer.",
            expect_json=False,
            on_delta=on_delta,
        )
        assert out == "Hello"
        assert deltas == ["Hel", "lo"]
        assert first_proc.terminated is True
        sent_env = json.loads(first_proc.stdin.written[0].decode("utf-8"))
        assert sent_env["event"] == "user"
        assert sent_env["message"]["role"] == "user"
        assert "<INSTRUCTIONS>" in sent_env["message"]["content"]

        # Replenishment task spawned a fresh spare; aclose() terminates it cleanly
        await asyncio.gather(*list(agy_llm._warm_spare_tasks))
        assert len(spawned_agy) == 2
        second_proc = spawned_agy[1][1]
        assert second_proc.terminated is False
        await agy_llm.aclose()
        assert second_proc.terminated is True
        assert agy_llm._warm_spare is None

        # 2. Broken warm spare transparently falls back to hardened one-shot
        fallback_calls: list[tuple[str, ...]] = []

        async def fake_agy_fallback_exec(*args: Any, **_kwargs: Any):
            str_args = tuple(str(a) for a in args)
            fallback_calls.append(str_args)
            if "--input-format" in str_args:
                return _FakeWarmProc([], broken_stdin=True)
            return _FakeWarmProc([])

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_agy_fallback_exec)
        agy_llm.prewarm_fast_model()
        await asyncio.gather(*list(agy_llm._warm_spare_tasks))
        fb_out = await agy_llm._complete_text(
            model="gemini-4-hydrogen:low",
            system="Sys",
            user="Prompt",
            expect_json=True,
        )
        assert fb_out == "FALLBACK_ONESHOT"
        assert len(fallback_calls) == 2
        assert "--input-format" in fallback_calls[0]
        assert "--input-format" not in fallback_calls[1]
        await agy_llm.aclose()

        # 3. claude warm spare uses {"type":"user"}, streams content_block_delta, and locks down tools/MCP
        spawned_claude: list[tuple[tuple[str, ...], _FakeWarmProc]] = []

        async def fake_claude_exec(*args: Any, **_kwargs: Any):
            str_args = tuple(str(a) for a in args)
            proc = _FakeWarmProc(
                [
                    json.dumps(
                        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "Hi "}}}
                    ).encode()
                    + b"\n",
                    json.dumps(
                        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "there"}}}
                    ).encode()
                    + b"\n",
                    json.dumps({"type": "result", "is_error": False, "result": "Hi there"}).encode() + b"\n",
                ]
            )
            spawned_claude.append((str_args, proc))
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_claude_exec)
        cl_cfg = LLMRuntimeConfig(tmp_path)
        cl_cfg.provider = "claude_cli"
        cl_cfg.base_url = "cli://claude"
        cl_cfg.fast_model = "claude-haiku-4-5-20251001"
        cl_llm = InterviewCopilotLLM(cl_cfg)
        cl_llm.prewarm_fast_model()
        await asyncio.gather(*list(cl_llm._warm_spare_tasks))
        assert len(spawned_claude) == 1
        cl_args, cl_proc = spawned_claude[0]
        assert "--strict-mcp-config" in cl_args
        assert "--setting-sources" in cl_args
        assert "--disable-slash-commands" in cl_args
        assert "--tools" in cl_args

        cl_deltas: list[str] = []

        async def on_cl_delta(chunk: str) -> None:
            cl_deltas.append(chunk)

        cl_out = await cl_llm._complete_text(
            model="claude-haiku-4-5-20251001",
            system="Coach",
            user="Turn 1",
            expect_json=False,
            on_delta=on_cl_delta,
        )
        assert cl_out == "Hi there"
        assert cl_deltas == ["Hi ", "there"]
        assert cl_proc.terminated is True
        cl_env = json.loads(cl_proc.stdin.written[0].decode("utf-8"))
        assert cl_env["type"] == "user"
        assert cl_env["message"]["role"] == "user"
        await cl_llm.aclose()

    asyncio.run(_run_all())


def test_prompt_caching_wire_formats_prefix_stability_and_cli_sandbox_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Anthropic/OpenRouter cache_control, OpenAI prompt_cache_key, prefix stability, and isolated CLI cwd."""
    from mockery.llm import InterviewCopilotLLM, LLMRuntimeConfig, _cli_sandbox_cwd
    from mockery.models import InterviewSession, QAGroup, SessionConfig, Utterance

    # 1. Practice interviewer prompt keeps byte-identical stable prefix across turns & live workspace edits
    cfg = LLMRuntimeConfig(tmp_path)
    cfg.provider = "openai"
    cfg.base_url = "https://api.openai.com/v1"
    cfg.api_key = "sk-test"
    llm = InterviewCopilotLLM(cfg)

    captured_calls: list[dict[str, Any]] = []

    async def fake_complete(**kwargs: Any) -> str:
        captured_calls.append(kwargs)
        return json.dumps(
            {
                "interviewer_reply": "Tell me more about the failure mode.",
                "should_advance": False,
                "carl_signals": {"context": "strong", "action": "partial", "result": "missing", "learning": "missing"},
                "coach_nudge": {"tone": "tip", "message": "Great framing; quantify the impact next."},
            }
        )

    monkeypatch.setattr(llm, "_complete_text", fake_complete)
    q = {"prompt": "Design a multi-region ledger.", "carl_focus": "Consistency & latency", "follow_ups": ["Idempotency?"]}
    t1 = [{"role": "interviewer", "text": "Design a multi-region ledger."}, {"role": "candidate", "text": "I'd start with Spanner."}]
    t2 = t1 + [{"role": "interviewer", "text": "How do you handle retries?"}, {"role": "candidate", "text": "Idempotency keys."}]

    asyncio.run(
        llm.generate_practice_interviewer_reply(
            role_title="Staff Engineer",
            company="Acme",
            interview_type="system_design",
            target_level="L6",
            coach_style="supportive",
            current_question=q,
            next_question=None,
            recent_turns=t1,
            candidate_answer="I'd start with Spanner.",
            follow_up_count=0,
            existing_carl={"context": "missing", "action": "missing", "result": "missing", "learning": "missing"},
            workspace_context={"diagram_summary": "v1 diagram"},
            cache_key="mockery:practice:prac_1",
        )
    )
    asyncio.run(
        llm.generate_practice_interviewer_reply(
            role_title="Staff Engineer",
            company="Acme",
            interview_type="system_design",
            target_level="L6",
            coach_style="supportive",
            current_question=q,
            next_question=None,
            recent_turns=t2,
            candidate_answer="Idempotency keys in outbox.",
            follow_up_count=1,
            existing_carl={"context": "strong", "action": "partial", "result": "missing", "learning": "missing"},
            workspace_context={"diagram_summary": "v2 completely edited diagram"},
            cache_key="mockery:practice:prac_1",
        )
    )
    assert len(captured_calls) == 2
    p1 = captured_calls[0]["user"][: captured_calls[0]["cache_prefix_len"]]
    p2 = captured_calls[1]["user"][: captured_calls[1]["cache_prefix_len"]]
    # Turn 2's cached prefix MUST strictly start with Turn 1's cached prefix (append-only!)
    assert p2.startswith(p1)
    assert "Follow-ups already asked" not in p1
    assert "v1 diagram" not in p1
    assert captured_calls[0]["cache_key"] == "mockery:practice:prac_1"

    # 2. Live Copilot suggest_probes keeps finalized turns in stable prefix and drafts in mutable suffix
    captured_calls.clear()
    session = InterviewSession(id="sess_1", config=SessionConfig(role_title="Staff Engineer", interview_type="system_design"))
    session.utterances.extend(
        [
            Utterance(id="u1", speaker="interviewer", text="Design a ledger.", start_ms=0, end_ms=2000, final=True),
            Utterance(id="u2", speaker="candidate", text="We shard by account id.", start_ms=3000, end_ms=7000, final=True),
            Utterance(id="u3", speaker="candidate", text="Still speaking draft one", start_ms=8000, end_ms=10000, final=False),
        ]
    )
    qa = QAGroup(id="qa_1", index=1, title="Ledger", utterance_ids=["u1", "u2", "u3"])
    session.qa_groups.append(qa)
    asyncio.run(llm.suggest_probes(session, qa))
    session.utterances[-1].text = "Still speaking draft two with more words"
    asyncio.run(llm.suggest_probes(session, qa))
    sp1 = captured_calls[0]["user"][: captured_calls[0]["cache_prefix_len"]]
    sp2 = captured_calls[1]["user"][: captured_calls[1]["cache_prefix_len"]]
    assert sp1 == sp2
    assert "speaking live" not in sp1
    assert captured_calls[0]["cache_key"] == "mockery:copilot:sess_1:qa_1"

    # 3. Anthropic native payload attaches ephemeral cache_control on system + user prefix block
    anth_cfg = LLMRuntimeConfig(tmp_path)
    anth_cfg.provider = "anthropic"
    anth_cfg.base_url = "https://api.anthropic.com/v1"
    anth_cfg.api_key = "sk-ant-test"
    anth_llm = InterviewCopilotLLM(anth_cfg)
    sent_payloads: list[dict[str, Any]] = []

    class _FakeResp:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": "OK"}]}

    class _FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def post(self, _url: str, *, headers: dict[str, Any], json: dict[str, Any]):
            sent_payloads.append(json)
            return _FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    asyncio.run(
        anth_llm._complete_text(
            model="claude-haiku-4-5-20251001",
            system="Static interviewer instructions.",
            user="STABLE PREFIX CONTENT\n\nMUTABLE SUFFIX",
            cache_prefix_len=len("STABLE PREFIX CONTENT"),
        )
    )
    assert len(sent_payloads) == 1
    p_anth = sent_payloads[0]
    assert p_anth["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert p_anth["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert p_anth["messages"][0]["content"][0]["text"] == "STABLE PREFIX CONTENT"
    assert p_anth["messages"][0]["content"][1]["text"] == "\n\nMUTABLE SUFFIX"

    # 4. OpenRouter + Claude attaches cache_control; OpenAI direct attaches prompt_cache_key
    oai_calls: list[dict[str, Any]] = []

    class _FakeCompletions:
        async def create(self, **kwargs: Any):
            oai_calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    fake_oai_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    or_cfg = LLMRuntimeConfig(tmp_path)
    or_cfg.provider = "openrouter"
    or_cfg.base_url = "https://openrouter.ai/api/v1"
    or_cfg.api_key = "sk-or-test"
    or_llm = InterviewCopilotLLM(or_cfg)
    monkeypatch.setattr(or_llm, "_client", lambda: fake_oai_client)
    asyncio.run(
        or_llm._complete_text(
            model="anthropic/claude-sonnet-4",
            system="Static system.",
            user="PREFIX\n\nSUFFIX",
            cache_prefix_len=len("PREFIX"),
        )
    )
    assert oai_calls[-1]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert oai_calls[-1]["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    oai_direct_llm = InterviewCopilotLLM(cfg)
    monkeypatch.setattr(oai_direct_llm, "_client", lambda: fake_oai_client)
    asyncio.run(
        oai_direct_llm._complete_text(
            model="gpt-4.1-mini",
            system="Static system.",
            user="PREFIX\n\nSUFFIX",
            cache_prefix_len=len("PREFIX"),
            cache_key="mockery:practice:prac_99",
        )
    )
    assert oai_calls[-1]["extra_body"]["prompt_cache_key"] == "mockery:practice:prac_99"

    # 5. CLI subprocesses execute in an isolated empty sandbox directory (never the user's git repo)
    sandbox = _cli_sandbox_cwd()
    assert sandbox.is_dir()
    seen_cwds: list[str | None] = []

    class _FakeProc:
        returncode = 0

        async def communicate(self, _stdin: bytes | None = None):
            return (b'{"status":"SUCCESS","response":"OK"}', b"")

    async def fake_subproc(*_args: Any, **kwargs: Any):
        seen_cwds.append(kwargs.get("cwd"))
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subproc)
    agy_cfg = LLMRuntimeConfig(tmp_path)
    agy_cfg.provider = "agy"
    agy_cfg.base_url = "cli://agy"
    agy_llm = InterviewCopilotLLM(agy_cfg)
    out = asyncio.run(
        agy_llm._complete_text(model="gemini-4-hydrogen:low", system="Sys", user="Hi", expect_json=False)
    )
    assert out == "OK"
    assert seen_cwds == [str(sandbox)]


def test_practice_conversation_strictly_append_only_workspace_delta_and_roundtrip(tmp_path: Path) -> None:
    """Canonical PracticeConversation must be strictly append-only across Q advances, hints, skips, and workspace edits."""
    import random

    from mockery.practice_conversation import (
        build_practice_turn_messages,
        commit_practice_conversation_turn,
    )
    from mockery.routers.practice import serialize_practice_session

    store = PracticeSessionStore(tmp_path / "practice")
    q1 = PracticeQuestionItem(id="pq_1", title="Ledger", prompt="Design a distributed ledger.", status="active")
    q2 = PracticeQuestionItem(id="pq_2", title="Cache", prompt="Design a distributed cache.", status="pending")
    session = PracticeSession(
        id="prac_append_only",
        role_title="Staff Engineer",
        company="Acme",
        config=PracticeConfig(interview_type="system_design", target_level="L6"),
        questions=[q1, q2],
        turns=[PracticeTurn(role="interviewer", question_id="pq_1", text="Welcome! Design a distributed ledger.")],
    )

    # Turn 1 with initial diagram
    msgs1, u1, ws1, sys1, reset1 = build_practice_turn_messages(
        session,
        current_question=q1.model_dump(),
        next_question=q2.model_dump(),
        candidate_answer="I'd start with a write-ahead log.",
        follow_up_count=0,
        max_followups=3,
        workspace_context={"diagram_summary": "Client -> Gateway -> WAL"},
    )
    assert reset1 is True
    assert len(msgs1) == 2
    assert "Client -> Gateway -> WAL" in u1
    a1 = '{"interviewer_reply":"Nice start. How do you replicate the WAL?","should_advance":false}'
    session.turns.append(PracticeTurn(role="candidate", question_id="pq_1", text="I'd start with a write-ahead log."))
    session.turns.append(PracticeTurn(role="interviewer", question_id="pq_1", text="Nice start. How do you replicate the WAL?"))
    commit_practice_conversation_turn(
        session,
        system_prompt=sys1,
        user_content=u1,
        assistant_raw=a1,
        workspace_snapshot=ws1,
        synced_turn_count=len(session.turns),
        reset_conversation=reset1,
    )

    # Persist and reload; public serializer MUST strip internal llm_conversation
    store.save(session)
    reloaded = store.get(session.id)
    assert reloaded is not None and reloaded.llm_conversation is not None
    public_dump = serialize_practice_session(SimpleNamespace(practice_debrief_jobs={}), reloaded)
    assert "llm_conversation" not in public_dump

    # Turn 2: unchanged workspace omitted, prefix byte-identical to Turn 1 + a1
    msgs2, u2, ws2, sys2, reset2 = build_practice_turn_messages(
        reloaded,
        current_question=q1.model_dump(),
        next_question=q2.model_dump(),
        candidate_answer="Raft quorum across 3 zones.",
        follow_up_count=1,
        max_followups=3,
        workspace_context={"diagram_summary": "Client -> Gateway -> WAL"},
    )
    assert reset2 is False
    assert msgs2[:2] == msgs1
    assert msgs2[2] == {"role": "assistant", "content": a1}
    assert "LIVE SYSTEM DESIGN DIAGRAM STATE" not in u2
    a2 = '{"interviewer_reply":"Great! Moving to Q2.","should_advance":true}'
    reloaded.turns.append(PracticeTurn(role="candidate", question_id="pq_1", text="Raft quorum across 3 zones."))
    reloaded.turns.append(PracticeTurn(role="interviewer", question_id="pq_2", text="Great! Moving to Q2."))
    commit_practice_conversation_turn(
        reloaded,
        system_prompt=sys2,
        user_content=u2,
        assistant_raw=a2,
        workspace_snapshot=ws2,
        synced_turn_count=len(reloaded.turns),
        reset_conversation=reset2,
    )

    # Intervening coach hint + workspace clear + advance to Q2 -> prefix STILL strictly preserved!
    reloaded.turns.append(PracticeTurn(role="coach", question_id="pq_2", text="Think about eviction policy."))
    msgs3, u3, ws3, sys3, reset3 = build_practice_turn_messages(
        reloaded,
        current_question=q2.model_dump(),
        next_question=None,
        candidate_answer="Consistent hashing ring with LRU nodes.",
        follow_up_count=0,
        max_followups=3,
        workspace_context={"diagram_summary": ""},
    )
    assert reset3 is False
    assert msgs3[:-1] == msgs2 + [{"role": "assistant", "content": a2}]
    assert "Coach hint shown to candidate: Think about eviction policy." in u3
    assert "(cleared)" in u3

    # Randomized 20-step fuzz verifying strict prefix preservation under arbitrary events
    rng = random.Random(1337)
    prev_msgs = msgs3
    prev_raw = '{"interviewer_reply":"Tell me more.","should_advance":false}'
    reloaded.turns.append(PracticeTurn(role="candidate", question_id="pq_2", text="Consistent hashing ring."))
    reloaded.turns.append(PracticeTurn(role="interviewer", question_id="pq_2", text="Tell me more."))
    commit_practice_conversation_turn(
        reloaded,
        system_prompt=sys3,
        user_content=u3,
        assistant_raw=prev_raw,
        workspace_snapshot=ws3,
        synced_turn_count=len(reloaded.turns),
        reset_conversation=reset3,
    )
    for step in range(20):
        if rng.random() < 0.35:
            reloaded.turns.append(PracticeTurn(role="coach", question_id="pq_2", text=f"Hint {step}"))
        if rng.random() < 0.25:
            reloaded.turns.append(PracticeTurn(role="interviewer", question_id="pq_2", text=f"Manual skip {step}"))
        ws_ctx = {"code": f"def f(): return {step}" if rng.random() < 0.6 else "", "language": "python"}
        nxt_msgs, nxt_u, nxt_ws, nxt_sys, nxt_reset = build_practice_turn_messages(
            reloaded,
            current_question=q2.model_dump(),
            next_question=None,
            candidate_answer=f"Answer {step}",
            follow_up_count=step % 3,
            max_followups=3,
            workspace_context=ws_ctx,
        )
        assert nxt_reset is False
        assert nxt_msgs[:-1] == prev_msgs + [{"role": "assistant", "content": prev_raw}]
        prev_msgs = nxt_msgs
        prev_raw = json.dumps({"interviewer_reply": f"Reply {step}", "should_advance": False})
        reloaded.turns.append(PracticeTurn(role="candidate", question_id="pq_2", text=f"Answer {step}"))
        reloaded.turns.append(PracticeTurn(role="interviewer", question_id="pq_2", text=f"Reply {step}"))
        commit_practice_conversation_turn(
            reloaded,
            system_prompt=nxt_sys,
            user_content=nxt_u,
            assistant_raw=prev_raw,
            workspace_snapshot=nxt_ws,
            synced_turn_count=len(reloaded.turns),
            reset_conversation=nxt_reset,
        )


def test_persistent_cli_session_worker_caching_rehydration_and_lru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent `agy` & `claude` session workers reuse 1 process with delta-only stdin, rehydrate on crash, and evict LRU."""
    from mockery.llm import _MAX_CLI_SESSION_WORKERS, InterviewCopilotLLM, LLMRuntimeConfig

    class _InteractiveStdin:
        def __init__(self, on_write: Any) -> None:
            self.written: list[bytes] = []
            self.closed = False
            self._on_write = on_write
            self.fail_next = False

        def is_closing(self) -> bool:
            return self.closed

        def write(self, data: bytes) -> None:
            if self.fail_next:
                raise BrokenPipeError("worker died")
            self.written.append(data)
            self._on_write(data)

        async def drain(self) -> None:
            if self.fail_next:
                raise BrokenPipeError("worker died")

        def close(self) -> None:
            self.closed = True

    class _QueueStdout:
        def __init__(self) -> None:
            self._q: asyncio.Queue[bytes] = asyncio.Queue()

        def push(self, line: bytes) -> None:
            self._q.put_nowait(line)

        async def readline(self) -> bytes:
            return await self._q.get()

    class _IdleStderr:
        def __init__(self) -> None:
            self._ev = asyncio.Event()

        def finish(self) -> None:
            self._ev.set()

        async def read(self, _n: int = -1) -> bytes:
            await self._ev.wait()
            return b""

    class _FakePersistentProc:
        def __init__(self, backend: str) -> None:
            self.backend = backend
            self.returncode: int | None = None
            self.stdout = _QueueStdout()
            self.stderr = _IdleStderr()
            self.turn_idx = 0
            self.stdin = _InteractiveStdin(self._handle_write)
            self.terminated = False

        def _handle_write(self, data: bytes) -> None:
            self.turn_idx += 1
            env = json.loads(data.decode("utf-8"))
            txt = f'{{"reply":"turn_{self.turn_idx}","backend":"{self.backend}"}}'
            if self.backend == "agy":
                assert env["event"] == "user"
                self.stdout.push(
                    json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": txt}}).encode() + b"\n"
                )
            else:
                assert env["type"] == "user"
                self.stdout.push(
                    json.dumps({"type": "result", "is_error": False, "result": txt}).encode() + b"\n"
                )

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0
            self.stderr.finish()
            self.stdout.push(b"")

        def kill(self) -> None:
            self.terminate()

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/fake")

    async def _run() -> None:
        # 1. agy persistent session: prewarm + 3 turns on 1 single process, delta-only on turns 2 & 3, crash recovery on turn 4
        spawned: list[_FakePersistentProc] = []

        async def fake_exec(*args: Any, **_kwargs: Any):
            proc = _FakePersistentProc(str(args[0]))
            spawned.append(proc)
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        cfg = LLMRuntimeConfig(tmp_path)
        cfg.provider = "agy"
        cfg.base_url = "cli://agy"
        cfg.fast_model = "gemini-4-hydrogen:low"
        llm = InterviewCopilotLLM(cfg)

        llm.prewarm_session_worker("practice:s1", system="SYS_PROMPT", expect_json=True)
        await asyncio.gather(*list(llm._warm_spare_tasks))
        assert len(spawned) == 1
        assert spawned[0].stdin.written == []  # Prewarm sends 0 tokens!

        msgs = [
            {"role": "system", "content": "SYS_PROMPT"},
            {"role": "user", "content": "TURN_1_PLAN_AND_ANSWER"},
        ]
        r1 = await llm._complete_messages(msgs, model=cfg.fast_model, expect_json=True, session_key="practice:s1")
        assert len(spawned) == 1
        assert spawned[0].terminated is False
        sent1 = json.loads(spawned[0].stdin.written[0].decode())["message"]["content"]
        assert "<INSTRUCTIONS>" in sent1 and "TURN_1_PLAN_AND_ANSWER" in sent1

        # Turn 2: SAME process, sends ONLY delta user message!
        msgs.extend([{"role": "assistant", "content": r1}, {"role": "user", "content": "TURN_2_DELTA_ONLY"}])
        r2 = await llm._complete_messages(msgs, model=cfg.fast_model, expect_json=True, session_key="practice:s1")
        assert len(spawned) == 1
        sent2 = json.loads(spawned[0].stdin.written[1].decode())["message"]["content"]
        assert sent2 == "TURN_2_DELTA_ONLY"

        # Turn 3: SAME process, STILL delta-only!
        msgs.extend([{"role": "assistant", "content": r2}, {"role": "user", "content": "TURN_3_DELTA_ONLY"}])
        r3 = await llm._complete_messages(msgs, model=cfg.fast_model, expect_json=True, session_key="practice:s1")
        assert len(spawned) == 1
        sent3 = json.loads(spawned[0].stdin.written[2].decode())["message"]["content"]
        assert sent3 == "TURN_3_DELTA_ONLY"

        # Turn 4: simulate silent crash of process 0 -> auto-rehydrates on process 1 with full flat history!
        spawned[0].stdin.fail_next = True
        msgs.extend([{"role": "assistant", "content": r3}, {"role": "user", "content": "TURN_4_AFTER_CRASH"}])
        r4 = await llm._complete_messages(msgs, model=cfg.fast_model, expect_json=True, session_key="practice:s1")
        assert len(spawned) == 2
        assert spawned[0].terminated is True
        sent4 = json.loads(spawned[1].stdin.written[0].decode())["message"]["content"]
        assert "<CONVERSATION_HISTORY>" in sent4
        assert "TURN_1_PLAN_AND_ANSWER" in sent4
        assert "TURN_4_AFTER_CRASH" in sent4

        # Turn 5: new process 1 is now synced and back to delta-only!
        msgs.extend([{"role": "assistant", "content": r4}, {"role": "user", "content": "TURN_5_DELTA_AGAIN"}])
        await llm._complete_messages(msgs, model=cfg.fast_model, expect_json=True, session_key="practice:s1")
        assert len(spawned) == 2
        sent5 = json.loads(spawned[1].stdin.written[1].decode())["message"]["content"]
        assert sent5 == "TURN_5_DELTA_AGAIN"

        # LRU eviction: opening more than _MAX_CLI_SESSION_WORKERS evicts oldest idle worker
        for extra_idx in range(_MAX_CLI_SESSION_WORKERS):
            llm.prewarm_session_worker(f"practice:extra_{extra_idx}", system="SYS", expect_json=True)
        await asyncio.gather(*list(llm._warm_spare_tasks))
        assert len(llm._session_workers) == _MAX_CLI_SESSION_WORKERS
        assert spawned[1].terminated is True  # oldest worker ("practice:s1") was evicted & reaped

        await llm.aclose()
        assert all(p.terminated for p in spawned)

        # 2. claude persistent session: passes native --system-prompt once & sends delta-only on Turn 2
        cl_spawns: list[tuple[tuple[str, ...], _FakePersistentProc]] = []

        async def fake_cl_exec(*args: Any, **_kwargs: Any):
            proc = _FakePersistentProc("claude")
            cl_spawns.append((tuple(str(a) for a in args), proc))
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cl_exec)
        cl_cfg = LLMRuntimeConfig(tmp_path)
        cl_cfg.provider = "claude_cli"
        cl_cfg.base_url = "cli://claude"
        cl_cfg.fast_model = "claude-haiku-4-5-20251001"
        cl_llm = InterviewCopilotLLM(cl_cfg)
        cl_llm.prewarm_session_worker("practice:cl1", system="CLAUDE_SYS", expect_json=True)
        await asyncio.gather(*list(cl_llm._warm_spare_tasks))
        assert len(cl_spawns) == 1
        assert "CLAUDE_SYS" in cl_spawns[0][0][cl_spawns[0][0].index("--system-prompt") + 1]

        cl_msgs = [{"role": "system", "content": "CLAUDE_SYS"}, {"role": "user", "content": "CL_TURN_1"}]
        cl_r1 = await cl_llm._complete_messages(cl_msgs, model=cl_cfg.fast_model, expect_json=True, session_key="practice:cl1")
        assert len(cl_spawns) == 1
        cl_msgs.extend([{"role": "assistant", "content": cl_r1}, {"role": "user", "content": "CL_TURN_2"}])
        await cl_llm._complete_messages(cl_msgs, model=cl_cfg.fast_model, expect_json=True, session_key="practice:cl1")
        assert len(cl_spawns) == 1
        assert json.loads(cl_spawns[0][1].stdin.written[1].decode())["message"]["content"] == "CL_TURN_2"
        await cl_llm.aclose()
        assert cl_spawns[0][1].terminated is True

    asyncio.run(_run())


def test_multiturn_http_sliding_cache_and_authoritative_debrief_carl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify multi-turn HTTP sliding cache breakpoints and authoritative Debrief CARL write-back by question_id."""
    import httpx

    from mockery.llm import InterviewCopilotLLM, LLMRuntimeConfig

    # 1. Anthropic multi-turn attaches ephemeral cache_control on system + last message (sliding breakpoint)
    sent_anth: list[dict[str, Any]] = []

    class _FakeResp:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": "OK"}]}

    class _FakeClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def post(self, _url: str, *, headers: dict[str, Any], json: dict[str, Any]):
            sent_anth.append(json)
            return _FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    anth_cfg = LLMRuntimeConfig(tmp_path)
    anth_cfg.provider = "anthropic"
    anth_cfg.base_url = "https://api.anthropic.com/v1"
    anth_cfg.api_key = "sk-ant-test"
    anth_llm = InterviewCopilotLLM(anth_cfg)
    asyncio.run(
        anth_llm._complete_messages(
            [
                {"role": "system", "content": "Frozen system."},
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Reply 1"},
                {"role": "user", "content": "Turn 2"},
            ],
            model="claude-haiku-4-5-20251001",
        )
    )
    assert sent_anth[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent_anth[0]["messages"][0] == {"role": "user", "content": "Turn 1"}
    assert sent_anth[0]["messages"][1] == {"role": "assistant", "content": "Reply 1"}
    assert sent_anth[0]["messages"][2]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # 2. End-to-end Practice API: skip Q1, answer Q2, Debrief writes CARL back by question_id & matches Home score
    app = create_app(project_root=tmp_path, preload_asr=False)
    rt = app.state.runtime
    rt.llm_config.provider = "openai"
    rt.llm_config.api_key = "sk-test"

    async def fake_plan(**_kw: Any):
        return (
            [
                {"id": "q_skip", "title": "Skipped Q", "prompt": "Q1 prompt"},
                {"id": "q_ans", "title": "Answered Q", "prompt": "Q2 prompt"},
            ],
            "Welcome!",
        )

    async def fake_reply(**_kw: Any):
        return {
            "interviewer_reply": "Thanks, wrapping up!",
            "should_advance": True,
            "carl_signals": {"context": "partial", "action": "partial", "result": "missing", "learning": "missing"},
            "coach_nudge": {"tone": "tip", "message": "Add metrics."},
        }

    async def fake_debrief(**_kw: Any):
        return {
            "overall_encouragement": "Great rep!",
            "top_strengths": ["Clear framing."],
            "carl_primer": "CARL works.",
            "question_reviews": [
                {
                    "question_id": "q_ans",
                    "title": "Answered Q",
                    "prompt": "Q2 prompt",
                    "carl": {"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                    "strength": "Strong.",
                    "coaching_tip": "Keep going.",
                    "carl_reframe": "Example.",
                }
            ],
            "retry_drill": {"question_id": "q_ans", "focus": "Polish", "prompt": "Retry."},
        }

    monkeypatch.setattr(rt.llm, "generate_practice_plan", fake_plan)
    monkeypatch.setattr(rt.llm, "generate_practice_interviewer_reply", fake_reply)
    monkeypatch.setattr(rt.llm, "generate_practice_debrief", fake_debrief)

    with TestClient(app) as client:
        start = client.post("/api/practice/start", json={"interview_type": "behavioral", "question_count": 2}).json()
        sid = start["session"]["id"]
        assert "llm_conversation" not in start["session"]
        # Skip Q1 cleanly
        assert client.post(f"/api/practice/{sid}/next").status_code == 200
        # Answer Q2
        turn_res = client.post(f"/api/practice/{sid}/turn", json={"text": "I drove the rollout and cut p99 by 40%."})
        assert turn_res.status_code == 200
        assert "llm_conversation" not in turn_res.json()["session"]
        # Debrief writes authoritative CARL back onto q_ans (NOT q_skip!)
        deb_res = client.post(f"/api/practice/{sid}/debrief")
        assert deb_res.status_code == 200
        qs_by_id = {q["id"]: q for q in deb_res.json()["session"]["questions"]}
        assert qs_by_id["q_skip"]["carl_signals"]["context"] == "missing"
        assert qs_by_id["q_ans"]["carl_signals"] == {
            "context": "strong",
            "action": "strong",
            "result": "strong",
            "learning": "strong",
        }
        # Home history rail score agrees with Debrief!
        listed = client.get("/api/practice/sessions").json()["sessions"]
        assert listed[0]["id"] == sid
        assert listed[0]["score"] == 2.0  # avg across q_skip (0) and q_ans (4) -> partial on all 4 dims = 2.0
