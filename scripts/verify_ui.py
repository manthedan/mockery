#!/usr/bin/env python3
"""Checked-in headless Chrome CDP verification CLI for Mockery.

Subcommands:
  uat      Deterministic 24-check Browser UAT across empty first-run & fault-injected seeded studios.
  explore  Bombadil-style property-based UI explorer checking universal layout & runtime invariants.

Usage:
  uv run python scripts/verify_ui.py uat
  uv run python scripts/verify_ui.py explore --steps 40 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mockery.models import InterviewSession, QAGroup, SessionConfig, Utterance  # noqa: E402
from mockery.practice import (  # noqa: E402
    PracticeConfig,
    PracticeQuestionItem,
    PracticeSession,
    PracticeSessionStore,
    PracticeTurn,
)
from mockery.profiles import CandidateProfileStore  # noqa: E402
from mockery.redactor import RampartRedactor  # noqa: E402
from mockery.storage import SessionStore  # noqa: E402

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def find_chrome() -> str:
    env_bin = os.environ.get("MOCKERY_CHROME_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    for cand in CHROME_CANDIDATES:
        if Path(cand).exists():
            return cand
    found = shutil.which("google-chrome") or shutil.which("chromium")
    if found:
        return found
    raise RuntimeError("Google Chrome / Chromium not found. Set MOCKERY_CHROME_BIN to override.")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def seed_demo_data_dir(data_dir: Path, *, offline_llm: bool = True) -> None:
    """Populate an isolated temporary data dir with strictly synthetic demo sessions."""
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if offline_llm:
        env_text = (
            "LLM_PROVIDER=openai_compat\n"
            "LLM_BASE_URL=http://127.0.0.1:1/v1\n"
            "LLM_API_KEY=test-offline-key\n"
            "LLM_FAST_MODEL=offline-model\n"
            "LLM_CALIBRATION_MODEL=offline-model\n"
            "PARAKEET_MODEL=mlx-community/parakeet-tdt-0.6b-v2\n"
        )
    else:
        env_text = (
            "LLM_PROVIDER=agy\n"
            "LLM_BASE_URL=cli://agy\n"
            "LLM_FAST_MODEL=gemini-4-hydrogen:low\n"
            "LLM_CALIBRATION_MODEL=gemini-4-hydrogen:high\n"
            "PARAKEET_MODEL=mlx-community/parakeet-tdt-0.6b-v2\n"
        )
    (data_dir / ".env.local").write_text(env_text, encoding="utf-8")

    # 1. Synthetic Candidate Profile (fictional Jordan Vance / FinScale Payments)
    redactor = RampartRedactor()
    profile_store = CandidateProfileStore(data_dir / "profiles", redactor)
    resume_text = (
        "Jordan Vance\n"
        "San Francisco, CA | jordan.vance@finscale.io | (415) 555-0194 | https://github.com/jordanvance\n\n"
        "EXPERIENCE\n"
        "Staff Backend Engineer, FinScale Payments (2022–Present)\n"
        "- Architected a multi-region double-entry ledger handling 48,000 TPS at 99.995% availability.\n"
        "- Led zero-downtime migration from monolithic Postgres settlement tables to sharded Spanner partitions.\n"
        "- Built deterministic idempotency-key replay middleware adopted by 14 product teams.\n"
    )
    profile_store.set_resume(filename="Jordan_Vance_Resume.pdf", raw_text=resume_text)
    profile_store.add_job_listing(
        title="Staff Backend Engineer — Core Payments Ledger",
        company="Pied Piper",
        url="https://piedpiper.com/jobs/listing/staff-backend-engineer-core-payments/6102488",
        text=(
            "Pied Piper is looking for a Staff Backend Engineer (L6) to join Core Payments Ledger.\n"
            "Own multi-region double-entry ledger reliability, idempotency, and sub-30ms p99 authorization."
        ),
        target_level="L6",
    )

    practice_store = PracticeSessionStore(data_dir / "practice")
    now_ms = int(time.time() * 1000)

    # 2a. Completed Debrief + Retry Drill
    practice_store.save(
        PracticeSession(
            id="prac_debrief_retry",
            created_at_iso="2026-11-20T18:30:00+00:00",
            role_title="Staff Backend Engineer — Core Payments Ledger",
            company="Pied Piper",
            config=PracticeConfig(
                interview_type="behavioral",
                question_source="mixed",
                target_level="L6",
                coach_style="supportive",
                live_coach_nudges=True,
                voice_enabled=True,
                question_count=2,
            ),
            phase="completed",
            current_question_index=1,
            questions=[
                PracticeQuestionItem(
                    id="pq_d1",
                    bank_question_id="beh_own_01",
                    source="tailored",
                    category="behavioral",
                    title="Zero-Downtime Ledger Sharding",
                    competency="Ownership & 0-to-1",
                    prompt="Tell me about a time you drove a high-risk infrastructure migration from 0 to 1 without disrupting live customer traffic.",
                    status="completed",
                    follow_up_count=1,
                    carl_signals={"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                ),
                PracticeQuestionItem(
                    id="pq_d2",
                    bank_question_id="beh_conf_02",
                    source="bank",
                    category="behavioral",
                    title="Pushing Back on an Aggressive Launch Date",
                    competency="Conflict & Pushback",
                    prompt="Tell me about a time you pushed back on a product requirement or aggressive deadline.",
                    status="completed",
                    follow_up_count=1,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
            ],
            turns=[
                PracticeTurn(
                    id="pt_d1",
                    role="interviewer",
                    question_id="pq_d1",
                    text="Tell me about a time you drove a high-risk infrastructure migration without disrupting live traffic.",
                    timestamp_ms=now_ms - 86400_000,
                ),
                PracticeTurn(
                    id="pt_d2",
                    role="candidate",
                    question_id="pq_d1",
                    text="I built a dual-write shadow verifier and migrated cohorts with zero downtime, cutting p99 latency from 184 ms to 29 ms.",
                    timestamp_ms=now_ms - 86350_000,
                    carl_signals={"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                ),
                PracticeTurn(
                    id="pt_d3",
                    role="interviewer",
                    question_id="pq_d2",
                    text="Tell me about a time you pushed back on an aggressive launch date.",
                    timestamp_ms=now_ms - 86200_000,
                ),
                PracticeTurn(
                    id="pt_d4",
                    role="candidate",
                    question_id="pq_d2",
                    text="I proposed a capped-tier rollout so we could land idempotency guards before full Black Friday volume.",
                    timestamp_ms=now_ms - 86120_000,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
            ],
            debrief={
                "overall_encouragement": "Strong Staff-level composure and crisp technical ownership.",
                "carl_primer": "Context sets stakes · Action shows leverage · Result anchors impact · Learning shows growth.",
                "top_strengths": [
                    "Led with concrete latency and reliability stakes.",
                    "Paired pushback with a pragmatic phased-rollout alternative.",
                ],
                "question_reviews": [
                    {
                        "question_id": "pq_d1",
                        "title": "Zero-Downtime Ledger Sharding",
                        "prompt": "Tell me about a high-risk infrastructure migration.",
                        "carl": {"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                        "strength": "Immediate clarity on blast radius and dual-write shadow verification.",
                        "coaching_tip": "Note how partner teams aligned on freeze windows.",
                        "carl_reframe": "",
                    },
                    {
                        "question_id": "pq_d2",
                        "title": "Pushing Back on an Aggressive Launch Date",
                        "prompt": "Tell me about pushing back on an aggressive launch date.",
                        "carl": {"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                        "strength": "Constructive phased rollout proposal.",
                        "coaching_tip": "Quantify prevented exposure and add one process takeaway.",
                        "carl_reframe": "Cite $1.4M modeled exposure, $42M clean volume, and the new kickoff checklist.",
                    },
                ],
                "retry_drill": {
                    "question_id": "pq_d2",
                    "focus": "Quantify risk saved & add an org-level Learning",
                    "prompt": "Re-deliver the final 45 seconds naming the risk prevented, weekend outcome, and process change.",
                },
            },
            retry_history=[],
        )
    )
    time.sleep(0.02)

    # 2b. Active Coding Studio Session
    coding_code_js = (
        "class TTLLRUCache {\n"
        "  constructor(capacity) {\n"
        "    this.capacity = Math.max(0, capacity);\n"
        "    this.entries = new Map();\n"
        "  }\n"
        "  get(key, nowMs) {\n"
        "    const entry = this.entries.get(key);\n"
        "    if (!entry || entry.expiresAt <= nowMs) { this.entries.delete(key); return null; }\n"
        "    this.entries.delete(key); this.entries.set(key, entry);\n"
        "    return entry.value;\n"
        "  }\n"
        "  put(key, value, ttlMs, nowMs) {\n"
        "    if (this.capacity === 0 || ttlMs <= 0) return;\n"
        "    if (this.entries.has(key)) this.entries.delete(key);\n"
        "    else if (this.entries.size >= this.capacity) this.entries.delete(this.entries.keys().next().value);\n"
        "    this.entries.set(key, { value, expiresAt: nowMs + ttlMs });\n"
        "  }\n"
        "}\n"
    )
    coding_tests_js = (
        "const assert = (c, m) => { if (!c) throw new Error(m || 'Assertion failed'); };\n"
        "const c = new TTLLRUCache(2);\n"
        "c.put('a', 1, 100, 0);\n"
        "c.put('b', 2, 500, 10);\n"
        "assert(c.get('a', 50) === 1, 'a alive');\n"
        "assert(c.get('a', 100) === null, 'a expired');\n"
        "console.log('2/2 assertions passed');\n"
    )
    practice_store.save(
        PracticeSession(
            id="prac_code_active",
            created_at_iso="2026-11-21T14:15:00+00:00",
            role_title="Staff Backend Engineer — Core Payments Ledger",
            company="Pied Piper",
            config=PracticeConfig(
                interview_type="coding",
                question_source="bank_only",
                target_level="L6",
                coach_style="supportive",
                live_coach_nudges=True,
                voice_enabled=True,
                question_count=2,
            ),
            phase="active",
            current_question_index=0,
            questions=[
                PracticeQuestionItem(
                    id="pq_c1",
                    bank_question_id="code_ttl_lru_01",
                    source="bank",
                    category="coding",
                    title="LRU Cache with Per-Key TTL",
                    competency="Data Structures & Systems",
                    prompt="Implement TTLLRUCache(capacity) with get(key, now_ms) and put(key, value, ttl_ms, now_ms).",
                    starter_code={
                        "javascript": coding_code_js,
                        "python": "class TTLLRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n",
                    },
                    tests={"javascript": coding_tests_js, "python": "print('ALL TESTS PASSED')\n"},
                    status="active",
                    follow_up_count=1,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
                PracticeQuestionItem(
                    id="pq_c2",
                    bank_question_id="code_rate_01",
                    source="bank",
                    category="coding",
                    title="Sliding-Window Rate Limiter",
                    competency="Practical Systems",
                    prompt="Implement RateLimiter(limit, window_ms) with allow(user_id, now_ms) -> bool.",
                    status="pending",
                ),
            ],
            turns=[
                PracticeTurn(
                    id="pt_c1",
                    role="interviewer",
                    question_id="pq_c1",
                    text="Implement TTLLRUCache(capacity) with O(1) average get and put.",
                    timestamp_ms=now_ms - 7200_000,
                ),
                PracticeTurn(
                    id="pt_c2",
                    role="candidate",
                    question_id="pq_c1",
                    text="I'm using JavaScript's Map for O(1) LRU recency updates with lazy TTL checks.",
                    timestamp_ms=now_ms - 7140_000,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
            ],
            workspace_context={
                "language": "javascript",
                "code": coding_code_js,
                "code_by_lang": {
                    "javascript": coding_code_js,
                    "python": "class TTLLRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n",
                    "pseudocode": "// Map + min-heap for earliest expiry\n",
                },
                "tests_by_lang": {"javascript": coding_tests_js, "python": "print('ALL TESTS PASSED')\n"},
                "test_output": "[PASSED] JavaScript (2 ms)\n\nSTDOUT:\n2/2 assertions passed",
            },
        )
    )
    time.sleep(0.02)

    # 2c. Active System Design Session with Whiteboard Graph
    sd_graph = {
        "version": 1,
        "nodes": [
            {"id": "n_cli", "type": "client", "label": "Merchant SDKs", "note": "48k TPS · TLS 1.3", "x": 24, "y": 156},
            {"id": "n_gw", "type": "gateway", "label": "Global API Gateway", "note": "Auth · Idempotency", "x": 216, "y": 156},
            {"id": "n_svc", "type": "service", "label": "Ledger Core API", "note": "Double-entry validator", "x": 420, "y": 156},
            {"id": "n_db", "type": "database", "label": "Spanner Ledger DB", "note": "TrueTime · Sharded", "x": 636, "y": 96},
            {"id": "n_mq", "type": "queue", "label": "Settlement Stream", "note": "Kafka · Exactly-once", "x": 636, "y": 240},
        ],
        "edges": [
            {"id": "e1", "from": "n_cli", "to": "n_gw", "label": "HTTPS"},
            {"id": "e2", "from": "n_gw", "to": "n_svc", "label": "gRPC"},
            {"id": "e3", "from": "n_svc", "to": "n_db", "label": "ACID commit"},
            {"id": "e4", "from": "n_svc", "to": "n_mq", "label": "Outbox event"},
        ],
    }
    practice_store.save(
        PracticeSession(
            id="prac_sd_active",
            created_at_iso="2026-11-21T15:40:00+00:00",
            role_title="Staff Backend Engineer — Core Payments Ledger",
            company="Pied Piper",
            config=PracticeConfig(
                interview_type="system_design",
                question_source="mixed",
                target_level="L6",
                coach_style="supportive",
                live_coach_nudges=True,
                voice_enabled=True,
                question_count=2,
            ),
            phase="active",
            current_question_index=0,
            questions=[
                PracticeQuestionItem(
                    id="pq_sd1",
                    bank_question_id="sd_ratelimit_01",
                    source="tailored",
                    category="system_design",
                    title="Global Payments Gateway & Double-Entry Ledger",
                    competency="Distributed Systems & API Platform",
                    prompt="Design a multi-region API gateway and double-entry payments ledger capable of 50,000 TPS.",
                    core_requirements=[
                        "Sub-5ms gateway overhead with per-merchant rate limiting",
                        "Strict double-entry balance invariants with idempotent retries",
                        "Transactional outbox to Kafka for zero-loss reconciliation",
                    ],
                    status="active",
                    follow_up_count=1,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
                PracticeQuestionItem(
                    id="pq_sd2",
                    bank_question_id="sd_rbac_01",
                    source="bank",
                    category="system_design",
                    title="Multi-Tenant Identity & Permissions",
                    competency="Enterprise Foundations",
                    prompt="Design a fine-grained authorization service (<10ms p99).",
                    status="pending",
                ),
            ],
            turns=[
                PracticeTurn(
                    id="pt_sd1",
                    role="interviewer",
                    question_id="pq_sd1",
                    text="Sketch your core data path on the whiteboard and walk me through idempotency.",
                    timestamp_ms=now_ms - 3600_000,
                ),
                PracticeTurn(
                    id="pt_sd2",
                    role="candidate",
                    question_id="pq_sd1",
                    text="Every mutation carries an Idempotency-Key committed atomically with double-entry rows and an outbox record.",
                    timestamp_ms=now_ms - 3520_000,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
            ],
            workspace_context={
                "diagram_graph": sd_graph,
                "diagram_summary": "Merchant SDKs -> Global API Gateway -> Ledger Core API -> Spanner Ledger DB & Settlement Stream",
                "checked_requirements": ["Sub-5ms gateway overhead with per-merchant rate limiting"],
            },
        )
    )
    time.sleep(0.02)

    # 2d. Active Behavioral Session
    practice_store.save(
        PracticeSession(
            id="prac_beh_active",
            created_at_iso="2026-11-21T16:50:00+00:00",
            role_title="Staff Backend Engineer — Core Payments Ledger",
            company="Pied Piper",
            config=PracticeConfig(
                interview_type="behavioral",
                question_source="mixed",
                target_level="L6",
                coach_style="supportive",
                live_coach_nudges=True,
                voice_enabled=True,
                question_count=3,
            ),
            phase="active",
            current_question_index=1,
            questions=[
                PracticeQuestionItem(
                    id="pq_b1",
                    bank_question_id="beh_own_01",
                    source="tailored",
                    category="behavioral",
                    title="Idempotency Replay Middleware",
                    competency="Ownership & 0-to-1",
                    prompt="Tell me about an ambiguous reliability gap you took from 0 to 1.",
                    status="completed",
                    follow_up_count=1,
                    carl_signals={"context": "strong", "action": "strong", "result": "strong", "learning": "strong"},
                ),
                PracticeQuestionItem(
                    id="pq_b2",
                    bank_question_id="beh_inf_01",
                    source="bank",
                    category="behavioral",
                    title="Cross-Team Technical Alignment",
                    competency="Influence Without Authority",
                    prompt="Tell me about a time you drove cross-team alignment when partner teams had competing priorities.",
                    status="active",
                    follow_up_count=1,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
                PracticeQuestionItem(
                    id="pq_b3",
                    bank_question_id="beh_fail_01",
                    source="bank",
                    category="behavioral",
                    title="Production Incident & Recovery",
                    competency="Failure & Recovery",
                    prompt="Walk me through a meaningful technical mistake you owned.",
                    status="pending",
                ),
            ],
            turns=[
                PracticeTurn(
                    id="pt_b1",
                    role="interviewer",
                    question_id="pq_b2",
                    text="Tell me about a time you had to drive cross-team alignment on a technical direction.",
                    timestamp_ms=now_ms - 600_000,
                ),
                PracticeTurn(
                    id="pt_b2",
                    role="candidate",
                    question_id="pq_b2",
                    text="I audited three teams' retry handlers and built a 2-day drop-in adapter so all three adopted deterministic idempotency.",
                    timestamp_ms=now_ms - 520_000,
                    carl_signals={"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                ),
            ],
        )
    )

    # 3. Past Copilot Session
    sess_store = SessionStore(data_dir / "sessions")
    sess_store.save(
        InterviewSession(
            id="20261119_143000_fin1",
            created_at_iso="2026-11-19T14:30:00-0800",
            config=SessionConfig(
                role_title="Senior Backend Engineer (L5)",
                interview_type="system_design",
                mode="remote_meet",
            ),
            utterances=[
                Utterance(
                    id="utt_1",
                    speaker="interviewer",
                    text="Let's design a webhook delivery service with retries and dead-letter queues.",
                    start_ms=0,
                    end_ms=4200,
                ),
                Utterance(
                    id="utt_2",
                    speaker="candidate",
                    text="I'd separate ingestion from delivery workers using partitioned queues and exponential backoff.",
                    start_ms=4800,
                    end_ms=11800,
                ),
            ],
            qa_groups=[
                QAGroup(
                    id="qa_1",
                    index=1,
                    card_kind="phase",
                    title="Webhook Ingestion & Retry Queue",
                    main_question="Design a webhook delivery service with retries and dead-letter queues.",
                    utterance_ids=["utt_1", "utt_2"],
                    started_ms=0,
                    updated_ms=11800,
                )
            ],
            calibration_markdown="## Summary\nSolid queueing and backoff instincts.",
        )
    )


class CDPClient:
    def __init__(self, ws_url: str, screenshot_dir: Path | None = None) -> None:
        self.ws_url = ws_url
        self.screenshot_dir = screenshot_dir
        self._id = 0
        self.ws: Any = None
        self.console_errors: list[str] = []
        self.exceptions: list[str] = []
        self.http_errors: list[dict[str, Any]] = []
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> CDPClient:
        self.ws = await websockets.connect(self.ws_url, max_size=50 * 1024 * 1024)
        self._reader_task = asyncio.create_task(self._read_loop())
        await self.send("Page.enable")
        await self.send("Runtime.enable")
        await self.send("Log.enable")
        await self.send("Network.enable")
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _read_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    method = msg.get("method", "")
                    params = msg.get("params") or {}
                    if method == "Runtime.exceptionThrown":
                        details = params.get("exceptionDetails") or {}
                        exc = details.get("exception") or {}
                        self.exceptions.append(str(exc.get("description") or details.get("text") or details))
                    elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
                        parts = [str(a.get("value") or a.get("description") or "") for a in (params.get("args") or [])]
                        self.console_errors.append(" ".join(parts).strip())
                    elif method == "Network.responseReceived":
                        resp = params.get("response") or {}
                        status = int(resp.get("status") or 0)
                        if status >= 400:
                            self.http_errors.append({"status": status, "url": str(resp.get("url") or "")})
        except asyncio.CancelledError:
            pass

    async def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        mid = self._id
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        res = await asyncio.wait_for(fut, timeout=20.0)
        if "error" in res:
            raise RuntimeError(f"CDP {method} error: {res['error']}")
        return res.get("result") or {}

    async def eval(self, expr: str) -> Any:
        res = await self.send(
            "Runtime.evaluate",
            {"expression": expr, "awaitPromise": True, "returnByValue": True},
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JS eval error: {res['exceptionDetails']}")
        return (res.get("result") or {}).get("value")

    async def set_viewport(self, w: int, h: int, dpr: float = 2.0) -> None:
        await self.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": w, "height": h, "deviceScaleFactor": dpr, "mobile": False},
        )

    async def screenshot(self, filename: str) -> Path | None:
        if self.screenshot_dir is None:
            return None
        data = await self.send("Page.captureScreenshot", {"format": "png"})
        out = self.screenshot_dir / filename
        out.write_bytes(base64.b64decode(data["data"]))
        return out

    async def navigate(self, url: str) -> None:
        await self.send("Page.navigate", {"url": url})
        await asyncio.sleep(0.9)

    async def press_key(self, key: str, code: str = "", modifiers: int = 0) -> None:
        params: dict[str, Any] = {"type": "keyDown", "key": key, "modifiers": modifiers}
        if code:
            params["code"] = code
        await self.send("Input.dispatchKeyEvent", params)
        params["type"] = "keyUp"
        await self.send("Input.dispatchKeyEvent", params)

    async def check_invariants(self, label: str) -> dict[str, Any]:
        return await self.eval(
            f"""(() => {{
              const docW = document.documentElement.scrollWidth;
              const winW = window.innerWidth;
              const overflowX = docW > winW + 2;
              const clippedButtons = [];
              document.querySelectorAll('button:not([hidden]), a.btn:not([hidden])').forEach(el => {{
                const st = getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return;
                const r = el.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) return;
                if (r.right > winW + 3 && !el.closest('[style*="overflow"]')) {{
                  clippedButtons.push({{id: el.id || '', text: (el.innerText || '').trim().slice(0, 30), right: Math.round(r.right), winW}});
                }}
              }});
              const toastEl = document.getElementById('toastBanner');
              const compEl = document.getElementById('practiceComposerWrap');
              let toastCollision = false;
              if (toastEl && !toastEl.classList.contains('hidden') && compEl) {{
                const cst = getComputedStyle(compEl);
                const tr = toastEl.getBoundingClientRect();
                const cr = compEl.getBoundingClientRect();
                if (cst.display !== 'none' && cr.height > 0 && tr.height > 0) {{
                  toastCollision = !(tr.bottom < cr.top - 4 || tr.top > cr.bottom + 4);
                }}
              }}
              return {{ label: {json.dumps(label)}, docW, winW, overflowX, clippedButtons: clippedButtons.slice(0, 8), toastCollision }};
            }})()"""
        )


def start_mockery_server(data_dir: Path, port: int) -> subprocess.Popen[bytes]:
    mockery_bin = Path(sys.executable).parent / "mockery"
    env = dict(os.environ)
    env["MOCKERY_NO_BROWSER"] = "1"
    proc = subprocess.Popen(
        [str(mockery_bin), "--host", "127.0.0.1", "--port", str(port), "--data-dir", str(data_dir), "--no-browser"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=1.0)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"Mockery server on port {port} failed to start")


async def launch_chrome() -> tuple[asyncio.subprocess.Process, Path, str]:
    cdp_port = find_free_port()
    profile = Path(tempfile.mkdtemp(prefix="mockery_cdp_chrome_"))
    proc = await asyncio.create_subprocess_exec(
        find_chrome(),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        f"--remote-debugging-port={cdp_port}",
        "--window-size=1440,900",
        "about:blank",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    ws_url = None
    async with httpx.AsyncClient() as client:
        for _ in range(60):
            try:
                r = await client.get(f"http://127.0.0.1:{cdp_port}/json/list", timeout=1.0)
                for t in r.json():
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        ws_url = t["webSocketDebuggerUrl"]
                        break
                if ws_url:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.15)
    if not ws_url:
        proc.terminate()
        raise RuntimeError("Could not obtain Chrome CDP WebSocket URL")
    return proc, profile, ws_url


async def run_uat(out_dir: Path | None) -> int:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")

    chrome_proc, chrome_profile, ws_url = await launch_chrome()
    srv_empty = None
    srv_seeded = None
    empty_dir = Path(tempfile.mkdtemp(prefix="mockery_uat_empty_"))
    seeded_dir = Path(tempfile.mkdtemp(prefix="mockery_uat_seeded_"))

    try:
        # SUITE 1: Cold empty first-run
        port_empty = find_free_port()
        srv_empty = start_mockery_server(empty_dir, port_empty)
        base_empty = f"http://127.0.0.1:{port_empty}"

        async with CDPClient(ws_url, out_dir) as cdp:
            await cdp.set_viewport(1440, 900)
            await cdp.navigate(base_empty)
            await asyncio.sleep(0.6)

            ob_state = await cdp.eval(
                """(() => {
                  const m = document.getElementById('onboardingModal');
                  const open = m && !m.classList.contains('hidden') && getComputedStyle(m).display !== 'none';
                  return { open, title: document.getElementById('onboardingModalTitle')?.innerText || '' };
                })()"""
            )
            await cdp.screenshot("01_empty_first_run_onboarding.png")
            record("first_run_onboarding_visible", bool(ob_state.get("open")), str(ob_state))

            await cdp.eval("document.getElementById('onboardSkipBtn')?.click() || document.getElementById('closeOnboardingBtn')?.click()")
            await asyncio.sleep(0.35)
            ob_closed = await cdp.eval(
                "(() => { const m = document.getElementById('onboardingModal'); return !m || m.classList.contains('hidden') || getComputedStyle(m).display === 'none'; })()"
            )
            record("first_run_skip_to_empty_home", bool(ob_closed), f"obClosed={ob_closed}")

            await cdp.eval("document.getElementById('profileBtn')?.click()")
            await asyncio.sleep(0.35)
            prof_empty = await cdp.eval(
                """(() => {
                  const m = document.getElementById('profileModal');
                  const open = m && !m.classList.contains('hidden') && getComputedStyle(m).display !== 'none';
                  const dropVis = !document.getElementById('resumeEmptyState')?.classList.contains('hidden');
                  return { open, dropVis };
                })()"""
            )
            record("empty_profile_shows_resume_dropzone", bool(prof_empty.get("open") and prof_empty.get("dropVis")), str(prof_empty))

            await cdp.press_key("Escape", "Escape")
            await asyncio.sleep(0.25)
            prof_closed = await cdp.eval(
                "(() => { const m = document.getElementById('profileModal'); return !m || m.classList.contains('hidden') || getComputedStyle(m).display === 'none'; })()"
            )
            record("escape_closes_profile_modal", bool(prof_closed), f"closed={prof_closed}")

            await cdp.eval("document.documentElement.setAttribute('data-theme', 'light')")
            await cdp.set_viewport(860, 900)
            await asyncio.sleep(0.25)
            lay_empty = await cdp.check_invariants("empty_home_860x900_light")
            record(
                "narrow_860x900_home_no_overflow",
                not lay_empty["overflowX"] and not lay_empty["clippedButtons"],
                str(lay_empty),
            )
            record("suite1_zero_js_exceptions", len(cdp.exceptions) == 0, f"exceptions={cdp.exceptions}")

        srv_empty.terminate()
        srv_empty.wait(timeout=5)
        srv_empty = None

        # SUITE 2: Seeded studios + adversarial offline/mic/physics checks
        seed_demo_data_dir(seeded_dir, offline_llm=True)
        port_seeded = find_free_port()
        srv_seeded = start_mockery_server(seeded_dir, port_seeded)
        base_seeded = f"http://127.0.0.1:{port_seeded}"

        async with CDPClient(ws_url, out_dir) as cdp:
            await cdp.set_viewport(1440, 900)
            await cdp.navigate(base_seeded)
            await asyncio.sleep(0.8)

            role_test = await cdp.eval(
                """(() => {
                  const el = document.getElementById('setupRoleInput');
                  if (!el) return { ok: false };
                  el.focus();
                  el.value = 'Principal Systems Engineer\\nPlatform Core';
                  el.dispatchEvent(new Event('input', { bubbles: true }));
                  return { tag: el.tagName, value: el.value, hasNewline: el.value.includes('\\n') };
                })()"""
            )
            record(
                "home_role_textarea_strips_newlines",
                role_test.get("tag") == "TEXTAREA" and not role_test.get("hasNewline"),
                str(role_test),
            )

            # Profile redaction live toggle
            await cdp.eval("document.getElementById('profileBtn')?.click()")
            await asyncio.sleep(0.35)
            red_before = await cdp.eval("document.getElementById('resumeRedactedPreview')?.innerText || ''")
            await cdp.eval("document.querySelector('#resumePiiChips button, #resumePiiChips .pii-chip')?.click()")
            await asyncio.sleep(0.45)
            red_after = await cdp.eval("document.getElementById('resumeRedactedPreview')?.innerText || ''")
            await cdp.eval("document.querySelector('#resumePiiChips button, #resumePiiChips .pii-chip')?.click()")
            await asyncio.sleep(0.25)
            await cdp.press_key("Escape", "Escape")
            await asyncio.sleep(0.25)
            record("profile_redaction_chip_toggles_preview", bool(red_before) and red_before != red_after, f"changed={red_before != red_after}")

            # Behavioral: Mic denied + toast clearance + offline LLM recovery
            await cdp.eval("""document.querySelector('[data-open-practice="prac_beh_active"]')?.click()""")
            await asyncio.sleep(0.8)
            await cdp.eval(
                """(() => {
                  if (!navigator.mediaDevices) navigator.mediaDevices = {};
                  navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'));
                })()"""
            )
            dock_probe = """(() => {
              const box = document.getElementById('practiceAnswerInput');
              return {
                voiceFirst: !!document.getElementById('practiceComposerWrap')?.classList.contains('is-voice-first'),
                boxShown: !!(box && box.offsetParent),
              };
            })()"""
            dock_before = await cdp.eval(dock_probe)
            record(
                "hands_free_dock_is_mic_only",
                dock_before["voiceFirst"] and not dock_before["boxShown"],
                str(dock_before),
            )
            await cdp.eval("document.getElementById('practiceMicBtn')?.click()")
            await asyncio.sleep(0.5)
            dock_denied = await cdp.eval(dock_probe)
            record(
                "mic_denied_brings_back_typing_box",
                dock_denied["boxShown"] and not dock_denied["voiceFirst"],
                str(dock_denied),
            )
            mic_inv = await cdp.check_invariants("mic_denied")
            mic_state = await cdp.eval(
                """(() => {
                  const btn = document.getElementById('practiceMicBtn');
                  const status = document.getElementById('practiceMicStatus')?.innerText || '';
                  const isRec = btn?.classList.contains('recording') || btn?.getAttribute('aria-pressed') === 'true';
                  return { isRec, status };
                })()"""
            )
            record("mic_denied_graceful_recovery", (not mic_state.get("isRec")) and bool(mic_state.get("status")), str(mic_state))
            record("toast_clears_bottom_composer", not mic_inv.get("toastCollision"), str(mic_inv))

            await cdp.eval("document.getElementById('practiceHintBtn')?.click()")
            for _ in range(25):
                if not await cdp.eval("document.getElementById('practiceHintBtn')?.disabled || false"):
                    break
                await asyncio.sleep(0.25)
            turns_before = await cdp.eval("document.querySelectorAll('#practiceFeed .pturn').length")
            await cdp.eval(
                """(() => {
                  const inp = document.getElementById('practiceAnswerInput');
                  inp.value = 'We cut duplicate charges by 96% with deterministic idempotency keys.';
                  inp.dispatchEvent(new Event('input', { bubbles: true }));
                  document.getElementById('practiceSendBtn')?.click();
                })()"""
            )
            for _ in range(25):
                if not await cdp.eval("document.getElementById('practiceSendBtn')?.disabled || false"):
                    break
                await asyncio.sleep(0.3)
            offline_state = await cdp.eval(
                """(() => ({
                  sendDisabled: !!document.getElementById('practiceSendBtn')?.disabled,
                  inputDisabled: !!document.getElementById('practiceAnswerInput')?.disabled,
                  turnCount: document.querySelectorAll('#practiceFeed .pturn').length,
                }))()"""
            )
            record(
                "llm_offline_recovers_without_stuck_composer",
                not offline_state["sendDisabled"] and not offline_state["inputDisabled"] and offline_state["turnCount"] > turns_before,
                f"before={turns_before}, after={offline_state}",
            )

            # Coding studio: cross-session mic reset, Tab/Esc-Tab, failing test, Python tab, 1024x768
            await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
            await asyncio.sleep(0.45)
            await cdp.eval("""document.querySelector('[data-open-practice="prac_code_active"]')?.click()""")
            await asyncio.sleep(0.8)
            mic_after = await cdp.eval("document.getElementById('practiceMicStatus')?.innerText || ''")
            record("mic_error_resets_across_sessions", mic_after == "Transcribed privately on this Mac", f"status={mic_after!r}")

            await cdp.eval(
                """(() => {
                  const ta = document.querySelector('#practiceStudioSlot textarea');
                  ta.focus(); ta.selectionStart = ta.selectionEnd = 0;
                })()"""
            )
            await cdp.press_key("Tab", "Tab")
            await asyncio.sleep(0.1)
            starts_space = await cdp.eval("document.querySelector('#practiceStudioSlot textarea')?.value.startsWith('  ')")
            await cdp.press_key("Escape", "Escape")
            await asyncio.sleep(0.1)
            await cdp.press_key("Tab", "Tab")
            await asyncio.sleep(0.1)
            escaped = await cdp.eval("document.activeElement !== document.querySelector('#practiceStudioSlot textarea')")
            record("coding_editor_tab_indents_and_esc_tab_escapes", bool(starts_space and escaped), f"indent={starts_space}, escaped={escaped}")

            await cdp.eval(
                """(() => {
                  const ta = document.querySelector('#practiceStudioSlot textarea');
                  ta.value = 'class TTLLRUCache { constructor() {} get() { return null; } put() {} }';
                  ta.dispatchEvent(new Event('input', { bubbles: true }));
                  const runBtn = [...document.querySelectorAll('#practiceStudioSlot button')].find(b => /Run Tests/i.test(b.innerText));
                  runBtn?.click();
                })()"""
            )
            await asyncio.sleep(0.9)
            fail_res = await cdp.eval(
                r"""(() => {
                  const slot = document.getElementById('practiceStudioSlot');
                  const txt = slot?.innerText || '';
                  const runBtn = [...slot.querySelectorAll('button')].find(b => /Run Tests/i.test(b.innerText));
                  return { hasFail: /fail|assert|error|0\//i.test(txt), runDisabled: !!runBtn?.disabled };
                })()"""
            )
            record("coding_failing_tests_surface_cleanly", bool(fail_res["hasFail"] and not fail_res["runDisabled"]), str(fail_res))

            await cdp.eval("""document.querySelector('#practiceStudioSlot [data-cs-lang="python"]')?.click()""")
            await asyncio.sleep(0.4)
            py_ok = await cdp.eval("/python|pyodide|install|def /i.test(document.getElementById('practiceStudioSlot')?.innerText || '')")
            record("coding_python_tab_graceful_without_pack", bool(py_ok), f"mentionsPython={py_ok}")

            await cdp.eval("""document.querySelector('#practiceStudioSlot [data-cs-lang="javascript"]')?.click()""")
            await cdp.set_viewport(1024, 768)
            await asyncio.sleep(0.3)
            lay_code = await cdp.check_invariants("coding_1024x768")
            record("narrow_1024x768_coding_no_overflow", not lay_code["overflowX"] and not lay_code["clippedButtons"], str(lay_code))

            # System design whiteboard physics + persistence + collapse + narrow 1024x768
            await cdp.set_viewport(1440, 900)
            await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
            await asyncio.sleep(0.45)
            await cdp.eval("""document.querySelector('[data-open-practice="prac_sd_active"]')?.click()""")
            await asyncio.sleep(0.8)

            before_nodes = await cdp.eval("document.querySelectorAll('#practiceStudioSlot g.sd-node').length")
            await cdp.eval("""[...document.querySelectorAll('#practiceStudioSlot button')].find(b => b.innerText.trim() === 'SVC')?.click()""")
            await asyncio.sleep(0.35)
            after_add = await cdp.eval("document.querySelectorAll('#practiceStudioSlot g.sd-node').length")
            record("whiteboard_add_svc_node_increments_count", after_add == before_nodes + 1, f"{before_nodes}->{after_add}")

            url_before = await cdp.eval("window.location.href")
            await cdp.press_key("Enter", "Enter")
            await asyncio.sleep(0.12)
            await cdp.press_key("Backspace", "Backspace")
            await asyncio.sleep(0.3)
            after_del = await cdp.eval("document.querySelectorAll('#practiceStudioSlot g.sd-node').length")
            url_after = await cdp.eval("window.location.href")
            record("whiteboard_backspace_deletes_without_navigating", after_del == before_nodes and url_before == url_after, f"after_del={after_del}")

            await cdp.eval("""document.querySelector('#practiceStudioSlot [data-sd-undo]')?.click()""")
            await asyncio.sleep(0.35)
            after_undo = await cdp.eval("document.querySelectorAll('#practiceStudioSlot g.sd-node').length")
            record("whiteboard_undo_restores_deleted_node", after_undo == before_nodes + 1, f"after_undo={after_undo}")

            split_before = await cdp.eval("Math.round(document.getElementById('practiceActiveView')?.getBoundingClientRect().width || 0)")
            await cdp.eval(
                """(() => {
                  const h = document.getElementById('practiceSplitHandle');
                  if (!h) return;
                  const r = h.getBoundingClientRect();
                  const x1 = r.left + r.width / 2, y1 = r.top + r.height / 2;
                  h.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, pointerId: 1, button: 0, clientX: x1, clientY: y1 }));
                  h.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, cancelable: true, pointerId: 1, button: 0, clientX: x1 + 90, clientY: y1 }));
                  h.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true, pointerId: 1, button: 0, clientX: x1 + 90, clientY: y1 }));
                })()"""
            )
            await asyncio.sleep(0.3)
            split_after = await cdp.eval("Math.round(document.getElementById('practiceActiveView')?.getBoundingClientRect().width || 0)")
            record("split_handle_drag_resizes_conversation_pane", abs(split_after - split_before) >= 25, f"{split_before}->{split_after}")

            await asyncio.sleep(1.3)
            await cdp.navigate(base_seeded)
            await asyncio.sleep(0.7)
            await cdp.eval("""document.querySelector('[data-open-practice="prac_sd_active"]')?.click()""")
            await asyncio.sleep(0.8)
            reloaded_nodes = await cdp.eval("document.querySelectorAll('#practiceStudioSlot g.sd-node').length")
            reloaded_split = await cdp.eval("Math.round(document.getElementById('practiceActiveView')?.getBoundingClientRect().width || 0)")
            record(
                "whiteboard_and_split_persist_across_reload",
                reloaded_nodes == before_nodes + 1 and abs(reloaded_split - split_after) <= 10,
                f"nodes={reloaded_nodes}, split={reloaded_split} vs {split_after}",
            )

            await cdp.eval("document.getElementById('practiceConvToggleBtn')?.click()")
            await asyncio.sleep(0.35)
            collapsed = await cdp.eval(
                """(() => ({
                  collapsedAttr: document.getElementById('practiceWorkspace')?.hasAttribute('data-conv-collapsed'),
                  slotW: Math.round(document.getElementById('practiceStudioSlot')?.getBoundingClientRect().width || 0),
                }))()"""
            )
            record("split_conv_collapse_expands_whiteboard", bool(collapsed["collapsedAttr"]) and collapsed["slotW"] >= 950, str(collapsed))

            await cdp.eval("document.getElementById('practiceConvToggleBtn')?.click()")
            await cdp.set_viewport(1024, 768)
            await asyncio.sleep(0.35)
            lay_wb = await cdp.check_invariants("whiteboard_1024x768")
            record("narrow_1024x768_whiteboard_no_overflow", not lay_wb["overflowX"] and not lay_wb["clippedButtons"], str(lay_wb))
            record("suite2_zero_uncaught_js_exceptions", len(cdp.exceptions) == 0, f"exceptions={cdp.exceptions}")

    finally:
        if srv_empty:
            srv_empty.terminate()
        if srv_seeded:
            srv_seeded.terminate()
        chrome_proc.terminate()
        shutil.rmtree(chrome_profile, ignore_errors=True)
        shutil.rmtree(empty_dir, ignore_errors=True)
        shutil.rmtree(seeded_dir, ignore_errors=True)

    passed = sum(1 for c in checks if c["passed"])
    failed = len(checks) - passed
    print(f"\nUAT Summary: {passed}/{len(checks)} passed, {failed} failed")
    return 0 if failed == 0 else 1


async def run_explore(steps: int, seed: int) -> int:
    """Bombadil-style property-based UI explorer asserting universal UI invariants at every step."""
    rng = random.Random(seed)
    chrome_proc, chrome_profile, ws_url = await launch_chrome()
    seeded_dir = Path(tempfile.mkdtemp(prefix="mockery_explore_"))
    srv = None
    failures: list[str] = []
    trace: list[str] = []

    viewports = [(1440, 900), (1280, 820), (1024, 768), (900, 760)]
    sessions = ["prac_beh_active", "prac_sd_active", "prac_code_active", "prac_debrief_retry"]

    try:
        seed_demo_data_dir(seeded_dir, offline_llm=True)
        port = find_free_port()
        srv = start_mockery_server(seeded_dir, port)
        base_url = f"http://127.0.0.1:{port}"

        async with CDPClient(ws_url) as cdp:
            await cdp.set_viewport(1440, 900)
            await cdp.navigate(base_url)
            await asyncio.sleep(0.8)

            for step in range(1, steps + 1):
                action = rng.choice(
                    [
                        "open_session",
                        "go_home",
                        "toggle_theme",
                        "resize",
                        "open_profile",
                        "open_settings",
                        "press_escape",
                        "toggle_conv_rail",
                        "switch_home_mode",
                        "whiteboard_add_or_undo",
                        "coding_switch_lang",
                    ]
                )
                desc = action

                if action == "open_session":
                    sid = rng.choice(sessions)
                    desc = f"open_session({sid})"
                    await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
                    await asyncio.sleep(0.18)
                    await cdp.eval(f"""document.querySelector('[data-open-practice="{sid}"]')?.click()""")
                    await asyncio.sleep(0.45)
                elif action == "go_home":
                    await cdp.press_key("Escape", "Escape")
                    await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
                    await asyncio.sleep(0.25)
                elif action == "toggle_theme":
                    theme = rng.choice(["dark", "light"])
                    desc = f"theme({theme})"
                    await cdp.eval(f"document.documentElement.setAttribute('data-theme', {json.dumps(theme)})")
                    await asyncio.sleep(0.1)
                elif action == "resize":
                    w, h = rng.choice(viewports)
                    desc = f"resize({w}x{h})"
                    await cdp.set_viewport(w, h)
                    await asyncio.sleep(0.18)
                elif action == "open_profile":
                    await cdp.eval("document.getElementById('profileBtn')?.click() || document.getElementById('practiceProfileBtn')?.click()")
                    await asyncio.sleep(0.25)
                elif action == "open_settings":
                    await cdp.eval("document.getElementById('settingsBtn')?.click() || document.getElementById('practiceSettingsBtn')?.click()")
                    await asyncio.sleep(0.25)
                elif action == "press_escape":
                    await cdp.press_key("Escape", "Escape")
                    await asyncio.sleep(0.15)
                elif action == "toggle_conv_rail":
                    await cdp.eval("document.getElementById('practiceConvToggleBtn')?.click()")
                    await asyncio.sleep(0.2)
                elif action == "switch_home_mode":
                    await cdp.eval(
                        """(() => {
                          const btns = [...document.querySelectorAll('#homeModeSwitch button')];
                          const inactive = btns.find(b => b.getAttribute('aria-selected') !== 'true');
                          inactive?.click();
                        })()"""
                    )
                    await asyncio.sleep(0.18)
                elif action == "whiteboard_add_or_undo":
                    await cdp.eval(
                        """(() => {
                          const btn = [...document.querySelectorAll('#practiceStudioSlot button')].find(b => b.innerText.trim() === 'SVC');
                          if (btn) btn.click();
                        })()"""
                    )
                    await asyncio.sleep(0.2)
                elif action == "coding_switch_lang":
                    lang = rng.choice(["javascript", "python", "pseudocode"])
                    desc = f"coding_lang({lang})"
                    await cdp.eval(f"""document.querySelector('#practiceStudioSlot [data-cs-lang="{lang}"]')?.click()""")
                    await asyncio.sleep(0.2)

                trace.append(f"{step:02d}:{desc}")
                inv = await cdp.check_invariants(desc)
                if inv["overflowX"]:
                    failures.append(f"step {step} ({desc}): horizontal overflow docW={inv['docW']} > winW={inv['winW']}")
                if inv["clippedButtons"]:
                    failures.append(f"step {step} ({desc}): clipped buttons {inv['clippedButtons']}")
                if inv["toastCollision"]:
                    failures.append(f"step {step} ({desc}): toast overlaps bottom practice composer")
                if cdp.exceptions:
                    failures.append(f"step {step} ({desc}): uncaught JS exception {cdp.exceptions[-1]}")
                    break

    finally:
        if srv:
            srv.terminate()
        chrome_proc.terminate()
        shutil.rmtree(chrome_profile, ignore_errors=True)
        shutil.rmtree(seeded_dir, ignore_errors=True)

    print(f"UI Explorer (seed={seed}, steps={len(trace)}): {'PASS' if not failures else 'FAIL'}")
    print("Trace:", " -> ".join(trace[:12]) + (" -> ..." if len(trace) > 12 else ""))
    if failures:
        for f in failures:
            print(f"  [INVARIANT VIOLATION] {f}", file=sys.stderr)
        print(f"Reproduce with: uv run python scripts/verify_ui.py explore --seed {seed} --steps {steps}", file=sys.stderr)
        return 1
    return 0


async def run_snapshots(out_dir: Path) -> int:
    """Capture a deterministic Dark + Light verification screenshot pack across all core screens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chrome_proc, chrome_profile, ws_url = await launch_chrome()
    seeded_dir = Path(tempfile.mkdtemp(prefix="mockery_snaps_"))
    srv = None
    captured: list[str] = []

    try:
        seed_demo_data_dir(seeded_dir, offline_llm=False)
        port = find_free_port()
        srv = start_mockery_server(seeded_dir, port)
        base_url = f"http://127.0.0.1:{port}"

        async with CDPClient(ws_url, out_dir) as cdp:
            await cdp.set_viewport(1440, 900)
            await cdp.navigate(base_url)
            await asyncio.sleep(0.9)

            for theme in ("dark", "light"):
                await cdp.eval(f"document.documentElement.setAttribute('data-theme', {json.dumps(theme)})")
                await asyncio.sleep(0.2)

                # 1. Home
                await cdp.press_key("Escape", "Escape")
                await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
                await asyncio.sleep(0.35)
                await cdp.screenshot(f"01_home_{theme}.png")
                captured.append(f"01_home_{theme}.png")

                # 2. Profile modal
                await cdp.eval("document.getElementById('profileBtn')?.click()")
                await asyncio.sleep(0.35)
                await cdp.screenshot(f"02_profile_{theme}.png")
                captured.append(f"02_profile_{theme}.png")
                await cdp.press_key("Escape", "Escape")
                await asyncio.sleep(0.2)

                # 3. Settings modal
                await cdp.eval("document.getElementById('settingsBtn')?.click()")
                await asyncio.sleep(0.35)
                await cdp.screenshot(f"03_settings_{theme}.png")
                captured.append(f"03_settings_{theme}.png")
                await cdp.press_key("Escape", "Escape")
                await asyncio.sleep(0.2)

                # 4. Behavioral studio
                await cdp.eval("""document.querySelector('[data-open-practice="prac_beh_active"]')?.click()""")
                await asyncio.sleep(0.7)
                await cdp.screenshot(f"04_behavioral_{theme}.png")
                captured.append(f"04_behavioral_{theme}.png")

                # 5. System design whiteboard
                await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
                await asyncio.sleep(0.3)
                await cdp.eval("""document.querySelector('[data-open-practice="prac_sd_active"]')?.click()""")
                await asyncio.sleep(0.7)
                await cdp.screenshot(f"05_whiteboard_{theme}.png")
                captured.append(f"05_whiteboard_{theme}.png")

                # 6. Coding studio
                await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
                await asyncio.sleep(0.3)
                await cdp.eval("""document.querySelector('[data-open-practice="prac_code_active"]')?.click()""")
                await asyncio.sleep(0.7)
                await cdp.screenshot(f"06_coding_{theme}.png")
                captured.append(f"06_coding_{theme}.png")

                # 7. Debrief & retry drill
                await cdp.eval("document.getElementById('practiceHomeBtn')?.click()")
                await asyncio.sleep(0.3)
                await cdp.eval("""document.querySelector('[data-open-practice="prac_debrief_retry"]')?.click()""")
                await asyncio.sleep(0.7)
                await cdp.screenshot(f"07_debrief_{theme}.png")
                captured.append(f"07_debrief_{theme}.png")

            if cdp.exceptions:
                print(f"Snapshot capture encountered JS exceptions: {cdp.exceptions}", file=sys.stderr)
                return 1
    finally:
        if srv:
            srv.terminate()
        chrome_proc.terminate()
        shutil.rmtree(chrome_profile, ignore_errors=True)
        shutil.rmtree(seeded_dir, ignore_errors=True)

    print(f"Captured {len(captured)} Dark + Light verification snapshots in {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Mockery headless Chrome CDP verification CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_uat = sub.add_parser("uat", help="Run deterministic 24-check Browser UAT suite.")
    p_uat.add_argument("--out-dir", type=Path, default=None, help="Optional directory to save UAT screenshots.")

    p_exp = sub.add_parser("explore", help="Run Bombadil-style property-based UI invariant explorer.")
    p_exp.add_argument("--steps", type=int, default=35, help="Number of randomized UI interaction steps.")
    p_exp.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    p_snap = sub.add_parser("snapshots", help="Capture deterministic Dark + Light verification screenshot pack.")
    p_snap.add_argument("--out-dir", type=Path, required=True, help="Directory where PNG snapshots will be written.")

    args = parser.parse_args()
    if args.cmd == "uat":
        return asyncio.run(run_uat(args.out_dir))
    if args.cmd == "explore":
        return asyncio.run(run_explore(args.steps, args.seed))
    if args.cmd == "snapshots":
        return asyncio.run(run_snapshots(args.out_dir))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
