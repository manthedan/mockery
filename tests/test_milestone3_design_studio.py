from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from mockery.server import create_app


def test_system_design_playbook_workspace_roundtrip_and_markdown(tmp_path: Path, monkeypatch):
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "none"
    runtime.llm_config.api_key = ""

    # Spy on LLM turn/hint/debrief calls to verify diagram_summary reaches the AI interviewer
    seen_diagrams: list[str] = []
    orig_reply = runtime.llm.generate_practice_interviewer_reply

    async def _spy_reply(**kwargs):
        ws = kwargs.get("workspace_context") or {}
        if ws.get("diagram_summary"):
            seen_diagrams.append(str(ws["diagram_summary"]))
        return await orig_reply(**kwargs)

    monkeypatch.setattr(runtime.llm, "generate_practice_interviewer_reply", _spy_reply)

    client = TestClient(app)

    # 1. Start a system_design session pinned to sd_ai_stream_01
    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Staff Full-Stack Engineer",
            "company": "Acme AI",
            "interview_type": "system_design",
            "question_source": "bank_only",
            "target_level": "L5",
            "selected_question_id": "sd_ai_stream_01",
        },
    )
    assert start_res.status_code == 200
    sess = start_res.json()["session"]
    sid = sess["id"]
    # Blank canvas default: no pre-populated diagram_graph on Turn 0
    assert not sess["workspace_context"].get("diagram_graph")
    assert len(sess["questions"]) == 1
    q0 = sess["questions"][0]
    assert q0["bank_question_id"] == "sd_ai_stream_01"
    # Verify playbook fields from system_design.json are cleanly extracted!
    assert len(q0["core_requirements"]) >= 3
    assert len(q0["deep_dives"]) >= 2

    # 2. Save whiteboard graph + summary via /workspace
    graph = {
        "version": 1,
        "nodes": [
            {"id": "n_cli", "type": "client", "label": "Web App", "note": "SSE client", "x": 36, "y": 120},
            {"id": "n_gw", "type": "gateway", "label": "Edge Gateway", "note": "Auth + Rate limit", "x": 240, "y": 120},
            {"id": "n_buf", "type": "queue", "label": "Redis Streams", "note": "Token resume buffer", "x": 468, "y": 120},
        ],
        "edges": [
            {"id": "e_1", "from": "n_cli", "to": "n_gw", "label": "HTTPS / SSE"},
            {"id": "e_2", "from": "n_gw", "to": "n_buf", "label": "Resume offset"},
        ],
    }
    summary = (
        "System Design Whiteboard (3 components, 2 connections):\n"
        "Components:\n"
        "- [Client] Web App — SSE client\n"
        "- [API Gateway] Edge Gateway — Auth + Rate limit\n"
        "- [Queue] Redis Streams — Token resume buffer\n"
        "Data Flows:\n"
        "- Web App -> Edge Gateway (HTTPS / SSE)\n"
        "- Edge Gateway -> Redis Streams (Resume offset)"
    )

    notes = (
        "API: POST /v1/chats/{id}/messages -> SSE stream\n"
        "Schema: messages(chat_id, msg_id, parent_id, content) PK(chat_id, msg_id)\n"
        "Math: 50k concurrent streams * 20 tok/s = 1M tok/s fan-out"
    )
    ws_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={
            "workspace_context": {
                "diagram_graph": graph,
                "diagram_summary": summary,
                "design_notes": notes,
                "checked_requirements": [q0["core_requirements"][0]],
            }
        },
    )
    assert ws_res.status_code == 200

    # 3. Reopen saved session and confirm graph, design notes & checked requirements persist verbatim
    loaded = client.get(f"/api/practice/sessions/{sid}")
    assert loaded.status_code == 200
    loaded_ws = loaded.json()["session"]["workspace_context"]
    assert len(loaded_ws["diagram_graph"]["nodes"]) == 3
    assert len(loaded_ws["diagram_graph"]["edges"]) == 2
    assert loaded_ws["design_notes"] == notes
    assert loaded_ws["checked_requirements"] == [q0["core_requirements"][0]]

    # 4. Ask AI to review diagram via /turn -> diagram_summary flows into LLM
    turn_res = client.post(
        f"/api/practice/{sid}/turn",
        json={
            "text": "Please review my current System Design whiteboard diagram.",
            "expected_question_id": q0["id"],
        },
    )
    assert turn_res.status_code == 200
    assert seen_diagrams and "Redis Streams" in seen_diagrams[-1]

    # 5. Debrief + Markdown export includes whiteboard summary & design notes
    debrief_res = client.post(f"/api/practice/{sid}/debrief")
    assert debrief_res.status_code == 200
    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert "## System Design Whiteboard Summary" in md_res.text
    assert "Edge Gateway -> Redis Streams (Resume offset)" in md_res.text
    assert "## System Design Notes (API / Schema / Math)" in md_res.text
    assert "POST /v1/chats/{id}/messages" in md_res.text

    # 6. Verify system_design_studio add-on flag toggles cleanly
    off_res = client.post("/api/addons/modules", json={"modules": {"system_design_studio": False}})
    assert off_res.status_code == 200
    assert off_res.json()["modules_map"]["system_design_studio"] is False


def test_design_studio_js_pure_helpers_via_node():
    repo_root = Path(__file__).resolve().parents[1]
    script = r"""
import {
  normalizeDiagramGraph,
  serializeDiagramSummary,
  createStarterGraph,
  deriveDesignAllowedActions,
  routeEdge,
  placeEdgeLabel,
} from './src/mockery/static/design_studio.js';

// 0. Notes-only design rounds still permit Ask AI Review with an empty canvas
const emptyNoNotes = deriveDesignAllowedActions({ interaction: { mode: 'idle' }, nodeCount: 0, hasNotes: false });
const emptyWithNotes = deriveDesignAllowedActions({ interaction: { mode: 'idle' }, nodeCount: 0, hasNotes: true });
if (emptyNoNotes.includes('ask_ai_review')) throw new Error('Empty canvas without notes should not allow review');
if (!emptyWithNotes.includes('ask_ai_review')) throw new Error('Notes-only workspace should allow review');

// 0b. Studio notes limit matches backend API (16,000 chars) so 9,000+ char notes never truncate on load
import fs from 'node:fs';
const studioSrc = fs.readFileSync('./src/mockery/static/design_studio.js', 'utf8');
if (!studioSrc.includes('const MAX_DESIGN_NOTES_CHARS = 16000;')) {
  throw new Error('Expected MAX_DESIGN_NOTES_CHARS to match 16000 API limit');
}

// 1. Starter graph adapts to streaming AI question
const g1 = createStarterGraph({ title: 'Streaming AI Assistant', prompt: 'Design Claude.ai' });
if (g1.nodes.length < 4 || g1.edges.length < 4) throw new Error('Expected rich starter graph');

// 2. Serialization produces readable components & data flows
const summary = serializeDiagramSummary(g1);
if (!summary.includes('Components:') || !summary.includes('Data Flows:')) {
  throw new Error('Missing expected sections in diagram summary');
}

// 3. Normalization drops invalid/self edges, duplicate IDs, clamps out-of-bounds coords
const dirty = {
  nodes: [
    { id: 'a', type: 'unknown_type', label: '  Alpha  ', note: 'n1', x: -500, y: 9999 },
    { id: 'a', type: 'client', label: 'Duplicate ID' },
    { id: 'b', type: 'database', label: 'Beta', x: 300, y: 120 },
    { id: { bad: true }, type: 'service', label: { nested: 1 }, note: ['arr'], x: 'bad', y: null },
  ],
  edges: [
    { id: 'e1', from: 'a', to: 'a', label: 'self-loop' },
    { id: 'e2', from: 'a', to: 'missing', label: 'dangling' },
    { id: 'e3', from: 'a', to: 'b', label: 'SQL' },
    { id: 'e4', from: 'a', to: 'b', label: 'Duplicate route' },
  ],
};
const clean = normalizeDiagramGraph(dirty);
if (clean.nodes.length !== 3) throw new Error('Expected 3 normalized nodes (including coerced object)');
if (clean.nodes[0].type !== 'service') throw new Error('Unknown type should fallback to service');
if (clean.nodes[0].x < 12 || clean.nodes[0].y > 420) throw new Error('Coords not clamped');
if (clean.edges.length !== 1 || clean.edges[0].id !== 'e3') throw new Error('Invalid/duplicate edges not filtered');

// 4. Emoji-safe truncation & flow budget preservation + XML control-char stripping
const hugeNodes = [];
for (let i = 0; i < 40; i++) {
  hugeNodes.push({
    id: 'n' + i,
    type: 'service',
    label: 'Svc\u0000' + i + ' 🚀🔥',
    note: '🚀'.repeat(70) + '\u0007',
    x: 48,
    y: 48,
  });
}
const hugeSummary = serializeDiagramSummary({
  nodes: hugeNodes,
  edges: [{ id: 'ef', from: 'n0', to: 'n1', label: 'gRPC 🚀' }],
});
if (hugeSummary.length > 6000) throw new Error('Summary exceeded 6000 chars');
if (!hugeSummary.includes('Data Flows:') || !hugeSummary.includes('gRPC 🚀')) {
  throw new Error('Long component notes starved Data Flows section');
}
if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(hugeSummary)) {
  throw new Error('XML control chars leaked into summary');
}
// Check UTF-8 / lone surrogate integrity
for (let i = 0; i < hugeSummary.length; i++) {
  const c = hugeSummary.charCodeAt(i);
  if (c >= 0xd800 && c <= 0xdbff) {
    const next = hugeSummary.charCodeAt(i + 1);
    if (!(next >= 0xdc00 && next <= 0xdfff)) throw new Error('Unpaired high surrogate in summary');
    i++;
  } else if (c >= 0xdc00 && c <= 0xdfff) {
    throw new Error('Unpaired low surrogate in summary');
  }
}

// 5. Edge routing: arrows leave the source border, stop just short of the target border,
//    and label pills never sit on a box.
const NODE_W = 148, NODE_H = 58;
const rectOf = (n) => ({ x1: n.x, y1: n.y, x2: n.x + NODE_W, y2: n.y + NODE_H });
const onBorder = (p, r) =>
  p.x >= r.x1 && p.x <= r.x2 && p.y >= r.y1 && p.y <= r.y2 &&
  (p.x === r.x1 || p.x === r.x2 || p.y === r.y1 || p.y === r.y2);
const gapTo = (p, r) =>
  Math.hypot(Math.max(r.x1 - p.x, 0, p.x - r.x2), Math.max(r.y1 - p.y, 0, p.y - r.y2));
const layouts = [
  g1,
  createStarterGraph({ title: 'URL shortener' }),
  // Diagonal neighbours: the layout where labels used to bury their arrows.
  {
    nodes: [
      { id: 'gw', x: 216, y: 155 }, { id: 'cache', x: 216, y: 34 }, { id: 'svc', x: 420, y: 155 },
      { id: 'db', x: 636, y: 71 }, { id: 'mq', x: 636, y: 239 }, { id: 'ai', x: 756, y: 335 },
    ],
    edges: [
      { from: 'gw', to: 'cache' }, { from: 'gw', to: 'svc' }, { from: 'svc', to: 'gw' },
      { from: 'svc', to: 'db' }, { from: 'svc', to: 'mq' }, { from: 'mq', to: 'ai' },
    ],
  },
];
for (const layout of layouts) {
  const byId = new Map(layout.nodes.map((n) => [n.id, n]));
  const boxes = layout.nodes.map(rectOf);
  for (const e of layout.edges) {
    const a = byId.get(e.from);
    const b = byId.get(e.to);
    const paired = layout.edges.some((o) => o.from === e.to && o.to === e.from);
    const pts = routeEdge(a, b, { paired, obstacles: boxes });
    const tag = e.from + ' → ' + e.to + ' ' + JSON.stringify(pts);
    if (pts.length < 2 || pts.length > 3) throw new Error('Unexpected route shape: ' + tag);
    if (!onBorder(pts[0], rectOf(a))) throw new Error('Edge does not start on the source border: ' + tag);
    const gap = gapTo(pts[pts.length - 1], rectOf(b));
    if (!(gap > 0 && gap <= 6)) throw new Error('Arrow does not stop just outside the target border: ' + tag);
    const lbl = placeEdgeLabel(pts, 80, 17, { boxes });
    const pill = { x1: lbl.x - 40, y1: lbl.y - 8.5, x2: lbl.x + 40, y2: lbl.y + 8.5 };
    if (boxes.some((r) => pill.x1 < r.x2 && pill.x2 > r.x1 && pill.y1 < r.y2 && pill.y2 > r.y1)) {
      throw new Error('Label pill overlaps a box: ' + tag);
    }
  }
}
console.log(JSON.stringify({ ok: true, nodes: g1.nodes.length, edges: g1.edges.length, summaryLen: hugeSummary.length }));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True


def test_workspace_validation_markdown_fence_and_playbook_coercion(tmp_path: Path):
    import asyncio

    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "none"
    runtime.llm_config.api_key = ""

    # generate_practice_plan safely handles string/non-iterable playbook fields without splitting chars
    questions, _opening = asyncio.run(
        runtime.llm.generate_practice_plan(
            interview_type="system_design",
            question_source="bank_only",
            role_title="Engineer",
            company="Acme",
            target_level="L5",
            question_count=1,
            redacted_resume="",
            jd_text="",
            bank_candidates=[
                {
                    "id": "sd_custom",
                    "title": "Custom Design",
                    "prompt": "Design X",
                    "playbook": {
                        "core_requirements": "Single requirement string",
                        "deep_dives": 42,
                    },
                }
            ],
        )
    )
    assert questions[0]["core_requirements"] == ["Single requirement string"]
    assert questions[0]["deep_dives"] == []

    client = TestClient(app)

    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Engineer",
            "company": "Acme",
            "interview_type": "system_design",
            "question_source": "bank_only",
            "target_level": "L5",
            "selected_question_id": "sd_ai_stream_01",
        },
    )
    assert start_res.status_code == 200
    sid = start_res.json()["session"]["id"]

    # Invalid diagram_graph shapes rejected cleanly with 400
    bad_graph_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_graph": {"nodes": "not-a-list", "edges": []}}},
    )
    assert bad_graph_res.status_code == 400

    bad_type_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_graph": ["not-an-object"]}},
    )
    assert bad_type_res.status_code == 400

    # Invalid design_notes types or oversized payloads rejected cleanly with 400
    bad_notes_type = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"design_notes": {"not": "a string"}}},
    )
    assert bad_notes_type.status_code == 400

    bad_notes_len = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"design_notes": "x" * 16_001}},
    )
    assert bad_notes_len.status_code == 400

    # Dynamic Markdown fences prevent breakout when diagram_summary or design_notes contain ```
    tricky_summary = "Components:\n```\n<script>alert(1)</script>\n```\nData Flows:\n- A -> B"
    tricky_notes = "API:\n```\nPOST /v1/items\n```"
    ok_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_summary": tricky_summary, "design_notes": tricky_notes}},
    )
    assert ok_res.status_code == 200
    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert "````text\n" in md_res.text
    assert tricky_summary in md_res.text
    assert tricky_notes in md_res.text


def test_system_design_turn_budget_deep_dives_protocol_and_notes_deltas(tmp_path: Path):
    import asyncio

    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "openai"
    runtime.llm_config.api_key = "test-key"

    captured_messages: list[list[dict[str, str]]] = []

    async def _fake_complete_messages(messages, **_kwargs):
        captured_messages.append(list(messages))
        return json.dumps(
            {
                "interviewer_reply": "Assume 50k peak RPS and p99 under 200ms. How will you partition writes?",
                "should_advance": False,
                "carl_signals": {
                    "requirements": "strong",
                    "architecture": "partial",
                    "scale": "partial",
                    "tradeoffs": "missing",
                },
                "coach_nudge": {
                    "tone": "tip",
                    "message": "Good clarifying question on scale — now sketch the write path and partition key.",
                },
            }
        )

    runtime.llm._complete_messages = _fake_complete_messages  # type: ignore[method-assign]
    client = TestClient(app)

    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Senior Backend Engineer",
            "company": "Acme",
            "interview_type": "system_design",
            "question_source": "bank_only",
            "target_level": "L5",
            "selected_question_id": "sd_ai_stream_01",
        },
    )
    assert start_res.status_code == 200
    sess_data = start_res.json()["session"]
    sid = sess_data["id"]
    q0 = sess_data["questions"][0]

    # Turn 1 with design_notes: verify system prompt uses RAST (no CARL bleed-through),
    # Turn-1 plan includes deep_dives + SYSTEM_DESIGN_INTERVIEWER_RULES, and notes delta is sent.
    turn1 = client.post(
        f"/api/practice/{sid}/turn",
        json={
            "text": "Before I draw, what peak concurrent streams and p99 latency SLO should I design for?",
            "expected_question_id": q0["id"],
            "workspace_context": {
                "design_notes": "QPS target: TBD\nAPI: POST /v1/messages",
            },
        },
    )
    assert turn1.status_code == 200
    assert turn1.json()["reply"]["should_advance"] is False
    assert turn1.json()["session"]["questions"][0]["carl_signals"] == {
        "context": "strong",
        "action": "partial",
        "result": "partial",
        "learning": "missing",
    }

    sys_msg = captured_messages[-1][0]["content"]
    user_msg = captured_messages[-1][1]["content"]
    assert '"requirements": "strong|partial|missing"' in sys_msg
    assert "Requirements = " in sys_msg
    assert "Learning = " not in sys_msg
    assert "SYSTEM DESIGN CLARIFICATIONS, PACING & PUSHBACK" in sys_msg
    assert "Peak QPS?" in sys_msg
    assert "Deep-dive bottlenecks & failure modes to probe:" in user_msg
    assert q0["deep_dives"][0] in user_msg
    assert "LIVE SYSTEM DESIGN NOTES (API / SCHEMA / MATH):" in user_msg
    assert "POST /v1/messages" in user_msg
    assert "after 12 turns" in user_msg

    # Verify system_design does NOT force-advance at follow_up_count == 3 (old hard cap)
    session_obj = runtime.practice_store.get(sid)
    assert session_obj is not None
    session_obj.questions[0].follow_up_count = 5
    runtime.practice_store.save(session_obj)

    turn_mid = client.post(
        f"/api/practice/{sid}/turn",
        json={
            "text": "I'll shard the stream buffer by conversation_id to avoid cross-node fan-out.",
            "expected_question_id": q0["id"],
            "workspace_context": {"design_notes": ""},
        },
    )
    assert turn_mid.status_code == 200
    assert turn_mid.json()["reply"]["should_advance"] is False
    assert turn_mid.json()["session"]["phase"] == "active"
    # Clearing design_notes emits an explicit delta
    assert "LIVE SYSTEM DESIGN NOTES (API / SCHEMA / MATH):\n(cleared)" in captured_messages[-1][-1]["content"]

    # Verify hints and debrief receive design_notes too
    seen_hint_user: list[str] = []
    seen_debrief_user: list[str] = []

    async def _fake_complete_text(*, user: str, operation: str = "", **_kwargs) -> str:
        if operation == "practice_hint":
            seen_hint_user.append(user)
            return "Think about what happens when a client reconnects mid-stream."
        if operation == "practice_debrief":
            seen_debrief_user.append(user)
            return json.dumps(
                {
                    "overall_encouragement": "Solid design reasoning.",
                    "top_strengths": ["Clear streaming architecture."],
                    "carl_primer": "Requirements -> Architecture -> Scale -> Tradeoffs.",
                    "question_reviews": [
                        {
                            "question_id": q0["id"],
                            "title": q0["title"],
                            "prompt": q0["prompt"],
                            "carl": {
                                "requirements": "strong",
                                "architecture": "strong",
                                "scale": "partial",
                                "tradeoffs": "partial",
                            },
                            "strength": "Good scoping.",
                            "coaching_tip": "Quantify buffer retention.",
                            "carl_reframe": "Name the exact tradeoff on buffer TTL.",
                        }
                    ],
                    "retry_drill": {
                        "question_id": q0["id"],
                        "focus": "Defend One Bottleneck & Tradeoff (45s)",
                        "prompt": "Defend your reconnect buffer tradeoff in 45 seconds.",
                    },
                }
            )
        return "{}"

    runtime.llm._complete_text = _fake_complete_text  # type: ignore[method-assign]
    hint_res = client.post(
        f"/api/practice/{sid}/hint",
        json={"workspace_context": {"design_notes": "Buffer TTL = 5 min per stream"}},
    )
    assert hint_res.status_code == 200
    assert seen_hint_user and "Buffer TTL = 5 min per stream" in seen_hint_user[-1]

    debrief_res = client.post(f"/api/practice/{sid}/debrief", json={})
    assert debrief_res.status_code == 200
    assert seen_debrief_user and "FINAL DESIGN NOTES (API / SCHEMA / MATH):" in seen_debrief_user[-1]
    assert "Buffer TTL = 5 min per stream" in seen_debrief_user[-1]
    assert asyncio.iscoroutinefunction(runtime.llm.generate_practice_interviewer_reply)


def test_resumed_pre_upgrade_studio_conversation_reseeds_and_decodes_legacy_grades(tmp_path: Path):
    """P1 regression: pre-upgrade v1 CARL system prompt on resumed SD/coding sessions reseeds cleanly."""
    from mockery.practice_conversation import (
        PRACTICE_CONVERSATION_VERSION,
        PracticeConversation,
        PracticeConversationMessage,
    )
    from mockery.rubrics import CODING, SYSTEM_DESIGN

    # Legacy slot names still decode cleanly on studio rubrics without breaking STAR/GENERAL
    assert SYSTEM_DESIGN.signals_from_grader(
        {"context": "strong", "action": "partial", "result": "strong", "learning": "missing"}
    ) == {"context": "strong", "action": "partial", "result": "strong", "learning": "missing"}
    assert CODING.signals_from_grader(
        {"context": "strong", "action": "strong", "result": "partial", "learning": "strong"}
    ) == {"context": "strong", "action": "strong", "result": "partial", "learning": "strong"}

    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "openai"
    runtime.llm_config.api_key = "test-key"

    captured: list[list[dict[str, str]]] = []

    async def _fake_complete_messages(messages, **_kwargs):
        captured.append(list(messages))
        # Simulate a model emitting legacy keys on a resumed session -> still lands in meter
        return json.dumps(
            {
                "interviewer_reply": "Got it — assume 20k RPS. How do you partition the write path?",
                "should_advance": False,
                "carl_signals": {"context": "strong", "action": "strong", "result": "partial", "learning": "missing"},
                "coach_nudge": {"tone": "tip", "message": "Solid baseline — now explain hot-partition mitigation."},
            }
        )

    runtime.llm._complete_messages = _fake_complete_messages  # type: ignore[method-assign]
    client = TestClient(app)

    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Senior Backend Engineer",
            "company": "Acme",
            "interview_type": "system_design",
            "question_source": "bank_only",
            "target_level": "L5",
            "selected_question_id": "sd_ai_stream_01",
        },
    )
    assert start_res.status_code == 200
    sid = start_res.json()["session"]["id"]
    q0 = start_res.json()["session"]["questions"][0]

    # Inject a pre-upgrade v1 persisted conversation with an old CARL system prompt
    sess = runtime.practice_store.get(sid)
    assert sess is not None
    sess.llm_conversation = PracticeConversation(
        version=1,
        system='Legacy prompt asking for {"context": "strong|partial|missing", "learning": "strong|partial|missing"}',
        messages=[
            PracticeConversationMessage(role="user", content="Turn 1"),
            PracticeConversationMessage(role="assistant", content='{"interviewer_reply": "Tell me more"}'),
        ],
        last_synced_turn_count=len(sess.turns),
    )
    runtime.practice_store.save(sess)

    res = client.post(
        f"/api/practice/{sid}/turn",
        json={"text": "Peak QPS?", "expected_question_id": q0["id"]},
    )
    assert res.status_code == 200
    # Must reseed to v2 RAST system prompt with System Design protocol
    sent_system = captured[-1][0]["content"]
    assert '"requirements": "strong|partial|missing"' in sent_system
    assert "SYSTEM DESIGN CLARIFICATIONS, PACING & PUSHBACK" in sent_system
    # Legacy grades still decoded (not silently dropped to heuristic)
    assert res.json()["session"]["questions"][0]["carl_signals"] == {
        "context": "strong",
        "action": "strong",
        "result": "partial",
        "learning": "missing",
    }
    reloaded = runtime.practice_store.get(sid)
    assert reloaded is not None and reloaded.llm_conversation is not None
    assert reloaded.llm_conversation.version == PRACTICE_CONVERSATION_VERSION


def test_offline_system_design_answers_clarifications_and_runs_past_three_probes(tmp_path: Path):
    """P2 regression: offline bundled SD question clarifies scope, runs past probe 3, and honors hard cap."""
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "none"
    runtime.llm_config.api_key = ""
    client = TestClient(app)

    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Senior Backend Engineer",
            "company": "Acme",
            "interview_type": "system_design",
            "question_source": "bank_only",
            "target_level": "L5",
            "selected_question_id": "sd_ai_stream_01",
        },
    )
    assert start_res.status_code == 200
    sid = start_res.json()["session"]["id"]
    q0 = start_res.json()["session"]["questions"][0]

    # Turn 1: scoping clarification gets concrete assumptions and does NOT advance
    clarify = client.post(
        f"/api/practice/{sid}/turn",
        json={"text": "What peak QPS and p99 should I assume?", "expected_question_id": q0["id"]},
    )
    assert clarify.status_code == 200
    assert clarify.json()["reply"]["should_advance"] is False
    reply_txt = clarify.json()["reply"]["interviewer_reply"]
    assert "50k" in reply_txt and "200ms" in reply_txt

    # Turns 2-5: continues well past the old 3-probe cutoff
    for i in range(4):
        step = client.post(
            f"/api/practice/{sid}/turn",
            json={
                "text": f"Architecture iteration {i + 1}: adding edge gateway, Redis Streams buffer, and sharded DB.",
                "expected_question_id": q0["id"],
            },
        )
        assert step.status_code == 200
        assert step.json()["reply"]["should_advance"] is False
        assert step.json()["session"]["phase"] == "active"

    # Hard cap at follow_up_count == 12 still ends the round cleanly
    sess = runtime.practice_store.get(sid)
    assert sess is not None
    sess.questions[0].follow_up_count = 12
    runtime.practice_store.save(sess)
    final_turn = client.post(
        f"/api/practice/{sid}/turn",
        json={"text": "Final tradeoff summary on consistency vs latency.", "expected_question_id": q0["id"]},
    )
    assert final_turn.status_code == 200
    assert final_turn.json()["reply"]["should_advance"] is True
    assert final_turn.json()["session"]["phase"] == "awaiting_debrief"


def test_long_design_notes_preserve_tail_across_reload_export_deltas_hints_and_debrief(tmp_path: Path):
    """P2 regressions #3 and #4: 9k notes never silently truncate, and edits past char 3,000 reach the model."""
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "openai"
    runtime.llm_config.api_key = "test-key"

    seen_turn_users: list[str] = []
    seen_hint_users: list[str] = []
    seen_debrief_users: list[str] = []

    async def _fake_complete_messages(messages, **_kwargs):
        seen_turn_users.append(messages[-1]["content"])
        return json.dumps(
            {
                "interviewer_reply": "Thanks for adding the storage math. How do you shard hot tenants?",
                "should_advance": False,
                "carl_signals": {
                    "requirements": "strong",
                    "architecture": "strong",
                    "scale": "strong",
                    "tradeoffs": "partial",
                },
                "coach_nudge": {"tone": "encourage", "message": "Clear capacity math — now defend hot-key sharding."},
            }
        )

    async def _fake_complete_text(*, user: str, operation: str = "", **_kwargs) -> str:
        if operation == "practice_hint":
            seen_hint_users.append(user)
            return "Double-check write amplification on your hottest partition."
        if operation == "practice_debrief":
            seen_debrief_users.append(user)
            return json.dumps(
                {
                    "overall_encouragement": "Strong end-to-end architecture and capacity math.",
                    "top_strengths": ["Concrete storage sizing."],
                    "carl_primer": "Requirements -> Architecture -> Scale -> Tradeoffs.",
                    "question_reviews": [
                        {
                            "question_id": q0["id"],
                            "title": q0["title"],
                            "prompt": q0["prompt"],
                            "carl": {
                                "requirements": "strong",
                                "architecture": "strong",
                                "scale": "strong",
                                "tradeoffs": "partial",
                            },
                            "strength": "Detailed math.",
                            "coaching_tip": "Name one consistency tradeoff.",
                            "carl_reframe": "Explain the failover tradeoff cleanly.",
                        }
                    ],
                    "retry_drill": {
                        "question_id": q0["id"],
                        "focus": "Defend One Bottleneck & Tradeoff (45s)",
                        "prompt": "Defend your shard key tradeoff in 45s.",
                    },
                }
            )
        return "{}"

    runtime.llm._complete_messages = _fake_complete_messages  # type: ignore[method-assign]
    runtime.llm._complete_text = _fake_complete_text  # type: ignore[method-assign]
    client = TestClient(app)

    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Staff Systems Engineer",
            "company": "Acme",
            "interview_type": "system_design",
            "question_source": "bank_only",
            "target_level": "L6",
            "selected_question_id": "sd_ai_stream_01",
        },
    )
    assert start_res.status_code == 200
    sid = start_res.json()["session"]["id"]
    q0 = start_res.json()["session"]["questions"][0]

    # Save 9,000-char notes, then do an unrelated graph update and verify full 9,000 chars survive + export
    base_notes = "API: POST /v1/chat\n" + ("# schema-line\n" * 640)
    assert len(base_notes) > 8_500
    tail_calc = "CAPACITY_CALC_42PB_PER_YEAR_AT_120K_QPS"
    notes_v1 = base_notes
    notes_v2 = base_notes + "\n" + tail_calc
    assert len(notes_v2) > 9_000

    assert client.post(f"/api/practice/{sid}/workspace", json={"workspace_context": {"design_notes": notes_v1}}).status_code == 200
    # Unrelated graph edit leaves notes untouched
    assert client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_graph": {"version": 1, "nodes": [{"id": "n1", "type": "service", "label": "API", "note": "", "x": 40, "y": 40}], "edges": []}}},
    ).status_code == 200
    reloaded_ws = client.get(f"/api/practice/sessions/{sid}").json()["session"]["workspace_context"]
    assert reloaded_ws["design_notes"] == notes_v1

    # Turn 1 with base_notes (>3k), then Turn 2 appends a distinctive calculation after char 9,000
    assert client.post(f"/api/practice/{sid}/turn", json={"text": "Here is my initial schema.", "expected_question_id": q0["id"]}).status_code == 200
    assert client.post(
        f"/api/practice/{sid}/turn",
        json={"text": "Please review my capacity math.", "expected_question_id": q0["id"], "workspace_context": {"design_notes": notes_v2}},
    ).status_code == 200
    assert tail_calc in seen_turn_users[-1]

    # Hint, debrief, and Markdown export ALL see the trailing calculation past char 9,000
    assert client.post(f"/api/practice/{sid}/hint", json={}).status_code == 200
    assert seen_hint_users and tail_calc in seen_hint_users[-1]

    assert client.post(f"/api/practice/{sid}/debrief", json={}).status_code == 200
    assert seen_debrief_users and tail_calc in seen_debrief_users[-1]

    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert tail_calc in md_res.text


def test_oversized_custom_bank_deep_dives_stay_within_conversation_budget(tmp_path: Path):
    """P2 regression #7: a 120k-char custom deep_dive never overflows MAX_PRACTICE_CONVERSATION_CHARS or loops reseeds."""
    from mockery.practice import PracticeConfig, PracticeQuestionItem, PracticeSession
    from mockery.practice_conversation import (
        MAX_PRACTICE_CONVERSATION_CHARS,
        build_practice_turn_messages,
        commit_practice_conversation_turn,
    )

    huge_dive = "hot-key-skew " * 10_000  # ~130,000 chars
    q = PracticeQuestionItem(
        id="pq_huge",
        category="system_design",
        title="Custom Huge Bank Item",
        prompt="Design a global event router.",
        deep_dives=[huge_dive, "failover", "backpressure"],
        follow_ups=["How do you shard? " * 500],
        status="active",
    )
    sess = PracticeSession(
        role_title="Staff Engineer",
        company="Acme",
        config=PracticeConfig(interview_type="system_design", target_level="L6"),
        questions=[q],
    )

    msgs1, user1, ws1, sys1, reset1 = build_practice_turn_messages(
        sess,
        current_question=q.model_dump(),
        next_question=None,
        candidate_answer="Peak QPS?",
        follow_up_count=0,
        max_followups=12,
    )
    assert reset1 is True
    assert len(sys1) + len(user1) < MAX_PRACTICE_CONVERSATION_CHARS // 2
    assert len(msgs1) == 2

    commit_practice_conversation_turn(
        sess,
        system_prompt=sys1,
        user_content=user1,
        assistant_raw='{"interviewer_reply": "Assume 100k RPS."}',
        workspace_snapshot=ws1,
        synced_turn_count=2,
        reset_conversation=reset1,
    )

    # Turn 2 MUST append normally without triggering an overflow reseed
    msgs2, _user2, _ws2, _sys2, reset2 = build_practice_turn_messages(
        sess,
        current_question=q.model_dump(),
        next_question=None,
        candidate_answer="I'll start with an ingestion gateway and Kafka.",
        follow_up_count=1,
        max_followups=12,
    )
    assert reset2 is False
    assert len(msgs2) == 4


def test_eval_harness_supports_system_design_rast_and_held_out_clarification_case():
    """P2 regression #6: eval harness uses RAST schema, 12-turn SD budget, design_notes, and scores RAST outputs."""
    from evals.metrics import (
        build_user_prompt_for_example,
        load_jsonl_dataset,
        render_system_prompt_for_example,
        score_practice_interviewer_reply,
    )

    repo_root = Path(__file__).resolve().parents[1]
    rows = {r["id"]: r for r in load_jsonl_dataset(repo_root / "evals" / "datasets" / "practice_interviewer.jsonl")}
    ex = rows["pi_16_val_sysdesign_clarify_past_three"]
    assert ex["split"] == "val"
    assert ex["follow_up_count"] == 4

    sys_p = render_system_prompt_for_example("practice_interviewer", ex)
    usr_p = build_user_prompt_for_example("practice_interviewer", ex)
    assert '"requirements": "strong|partial|missing"' in sys_p
    assert "SYSTEM DESIGN CLARIFICATIONS, PACING & PUSHBACK" in sys_p
    assert "after 12 turns" in usr_p
    assert "LIVE SYSTEM DESIGN NOTES (API / SCHEMA / MATH):" in usr_p
    assert "Partition key: tenant_id" in usr_p

    score, info = score_practice_interviewer_reply(ex["reference_response"], ex)
    assert score >= 0.95, f"Expected RAST clarification reference response to score high, got {score}: {info}"

