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

    ws_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={
            "workspace_context": {
                "diagram_graph": graph,
                "diagram_summary": summary,
                "checked_requirements": [q0["core_requirements"][0]],
            }
        },
    )
    assert ws_res.status_code == 200

    # 3. Reopen saved session and confirm graph & checked requirements persist verbatim
    loaded = client.get(f"/api/practice/sessions/{sid}")
    assert loaded.status_code == 200
    loaded_ws = loaded.json()["session"]["workspace_context"]
    assert len(loaded_ws["diagram_graph"]["nodes"]) == 3
    assert len(loaded_ws["diagram_graph"]["edges"]) == 2
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

    # 5. Debrief + Markdown export includes whiteboard summary
    debrief_res = client.post(f"/api/practice/{sid}/debrief")
    assert debrief_res.status_code == 200
    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert "## System Design Whiteboard Summary" in md_res.text
    assert "Edge Gateway -> Redis Streams (Resume offset)" in md_res.text

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
} from './src/mockery/static/design_studio.js';

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

    # Dynamic Markdown fences prevent breakout when diagram_summary or code contains ```
    tricky_summary = "Components:\n```\n<script>alert(1)</script>\n```\nData Flows:\n- A -> B"
    ok_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"diagram_summary": tricky_summary}},
    )
    assert ok_res.status_code == 200
    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert "````text\n" in md_res.text
    assert tricky_summary in md_res.text

