/* System Design Whiteboard Studio (Milestone 3)
 * 100% offline SVG architecture canvas registered as a Practice Studio plugin.
 * Ink & Paper aesthetic in Light & Dark modes, zero CDN dependencies.
 */

import { createMachine } from "./state_machines.js";

const CANVAS_W = 920;
const CANVAS_H = 420;
const NODE_W = 148;
const NODE_H = 58;
const GRID_STEP = 12;
const MAX_NODES = 50;
const MAX_EDGES = 100;
const MAX_HISTORY = 40;
const MAX_SUMMARY_COMP_CHARS = 3400;
const MAX_SUMMARY_FLOW_CHARS = 2400;

export const NODE_TYPES = [
  { type: "client", label: "Client", badge: "CLI", defaultTitle: "Web / Mobile Client", defaultNote: "Browser / App" },
  { type: "gateway", label: "API Gateway", badge: "GW", defaultTitle: "API Gateway / LB", defaultNote: "Auth · Rate Limit" },
  { type: "service", label: "Service", badge: "SVC", defaultTitle: "Core Service", defaultNote: "Stateless API" },
  { type: "database", label: "Database", badge: "DB", defaultTitle: "Primary DB", defaultNote: "Postgres / Sharded" },
  { type: "cache", label: "Cache", badge: "CACHE", defaultTitle: "Redis Cache", defaultNote: "Sub-ms reads" },
  { type: "queue", label: "Queue", badge: "MQ", defaultTitle: "Event Stream", defaultNote: "Kafka / Streams" },
  { type: "storage", label: "Storage / CDN", badge: "CDN", defaultTitle: "Object Store / CDN", defaultNote: "S3 + Edge" },
  { type: "worker", label: "Worker / LLM", badge: "AI", defaultTitle: "Async Worker", defaultNote: "Inference / Jobs" },
  { type: "note", label: "Note", badge: "NOTE", defaultTitle: "Capacity / Trade-off", defaultNote: "10k RPS · p99 < 100ms" },
];

const NODE_TYPE_MAP = new Map(NODE_TYPES.map((t) => [t.type, t]));

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function snap(v) {
  return Math.round(v / GRID_STEP) * GRID_STEP;
}

function shortLocalId(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
}

function stripInvalidXmlAndSurrogates(input) {
  const s = typeof input === "string" ? input : "";
  if (!s) return "";
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    // Drop XML 1.0 illegal control characters and non-characters
    if (
      code <= 0x08 ||
      code === 0x0b ||
      code === 0x0c ||
      (code >= 0x0e && code <= 0x1f) ||
      code === 0xfffe ||
      code === 0xffff
    ) {
      continue;
    }
    // High surrogate: keep only if followed by a valid low surrogate
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = i + 1 < s.length ? s.charCodeAt(i + 1) : 0;
      if (next >= 0xdc00 && next <= 0xdfff) {
        out += s[i] + s[i + 1];
        i += 1;
      }
      continue;
    }
    // Lone low surrogate: drop
    if (code >= 0xdc00 && code <= 0xdfff) {
      continue;
    }
    out += s[i];
  }
  return out;
}

function safeStr(val, fallback = "") {
  if (typeof val === "string") return stripInvalidXmlAndSurrogates(val);
  if (typeof val === "number" && Number.isFinite(val)) return String(val);
  if (typeof val === "boolean") return String(val);
  return fallback;
}

function safeNum(val, fallback) {
  if (typeof val === "number" && Number.isFinite(val)) return val;
  if (typeof val === "string") {
    const trimmed = val.trim();
    if (trimmed !== "") {
      const parsed = Number(trimmed);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return fallback;
}

function safeSliceChars(text, maxCodePoints) {
  const clean = stripInvalidXmlAndSurrogates(safeStr(text));
  const chars = Array.from(clean);
  if (chars.length <= maxCodePoints) return clean;
  return chars.slice(0, Math.max(0, maxCodePoints)).join("");
}

function xmlEscape(text) {
  return stripInvalidXmlAndSurrogates(safeStr(text))
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function truncateText(text, maxCodePoints) {
  const clean = stripInvalidXmlAndSurrogates(safeStr(text)).trim();
  const chars = Array.from(clean);
  if (chars.length <= maxCodePoints) return clean;
  return `${chars.slice(0, Math.max(1, maxCodePoints - 1)).join("")}…`;
}

export function normalizeDiagramGraph(raw) {
  const safe = { version: 1, nodes: [], edges: [] };
  if (!raw || typeof raw !== "object") return safe;

  const seenNodeIds = new Set();
  if (Array.isArray(raw.nodes)) {
    for (const n of raw.nodes) {
      if (!n || typeof n !== "object" || safe.nodes.length >= MAX_NODES) continue;
      const rawId = safeStr(n.id).trim();
      const id = safeSliceChars(rawId || shortLocalId("n"), 40);
      if (!id || seenNodeIds.has(id)) continue;
      seenNodeIds.add(id);
      const rawType = safeStr(n.type);
      const type = NODE_TYPE_MAP.has(rawType) ? rawType : "service";
      const meta = NODE_TYPE_MAP.get(type);
      const x = clamp(Math.round(safeNum(n.x, 60)), 12, CANVAS_W - NODE_W - 12);
      const y = clamp(Math.round(safeNum(n.y, 60)), 12, CANVAS_H - NODE_H - 12);
      const label = safeSliceChars(safeStr(n.label, meta.defaultTitle).trim(), 80) || meta.defaultTitle;
      const note = safeSliceChars(safeStr(n.note, "").trim(), 140);
      safe.nodes.push({ id, type, label, note, x, y });
    }
  }

  const seenEdgeIds = new Set();
  const seenPairs = new Set();
  if (Array.isArray(raw.edges)) {
    for (const e of raw.edges) {
      if (!e || typeof e !== "object" || safe.edges.length >= MAX_EDGES) continue;
      const from = safeStr(e.from);
      const to = safeStr(e.to);
      if (!seenNodeIds.has(from) || !seenNodeIds.has(to) || from === to) continue;
      const pairKey = `${from}->${to}`;
      if (seenPairs.has(pairKey)) continue;
      const rawId = safeStr(e.id).trim();
      const id = safeSliceChars(rawId || shortLocalId("e"), 40);
      if (!id || seenEdgeIds.has(id)) continue;
      seenEdgeIds.add(id);
      seenPairs.add(pairKey);
      safe.edges.push({
        id,
        from,
        to,
        label: safeSliceChars(safeStr(e.label, "").trim(), 60),
      });
    }
  }
  return safe;
}

function appendWithinBudget(lines, line, budget) {
  if (lines.used + line.length + 1 > budget) return false;
  lines.items.push(line);
  lines.used += line.length + 1;
  return true;
}

export function serializeDiagramSummary(graph) {
  const g = normalizeDiagramGraph(graph);
  if (g.nodes.length === 0) {
    return "System Design Whiteboard is currently empty (0 components).";
  }
  const nodeById = new Map(g.nodes.map((n) => [n.id, n]));
  const header = `System Design Whiteboard (${g.nodes.length} components, ${g.edges.length} connections):`;

  const compLines = { items: ["Components:"], used: 12 };
  let compOmitted = 0;
  for (const n of g.nodes) {
    const meta = NODE_TYPE_MAP.get(n.type) || NODE_TYPES[2];
    const notePart = n.note ? ` — ${n.note}` : "";
    const line = `- [${meta.label}] ${n.label}${notePart}`;
    if (!appendWithinBudget(compLines, line, MAX_SUMMARY_COMP_CHARS)) {
      compOmitted += 1;
    }
  }
  if (compOmitted > 0) {
    compLines.items.push(`- … (+${compOmitted} more component${compOmitted === 1 ? "" : "s"})`);
  }

  const out = [header, ...compLines.items];

  if (g.edges.length > 0) {
    const flowLines = { items: ["Data Flows:"], used: 12 };
    let flowOmitted = 0;
    for (const e of g.edges) {
      const a = nodeById.get(e.from);
      const b = nodeById.get(e.to);
      if (!a || !b) continue;
      const lbl = e.label ? ` (${e.label})` : "";
      const line = `- ${a.label} -> ${b.label}${lbl}`;
      if (!appendWithinBudget(flowLines, line, MAX_SUMMARY_FLOW_CHARS)) {
        flowOmitted += 1;
      }
    }
    if (flowOmitted > 0) {
      flowLines.items.push(`- … (+${flowOmitted} more flow${flowOmitted === 1 ? "" : "s"})`);
    }
    out.push(...flowLines.items);
  }
  return out.join("\n");
}

export function createStarterGraph(question = null) {
  const title = `${question?.title || ""} ${question?.prompt || ""}`.toLowerCase();
  if (title.includes("stream") || title.includes("assistant") || title.includes("claude") || title.includes("llm")) {
    return normalizeDiagramGraph({
      version: 1,
      nodes: [
        { id: "n_cli", type: "client", label: "Web Workspace", note: "SSE client + Artifact sandbox", x: 36, y: 168 },
        { id: "n_gw", type: "gateway", label: "Edge Gateway", note: "Auth · Rate limits · Sticky SSE", x: 240, y: 168 },
        { id: "n_orch", type: "service", label: "Chat Orchestrator", note: "Context assembly + Tool router", x: 444, y: 96 },
        { id: "n_llm", type: "worker", label: "Inference Pool", note: "Token streaming + KV affinity", x: 696, y: 60 },
        { id: "n_buf", type: "queue", label: "Stream Buffer", note: "Redis Streams (resumable offset)", x: 696, y: 192 },
        { id: "n_db", type: "database", label: "Conversation DB", note: "Branching message tree", x: 444, y: 276 },
      ],
      edges: [
        { id: "e_1", from: "n_cli", to: "n_gw", label: "HTTPS / SSE" },
        { id: "e_2", from: "n_gw", to: "n_orch", label: "Prompt + cursor" },
        { id: "e_3", from: "n_orch", to: "n_llm", label: "gRPC stream" },
        { id: "e_4", from: "n_llm", to: "n_buf", label: "Token chunks" },
        { id: "e_5", from: "n_buf", to: "n_gw", label: "Fan-out / resume" },
        { id: "e_6", from: "n_orch", to: "n_db", label: "Persist turns" },
      ],
    });
  }
  if (title.includes("collab") || title.includes("editor") || title.includes("docs")) {
    return normalizeDiagramGraph({
      version: 1,
      nodes: [
        { id: "n_cli", type: "client", label: "Editor Clients", note: "Local CRDT + Optimistic UI", x: 36, y: 168 },
        { id: "n_gw", type: "gateway", label: "WS Gateway", note: "Sticky routing by doc_id", x: 240, y: 168 },
        { id: "n_room", type: "service", label: "Doc Room Server", note: "Merge ops + Presence fan-out", x: 456, y: 108 },
        { id: "n_cache", type: "cache", label: "Presence Store", note: "Ephemeral cursors (Redis)", x: 696, y: 72 },
        { id: "n_db", type: "database", label: "Op Log + Snapshots", note: "Durable ops & version history", x: 696, y: 228 },
      ],
      edges: [
        { id: "e_1", from: "n_cli", to: "n_gw", label: "WebSocket" },
        { id: "e_2", from: "n_gw", to: "n_room", label: "Doc room" },
        { id: "e_3", from: "n_room", to: "n_cache", label: "Cursors" },
        { id: "e_4", from: "n_room", to: "n_db", label: "Append & compact" },
      ],
    });
  }
  return normalizeDiagramGraph({
    version: 1,
    nodes: [
      { id: "n_cli", type: "client", label: "Client Apps", note: "Web & Mobile", x: 36, y: 168 },
      { id: "n_gw", type: "gateway", label: "API Gateway", note: "TLS · Auth · Rate Limit", x: 240, y: 168 },
      { id: "n_svc", type: "service", label: "Core Service", note: "Stateless replicas", x: 456, y: 168 },
      { id: "n_cache", type: "cache", label: "Redis Cache", note: "Hot read path", x: 696, y: 84 },
      { id: "n_db", type: "database", label: "Primary DB", note: "Durable store + replicas", x: 696, y: 240 },
    ],
    edges: [
      { id: "e_1", from: "n_cli", to: "n_gw", label: "HTTPS" },
      { id: "e_2", from: "n_gw", to: "n_svc", label: "gRPC / REST" },
      { id: "e_3", from: "n_svc", to: "n_cache", label: "Read-through" },
      { id: "e_4", from: "n_svc", to: "n_db", label: "Write / Fallback" },
    ],
  });
}

function edgeBorderPoint(fromCenter, toCenter, boxW, boxH) {
  const dx = toCenter.x - fromCenter.x;
  const dy = toCenter.y - fromCenter.y;
  if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) return { ...toCenter };
  const hw = boxW / 2 + 4;
  const hh = boxH / 2 + 4;
  const tx = Math.abs(dx) > 0.001 ? hw / Math.abs(dx) : Infinity;
  const ty = Math.abs(dy) > 0.001 ? hh / Math.abs(dy) : Infinity;
  const t = Math.min(tx, ty);
  return {
    x: toCenter.x - dx * t,
    y: toCenter.y - dy * t,
  };
}

/* ---------------- Explicit Discriminated Interaction State Machine ---------------- */

export function createDesignInteractionState() {
  return { mode: "idle" };
}

export function getSelectedFromInteraction(interaction) {
  if (!interaction) return null;
  if (interaction.mode === "selected") {
    return { kind: interaction.kind, id: interaction.id };
  }
  if (interaction.mode === "dragging") {
    return { kind: "node", id: interaction.nodeId };
  }
  return null;
}

export function reduceDesignInteraction(current, action) {
  const cur = current && typeof current === "object" && current.mode ? current : { mode: "idle" };
  const act = action && typeof action === "object" ? action : {};

  switch (act.type) {
    case "SELECT_TOOL": {
      if (cur.mode === "dragging") return cur;
      if (cur.mode === "connecting") return { mode: "idle" };
      return cur;
    }
    case "START_CONNECT": {
      if (cur.mode === "dragging") return cur;
      return { mode: "connecting", fromId: act.fromId || null };
    }
    case "SET_CONNECT_SOURCE": {
      if (cur.mode !== "connecting") return cur;
      const nextFrom = act.fromId || null;
      return {
        mode: "connecting",
        fromId: cur.fromId && cur.fromId === nextFrom ? null : nextFrom,
      };
    }
    case "CANCEL_CONNECT": {
      if (cur.mode !== "connecting") return cur;
      return { mode: "idle" };
    }
    case "SELECT_ITEM": {
      if (cur.mode === "dragging") return cur;
      if ((act.kind !== "node" && act.kind !== "edge") || !act.id) return cur;
      return { mode: "selected", kind: act.kind, id: act.id, textEditDirty: false };
    }
    case "MARK_TEXT_DIRTY": {
      if (cur.mode !== "selected" || cur.textEditDirty) return cur;
      return { ...cur, textEditDirty: true };
    }
    case "CLEAR_TEXT_DIRTY": {
      if (cur.mode !== "selected" || !cur.textEditDirty) return cur;
      return { ...cur, textEditDirty: false };
    }
    case "DESELECT": {
      if (cur.mode === "selected") return { mode: "idle" };
      if (cur.mode === "connecting") return { mode: "connecting", fromId: null };
      return cur;
    }
    case "START_DRAG": {
      if (cur.mode !== "idle" && cur.mode !== "selected") return cur;
      if (!act.nodeId) return cur;
      return {
        mode: "dragging",
        nodeId: act.nodeId,
        pointerId: act.pointerId,
        captureEl: act.captureEl || null,
        offsetX: act.offsetX || 0,
        offsetY: act.offsetY || 0,
        startX: act.startX || 0,
        startY: act.startY || 0,
        moved: false,
        snapshot: act.snapshot || null,
      };
    }
    case "MOVE_DRAG": {
      if (cur.mode !== "dragging" || cur.pointerId !== act.pointerId) return cur;
      if (cur.moved) return cur;
      return { ...cur, moved: true };
    }
    case "COMMIT_DRAG": {
      if (cur.mode !== "dragging" || cur.pointerId !== act.pointerId) return cur;
      return { mode: "selected", kind: "node", id: cur.nodeId, textEditDirty: false };
    }
    case "CANCEL_DRAG": {
      if (cur.mode !== "dragging") return cur;
      if (act.pointerId != null && cur.pointerId !== act.pointerId) return cur;
      return { mode: "selected", kind: "node", id: cur.nodeId, textEditDirty: false };
    }
    case "RESET_IDLE": {
      return { mode: "idle" };
    }
    default:
      return cur;
  }
}

export function deriveDesignAllowedActions({
  interaction,
  nodeCount = 0,
  edgeCount = 0,
  historyLen = 0,
  futureLen = 0,
}) {
  const mode = interaction?.mode || "idle";
  if (mode === "dragging") {
    return ["move_node", "commit_drag", "cancel_drag"];
  }
  const actions = [];
  if (mode === "connecting") {
    actions.push("pick_connect_node", "cancel_connect");
  } else {
    actions.push("start_connect");
  }
  if (nodeCount < MAX_NODES) actions.push("add_node");
  if (mode === "selected") {
    actions.push("edit_selected", "delete_selected", "deselect");
    if (interaction?.kind === "edge") actions.push("reverse_edge");
  }
  if (historyLen > 0) actions.push("undo");
  if (futureLen > 0) actions.push("redo");
  actions.push("load_starter_template", "export_png");
  if (nodeCount > 0 || edgeCount > 0) actions.push("clear_canvas");
  if (nodeCount > 0) actions.push("ask_ai_review");
  return actions;
}

export function createSystemDesignStudioPlugin({ escapeHtml, showToast, clientMachines = null }) {
  return function mountSystemDesignStudio({
    containerEl,
    session,
    onWorkspaceChange,
    onAskAiReview,
  }) {
    const uid = shortLocalId("sd");
    const markerId = `${uid}_arrow`;
    const markerSelId = `${uid}_arrow_sel`;
    const gridId = `${uid}_grid`;

    const existingGraph = session?.workspace_context?.diagram_graph;
    const state = {
      session,
      graph: normalizeDiagramGraph(existingGraph),
      checkedReqs: new Set(
        Array.isArray(session?.workspace_context?.checked_requirements)
          ? session.workspace_context.checked_requirements
              .map((r) => safeStr(r).trim())
              .filter(Boolean)
          : []
      ),
      interaction: createDesignInteractionState(),
      history: [],
      future: [],
      playbookOpen: false,
      placeCursor: 0,
      destroyed: false,
    };

    const modeMachine = createMachine({
      name: "designStudio",
      initial: "idle",
      transitions: {
        idle: ["selected", "dragging", "connecting"],
        selected: ["idle", "dragging", "connecting"],
        dragging: ["selected", "idle"],
        connecting: ["idle", "selected"],
      },
      registry: clientMachines,
    });

    function dispatchInteraction(action, reason = null) {
      const prev = state.interaction;
      const next = reduceDesignInteraction(prev, action);
      state.interaction = next;
      if (next.mode !== prev.mode) {
        if (!modeMachine.transition(next.mode, { event: action?.type || "interaction", reason })) {
          modeMachine.reset(next.mode, { event: action?.type || "interaction", reason });
        }
      }
      return next;
    }

    function currentTool() {
      return state.interaction.mode === "connecting" ? "connect" : "select";
    }

    function currentSelected() {
      return getSelectedFromInteraction(state.interaction);
    }

    function getCurrentQuestion() {
      const qs = state.session?.questions || [];
      if (!qs.length) return null;
      const idx = Math.min(state.session?.current_question_index || 0, qs.length - 1);
      return qs[idx] || qs[0] || null;
    }

    function cloneGraph(g = state.graph) {
      return {
        version: 1,
        nodes: g.nodes.map((n) => ({ ...n })),
        edges: g.edges.map((e) => ({ ...e })),
      };
    }

    function pushHistory(snapshot = null) {
      state.history.push(snapshot || cloneGraph());
      if (state.history.length > MAX_HISTORY) state.history.shift();
      state.future = [];
    }

    function focusCanvas() {
      const svg = q("[data-sd-svg]");
      if (svg && typeof svg.focus === "function") {
        svg.focus({ preventScroll: true });
      }
    }

    function notifyChanged() {
      onWorkspaceChange?.();
    }

    function buildShell() {
      const paletteButtons = NODE_TYPES.map(
        (t) =>
          `<button type="button" class="sd-palette-btn" data-sd-add="${escapeHtml(t.type)}" title="Add ${escapeHtml(t.label)}">
             <span class="sd-palette-tag">${escapeHtml(t.badge)}</span>
             <span>${escapeHtml(t.label)}</span>
           </button>`
      ).join("");

      containerEl.innerHTML = `
        <div class="sd-studio" data-sd-root>
          <div class="sd-header">
            <div class="sd-header-left">
              <span class="practice-kicker">System Design Whiteboard</span>
              <span class="sd-count-pill" data-sd-count>0 components · 0 flows</span>
            </div>
            <div class="sd-header-actions">
              <button type="button" class="btn btn-quiet btn-xs" data-sd-playbook-toggle>Requirements</button>
              <button type="button" class="btn btn-quiet btn-xs" data-sd-template title="Load a starter architecture blueprint">Starter Template</button>
              <button type="button" class="btn btn-quiet btn-xs" data-sd-undo title="Undo (⌘/Ctrl+Z)" disabled>Undo</button>
              <button type="button" class="btn btn-quiet btn-xs" data-sd-redo title="Redo" disabled>Redo</button>
              <button type="button" class="btn btn-quiet btn-xs" data-sd-export title="Export whiteboard as PNG">Export PNG</button>
              <button type="button" class="btn btn-quiet btn-xs" data-sd-clear title="Clear whiteboard">Clear</button>
              <button type="button" class="btn btn-ink btn-xs" data-sd-review>Ask AI to review diagram</button>
            </div>
          </div>

          <div class="sd-playbook hidden" data-sd-playbook></div>

          <div class="sd-toolbar">
            <div class="sd-mode-group" role="group" aria-label="Whiteboard tool">
              <button type="button" class="sd-tool-btn active" data-sd-tool="select">Move / Select</button>
              <button type="button" class="sd-tool-btn" data-sd-tool="connect">Connect Arrow</button>
            </div>
            <div class="sd-palette" aria-label="Add architecture component">
              ${paletteButtons}
            </div>
          </div>

          <div class="sd-canvas-wrap">
            <svg class="sd-svg" data-sd-svg tabindex="0" viewBox="0 0 ${CANVAS_W} ${CANVAS_H}" role="application" aria-label="System design architecture canvas">
              <defs>
                <pattern id="${gridId}" width="24" height="24" patternUnits="userSpaceOnUse">
                  <circle cx="1.5" cy="1.5" r="1" class="sd-grid-dot"></circle>
                </pattern>
                <marker id="${markerId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
                  <path d="M 0 1.2 L 10 5 L 0 8.8 z" class="sd-arrow-fill"></path>
                </marker>
                <marker id="${markerSelId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
                  <path d="M 0 1.2 L 10 5 L 0 8.8 z" class="sd-arrow-fill-sel"></path>
                </marker>
              </defs>
              <rect x="0" y="0" width="${CANVAS_W}" height="${CANVAS_H}" fill="url(#${gridId})" class="sd-bg-rect" data-sd-bg></rect>
              <g data-sd-edges></g>
              <g data-sd-nodes></g>
            </svg>
            <div class="sd-empty-hint hidden" data-sd-empty>
              Click any component above or load a <strong>Starter Template</strong> to sketch your architecture.
            </div>
          </div>

          <div class="sd-inspector" data-sd-inspector></div>
        </div>
      `;
    }

    function q(sel) {
      return containerEl.querySelector(sel);
    }

    function renderPlaybook() {
      const box = q("[data-sd-playbook]");
      const toggleBtn = q("[data-sd-playbook-toggle]");
      if (!box) return;
      const curQ = getCurrentQuestion();
      const reqs = Array.isArray(curQ?.core_requirements) ? curQ.core_requirements : [];
      const dives = Array.isArray(curQ?.deep_dives) ? curQ.deep_dives : [];

      if (toggleBtn) {
        const covered = reqs.filter((r) => state.checkedReqs.has(r)).length;
        toggleBtn.textContent = reqs.length
          ? `Requirements (${covered}/${reqs.length})`
          : "Requirements";
      }

      box.classList.toggle("hidden", !state.playbookOpen);
      if (!state.playbookOpen) return;

      if (!reqs.length && !dives.length) {
        box.innerHTML = `<div class="sd-playbook-empty">Sketch core components, data flows, capacity estimates, and failure modes.</div>`;
        return;
      }

      box.innerHTML = `
        <div class="sd-playbook-grid">
          ${
            reqs.length
              ? `<div class="sd-playbook-col">
                  <div class="practice-kicker">Core Requirements Checklist</div>
                  <div class="sd-req-list">
                    ${reqs
                      .map((r, idx) => {
                        const checked = state.checkedReqs.has(r) ? "checked" : "";
                        return `<label class="sd-req-item"><input type="checkbox" data-sd-req-idx="${idx}" ${checked} /><span>${escapeHtml(r)}</span></label>`;
                      })
                      .join("")}
                  </div>
                </div>`
              : ""
          }
          ${
            dives.length
              ? `<div class="sd-playbook-col">
                  <div class="practice-kicker">Deep-Dive Angles to Anticipate</div>
                  <ul class="sd-dive-list">
                    ${dives.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}
                  </ul>
                </div>`
              : ""
          }
        </div>
      `;
    }

    function renderCanvas() {
      const edgesLayer = q("[data-sd-edges]");
      const nodesLayer = q("[data-sd-nodes]");
      const countEl = q("[data-sd-count]");
      const emptyEl = q("[data-sd-empty]");
      if (!edgesLayer || !nodesLayer) return;

      const { nodes, edges } = state.graph;
      const nodeMap = new Map(nodes.map((n) => [n.id, n]));

      if (countEl) {
        countEl.textContent = `${nodes.length} component${nodes.length === 1 ? "" : "s"} · ${edges.length} flow${edges.length === 1 ? "" : "s"}`;
      }
      if (emptyEl) emptyEl.classList.toggle("hidden", nodes.length > 0);

      // Update tool buttons + undo/redo buttons
      const activeTool = currentTool();
      const selected = currentSelected();
      const isDragging = state.interaction.mode === "dragging";
      containerEl.querySelectorAll("[data-sd-tool]").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-sd-tool") === activeTool);
      });
      const undoBtn = q("[data-sd-undo]");
      const redoBtn = q("[data-sd-redo]");
      if (undoBtn) undoBtn.disabled = isDragging || state.history.length === 0;
      if (redoBtn) redoBtn.disabled = isDragging || state.future.length === 0;

      // Render edges
      const edgeMarkup = [];
      for (const e of edges) {
        const a = nodeMap.get(e.from);
        const b = nodeMap.get(e.to);
        if (!a || !b) continue;
        const cA = { x: a.x + NODE_W / 2, y: a.y + NODE_H / 2 };
        const cB = { x: b.x + NODE_W / 2, y: b.y + NODE_H / 2 };
        const hasReverse = edges.some((o) => o.from === e.to && o.to === e.from);

        const isSel = selected?.kind === "edge" && selected.id === e.id;
        const mUrl = `url(#${isSel ? markerSelId : markerId})`;

        let pathD = "";
        let labelX = (cA.x + cB.x) / 2;
        let labelY = (cA.y + cB.y) / 2 - 8;

        if (hasReverse) {
          const dx = cB.x - cA.x;
          const dy = cB.y - cA.y;
          const dist = Math.hypot(dx, dy) || 1;
          const nx = -dy / dist;
          const ny = dx / dist;
          const ctrlX = (cA.x + cB.x) / 2 + nx * 28;
          const ctrlY = (cA.y + cB.y) / 2 + ny * 28;
          const start = edgeBorderPoint({ x: ctrlX, y: ctrlY }, cA, NODE_W, NODE_H);
          const end = edgeBorderPoint({ x: ctrlX, y: ctrlY }, cB, NODE_W, NODE_H);
          pathD = `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} Q ${ctrlX.toFixed(1)} ${ctrlY.toFixed(1)} ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
          labelX = 0.25 * start.x + 0.5 * ctrlX + 0.25 * end.x;
          labelY = 0.25 * start.y + 0.5 * ctrlY + 0.25 * end.y - 6;
        } else {
          const start = edgeBorderPoint(cB, cA, NODE_W, NODE_H);
          const end = edgeBorderPoint(cA, cB, NODE_W, NODE_H);
          pathD = `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} L ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
          labelX = (start.x + end.x) / 2;
          labelY = (start.y + end.y) / 2 - 8;
        }

        const lbl = truncateText(e.label, 24);
        edgeMarkup.push(`
          <g class="sd-edge-group ${isSel ? "is-selected" : ""}" data-sd-edge="${xmlEscape(e.id)}">
            <path d="${pathD}" class="sd-edge-hit" fill="none"></path>
            <path d="${pathD}" class="sd-edge-line" fill="none" marker-end="${mUrl}"></path>
            ${
              lbl
                ? `<g transform="translate(${labelX.toFixed(1)}, ${labelY.toFixed(1)})">
                     <rect x="${-(lbl.length * 3.4 + 6)}" y="-9" width="${lbl.length * 6.8 + 12}" height="17" rx="4" class="sd-edge-label-bg"></rect>
                     <text x="0" y="3" text-anchor="middle" class="sd-edge-label">${xmlEscape(lbl)}</text>
                   </g>`
                : ""
            }
          </g>
        `);
      }
      edgesLayer.innerHTML = edgeMarkup.join("");

      // Render nodes
      const nodeMarkup = [];
      for (const n of nodes) {
        const meta = NODE_TYPE_MAP.get(n.type) || NODE_TYPES[2];
        const isSel = selected?.kind === "node" && selected.id === n.id;
        const isConnectSrc =
          state.interaction.mode === "connecting" && state.interaction.fromId === n.id;
        const titleTxt = truncateText(n.label, 20);
        const noteTxt = truncateText(n.note, 24);
        const cls = [
          "sd-node",
          `sd-node-type-${n.type}`,
          isSel ? "is-selected" : "",
          isConnectSrc ? "is-connect-src" : "",
        ]
          .filter(Boolean)
          .join(" ");

        nodeMarkup.push(`
          <g class="${cls}" data-sd-node="${xmlEscape(n.id)}" transform="translate(${n.x}, ${n.y})">
            <rect x="0" y="0" width="${NODE_W}" height="${NODE_H}" rx="8" class="sd-node-box"></rect>
            <rect x="8" y="8" width="${Math.max(26, meta.badge.length * 6.5 + 10)}" height="15" rx="3" class="sd-node-badge-bg"></rect>
            <text x="${8 + Math.max(26, meta.badge.length * 6.5 + 10) / 2}" y="18.5" text-anchor="middle" class="sd-node-badge-txt">${xmlEscape(meta.badge)}</text>
            <text x="10" y="36" class="sd-node-title">${xmlEscape(titleTxt)}</text>
            ${noteTxt ? `<text x="10" y="50" class="sd-node-note">${xmlEscape(noteTxt)}</text>` : ""}
          </g>
        `);
      }
      nodesLayer.innerHTML = nodeMarkup.join("");
    }

    function renderInspector({ focusLabel = false } = {}) {
      const box = q("[data-sd-inspector]");
      if (!box) return;

      if (state.interaction.mode === "connecting") {
        const srcNode = state.interaction.fromId
          ? state.graph.nodes.find((n) => n.id === state.interaction.fromId)
          : null;
        box.innerHTML = `
          <div class="sd-inspector-status">
            <span>${
              srcNode
                ? `Connect from <strong>${escapeHtml(srcNode.label)}</strong> → click any target component (Esc to cancel)`
                : "Connect Mode: click a source component, then click a target component to draw a directed flow."
            }</span>
            <button type="button" class="btn btn-quiet btn-xs" data-sd-cancel-connect>Done connecting</button>
          </div>
        `;
        return;
      }

      const selected = currentSelected();
      const dragDisabled = state.interaction.mode === "dragging" ? "disabled" : "";
      if (selected?.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === selected.id);
        if (n) {
          const typeOpts = NODE_TYPES.map(
            (t) =>
              `<option value="${escapeHtml(t.type)}" ${t.type === n.type ? "selected" : ""}>${escapeHtml(t.label)}</option>`
          ).join("");
          box.innerHTML = `
            <div class="sd-inspector-form">
              <select class="field-input sd-insp-select" data-sd-edit-type aria-label="Component type" ${dragDisabled}>${typeOpts}</select>
              <input type="text" class="field-input sd-insp-input" data-sd-edit-label value="${escapeHtml(n.label)}" maxlength="80" placeholder="Component name" aria-label="Component name" ${dragDisabled} />
              <input type="text" class="field-input sd-insp-note" data-sd-edit-note value="${escapeHtml(n.note)}" maxlength="140" placeholder="Tech / scale notes (e.g. Redis · 50k RPS)" aria-label="Component notes" ${dragDisabled} />
              <button type="button" class="btn btn-quiet btn-xs" data-sd-connect-from="${escapeHtml(n.id)}" ${dragDisabled}>Connect →</button>
              <button type="button" class="btn btn-quiet btn-xs sd-danger-btn" data-sd-delete-selected ${dragDisabled}>Delete</button>
            </div>
          `;
          if (focusLabel && !dragDisabled) {
            const inp = box.querySelector("[data-sd-edit-label]");
            inp?.focus();
            inp?.select();
          }
          return;
        }
      }

      if (selected?.kind === "edge") {
        const e = state.graph.edges.find((x) => x.id === selected.id);
        if (e) {
          const fromNode = state.graph.nodes.find((n) => n.id === e.from);
          const toNode = state.graph.nodes.find((n) => n.id === e.to);
          box.innerHTML = `
            <div class="sd-inspector-form">
              <span class="sd-insp-flow">${escapeHtml(fromNode?.label || "?")} → ${escapeHtml(toNode?.label || "?")}</span>
              <input type="text" class="field-input sd-insp-input" data-sd-edit-edge-label value="${escapeHtml(e.label)}" maxlength="60" placeholder="Flow label (e.g. HTTPS, gRPC, SSE stream)" aria-label="Flow label" />
              <button type="button" class="btn btn-quiet btn-xs" data-sd-reverse-edge>Reverse ⇄</button>
              <button type="button" class="btn btn-quiet btn-xs sd-danger-btn" data-sd-delete-selected>Delete</button>
            </div>
          `;
          if (focusLabel) {
            const inp = box.querySelector("[data-sd-edit-edge-label]");
            inp?.focus();
            inp?.select();
          }
          return;
        }
      }

      box.innerHTML = `
        <div class="sd-inspector-hint">
          Drag components to arrange (Esc cancels drag) · Double-click to rename · Use <strong>Connect Arrow</strong> to draw data flows · Press <strong>Delete</strong> to remove selected item
        </div>
      `;
    }

    function renderAll(opts = {}) {
      renderPlaybook();
      renderCanvas();
      renderInspector(opts);
    }

    function svgPointFromEvent(ev) {
      const svg = q("[data-sd-svg]");
      if (!svg) return { x: 0, y: 0 };
      const pt = svg.createSVGPoint();
      pt.x = ev.clientX;
      pt.y = ev.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return { x: 0, y: 0 };
      const mapped = pt.matrixTransform(ctm.inverse());
      return { x: mapped.x, y: mapped.y };
    }

    function addNode(type) {
      if (state.interaction.mode === "dragging") return;
      if (state.graph.nodes.length >= MAX_NODES) {
        showToast(`Whiteboard supports up to ${MAX_NODES} components.`, 2400);
        return;
      }
      const meta = NODE_TYPE_MAP.get(type) || NODE_TYPES[2];
      pushHistory();
      const col = state.placeCursor % 4;
      const row = Math.floor(state.placeCursor / 4) % 4;
      state.placeCursor += 1;
      const x = clamp(snap(48 + col * 196 + ((state.placeCursor * 12) % 24)), 12, CANVAS_W - NODE_W - 12);
      const y = clamp(snap(36 + row * 88), 12, CANVAS_H - NODE_H - 12);
      const node = {
        id: shortLocalId("n"),
        type: meta.type,
        label: meta.defaultTitle,
        note: meta.defaultNote,
        x,
        y,
      };
      state.graph.nodes.push(node);
      dispatchInteraction({ type: "SELECT_ITEM", kind: "node", id: node.id }, "add_node");
      renderAll({ focusLabel: true });
      notifyChanged();
    }

    function deleteSelected() {
      if (state.interaction.mode !== "selected") return;
      const selected = state.interaction;
      pushHistory();
      if (selected.kind === "node") {
        const nid = selected.id;
        state.graph.nodes = state.graph.nodes.filter((n) => n.id !== nid);
        state.graph.edges = state.graph.edges.filter((e) => e.from !== nid && e.to !== nid);
      } else if (selected.kind === "edge") {
        const eid = selected.id;
        state.graph.edges = state.graph.edges.filter((e) => e.id !== eid);
      }
      dispatchInteraction({ type: "RESET_IDLE" }, "delete_selected");
      renderAll();
      notifyChanged();
    }

    function undo() {
      if (state.interaction.mode === "dragging" || !state.history.length) return;
      state.future.push(cloneGraph());
      state.graph = state.history.pop();
      dispatchInteraction({ type: "RESET_IDLE" }, "undo");
      renderAll();
      notifyChanged();
    }

    function redo() {
      if (state.interaction.mode === "dragging" || !state.future.length) return;
      state.history.push(cloneGraph());
      state.graph = state.future.pop();
      dispatchInteraction({ type: "RESET_IDLE" }, "redo");
      renderAll();
      notifyChanged();
    }

    function loadStarterTemplate() {
      if (state.interaction.mode === "dragging") return;
      pushHistory();
      state.graph = createStarterGraph(getCurrentQuestion());
      dispatchInteraction({ type: "RESET_IDLE" }, "starter_template");
      renderAll();
      notifyChanged();
      showToast("Starter blueprint loaded — drag, edit, or add components freely!", 2400);
    }

    function clearCanvas() {
      if (state.interaction.mode === "dragging") return;
      if (!state.graph.nodes.length && !state.graph.edges.length) return;
      pushHistory();
      state.graph = { version: 1, nodes: [], edges: [] };
      dispatchInteraction({ type: "RESET_IDLE" }, "clear_canvas");
      renderAll();
      notifyChanged();
    }

    function exportPng() {
      if (state.destroyed || state.interaction.mode === "dragging") return;
      const svg = q("[data-sd-svg]");
      if (!svg) return;
      const cs = getComputedStyle(document.documentElement);
      const bg = cs.getPropertyValue("--bg").trim() || "#faf8f4";
      const raised = cs.getPropertyValue("--raised").trim() || "#ffffff";
      const ink = cs.getPropertyValue("--ink").trim() || "#1b1917";
      const ink2 = cs.getPropertyValue("--ink-2").trim() || "#57534e";
      const ink3 = cs.getPropertyValue("--ink-3").trim() || "#78716c";
      const lineStrong = cs.getPropertyValue("--line-strong").trim() || "#d6d3d1";
      const accent = cs.getPropertyValue("--accent").trim() || "#1c1917";

      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", String(CANVAS_W));
      clone.setAttribute("height", String(CANVAS_H));

      const styleEl = document.createElementNS("http://www.w3.org/2000/svg", "style");
      styleEl.textContent = `
        .sd-bg-rect { fill: ${bg}; }
        .sd-grid-dot { fill: ${lineStrong}; opacity: 0.45; }
        .sd-node-box { fill: ${raised}; stroke: ${lineStrong}; stroke-width: 1.4; }
        .sd-node-badge-bg { fill: ${bg}; stroke: ${lineStrong}; stroke-width: 1; }
        .sd-node-badge-txt { fill: ${ink3}; font: 700 8.5px ui-monospace, Menlo, monospace; }
        .sd-node-title { fill: ${ink}; font: 600 12px -apple-system, BlinkMacSystemFont, sans-serif; }
        .sd-node-note { fill: ${ink2}; font: 400 10px -apple-system, BlinkMacSystemFont, sans-serif; }
        .sd-edge-hit { stroke: transparent; }
        .sd-edge-line { stroke: ${ink2}; stroke-width: 1.6; }
        .sd-arrow-fill { fill: ${ink2}; }
        .sd-arrow-fill-sel { fill: ${accent}; }
        .sd-edge-label-bg { fill: ${bg}; stroke: ${lineStrong}; stroke-width: 1; }
        .sd-edge-label { fill: ${ink2}; font: 500 10px -apple-system, BlinkMacSystemFont, sans-serif; }
      `;
      clone.insertBefore(styleEl, clone.firstChild);

      const bgFill = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bgFill.setAttribute("x", "0");
      bgFill.setAttribute("y", "0");
      bgFill.setAttribute("width", String(CANVAS_W));
      bgFill.setAttribute("height", String(CANVAS_H));
      bgFill.setAttribute("fill", bg);
      clone.insertBefore(bgFill, clone.querySelector("[data-sd-bg]"));

      const svgText = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        try {
          if (state.destroyed) return;
          const scale = 2;
          const canvas = document.createElement("canvas");
          canvas.width = CANVAS_W * scale;
          canvas.height = CANVAS_H * scale;
          const ctx = canvas.getContext("2d");
          ctx.scale(scale, scale);
          ctx.fillStyle = bg;
          ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
          ctx.drawImage(img, 0, 0, CANVAS_W, CANVAS_H);
          const pngUrl = canvas.toDataURL("image/png");
          if (state.destroyed) return;
          const a = document.createElement("a");
          a.href = pngUrl;
          a.download = "mockery_system_design.png";
          a.click();
          showToast("Exported architecture whiteboard PNG.", 2200);
        } catch {
          if (!state.destroyed) {
            showToast("Could not export PNG in this browser.", 2600);
          }
        } finally {
          URL.revokeObjectURL(url);
        }
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        if (!state.destroyed) {
          showToast("Could not render whiteboard PNG.", 2600);
        }
      };
      img.src = url;
    }

    /* ---------------- Event Handlers ---------------- */

    function releaseDragCapture(drag) {
      if (!drag?.captureEl || typeof drag.captureEl.releasePointerCapture !== "function") return;
      try {
        if (drag.captureEl.hasPointerCapture?.(drag.pointerId)) {
          drag.captureEl.releasePointerCapture(drag.pointerId);
        }
      } catch {
        // Ignore browsers that already released capture on cancel/up
      }
    }

    function cancelActiveDrag(pointerId = null, { render = true, reason = "cancel_drag" } = {}) {
      if (state.interaction.mode !== "dragging") return false;
      if (pointerId != null && state.interaction.pointerId !== pointerId) return false;
      const drag = state.interaction;
      dispatchInteraction({ type: "CANCEL_DRAG", pointerId }, reason);
      releaseDragCapture(drag);
      if (drag.snapshot) {
        state.graph = cloneGraph(drag.snapshot);
      } else {
        const node = state.graph.nodes.find((n) => n.id === drag.nodeId);
        if (node) {
          node.x = drag.startX;
          node.y = drag.startY;
        }
      }
      if (render && !state.destroyed) {
        renderAll();
      }
      return true;
    }

    function onClick(ev) {
      if (state.interaction.mode === "dragging") return;

      const addBtn = ev.target.closest("[data-sd-add]");
      if (addBtn) {
        addNode(addBtn.getAttribute("data-sd-add"));
        return;
      }

      const toolBtn = ev.target.closest("[data-sd-tool]");
      if (toolBtn) {
        const tool = toolBtn.getAttribute("data-sd-tool") || "select";
        if (tool === "connect") {
          dispatchInteraction({ type: "START_CONNECT", fromId: null }, "toolbar_connect");
        } else {
          dispatchInteraction({ type: "SELECT_TOOL" }, "toolbar_select");
        }
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-cancel-connect]")) {
        dispatchInteraction({ type: "CANCEL_CONNECT" }, "done_connecting");
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-playbook-toggle]")) {
        state.playbookOpen = !state.playbookOpen;
        renderPlaybook();
        return;
      }

      if (ev.target.closest("[data-sd-template]")) {
        loadStarterTemplate();
        return;
      }
      if (ev.target.closest("[data-sd-undo]")) {
        undo();
        return;
      }
      if (ev.target.closest("[data-sd-redo]")) {
        redo();
        return;
      }
      if (ev.target.closest("[data-sd-export]")) {
        exportPng();
        return;
      }
      if (ev.target.closest("[data-sd-clear]")) {
        clearCanvas();
        return;
      }
      if (ev.target.closest("[data-sd-review]")) {
        if (state.graph.nodes.length === 0) {
          showToast("Add a few components (or load Starter Template) before asking for a review.", 2600);
          return;
        }
        notifyChanged();
        onAskAiReview?.(
          "Please review my current System Design whiteboard diagram — highlight what looks solid and probe one key bottleneck, trade-off, or missing failure mode."
        );
        return;
      }

      const connectFromBtn = ev.target.closest("[data-sd-connect-from]");
      if (connectFromBtn) {
        dispatchInteraction(
          { type: "START_CONNECT", fromId: connectFromBtn.getAttribute("data-sd-connect-from") },
          "inspector_connect"
        );
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-delete-selected]")) {
        deleteSelected();
        return;
      }

      if (
        ev.target.closest("[data-sd-reverse-edge]") &&
        state.interaction.mode === "selected" &&
        state.interaction.kind === "edge"
      ) {
        const e = state.graph.edges.find((x) => x.id === state.interaction.id);
        if (e) {
          const existingOpposite = state.graph.edges.find(
            (x) => x.id !== e.id && x.from === e.to && x.to === e.from
          );
          if (existingOpposite) {
            dispatchInteraction(
              { type: "SELECT_ITEM", kind: "edge", id: existingOpposite.id },
              "select_opposite_edge"
            );
            renderAll();
            showToast("A flow in the opposite direction already exists — selected it.", 2400);
            return;
          }
          pushHistory();
          const tmp = e.from;
          e.from = e.to;
          e.to = tmp;
          dispatchInteraction({ type: "CLEAR_TEXT_DIRTY" }, "reverse_edge");
          renderAll();
          notifyChanged();
        }
        return;
      }

      const edgeGroup = ev.target.closest("[data-sd-edge]");
      if (edgeGroup) {
        const eid = edgeGroup.getAttribute("data-sd-edge");
        dispatchInteraction({ type: "SELECT_ITEM", kind: "edge", id: eid }, "click_edge");
        focusCanvas();
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-bg]")) {
        dispatchInteraction({ type: "DESELECT" }, "click_bg");
        focusCanvas();
        renderAll();
      }
    }

    function onDblClick(ev) {
      if (state.interaction.mode === "dragging") return;
      const nodeEl = ev.target.closest("[data-sd-node]");
      if (nodeEl) {
        dispatchInteraction(
          { type: "SELECT_ITEM", kind: "node", id: nodeEl.getAttribute("data-sd-node") },
          "dblclick_node"
        );
        renderAll({ focusLabel: true });
        return;
      }
      const edgeEl = ev.target.closest("[data-sd-edge]");
      if (edgeEl) {
        dispatchInteraction(
          { type: "SELECT_ITEM", kind: "edge", id: edgeEl.getAttribute("data-sd-edge") },
          "dblclick_edge"
        );
        renderAll({ focusLabel: true });
      }
    }

    function onPointerDown(ev) {
      const nodeEl = ev.target.closest("[data-sd-node]");
      if (!nodeEl) {
        if (ev.target.closest("[data-sd-svg]")) focusCanvas();
        return;
      }
      const nid = nodeEl.getAttribute("data-sd-node");
      const node = state.graph.nodes.find((n) => n.id === nid);
      if (!node) return;

      focusCanvas();

      if (state.interaction.mode === "connecting") {
        ev.preventDefault();
        const fromId = state.interaction.fromId;
        if (!fromId) {
          dispatchInteraction({ type: "SET_CONNECT_SOURCE", fromId: nid }, "pick_source");
          renderAll();
          return;
        }
        if (fromId === nid) {
          dispatchInteraction({ type: "SET_CONNECT_SOURCE", fromId: nid }, "clear_source");
          renderAll();
          return;
        }
        if (state.graph.edges.length >= MAX_EDGES) {
          showToast(`Maximum ${MAX_EDGES} flows reached.`, 2200);
          return;
        }
        const dup = state.graph.edges.find((e) => e.from === fromId && e.to === nid);
        if (dup) {
          dispatchInteraction({ type: "SELECT_ITEM", kind: "edge", id: dup.id }, "select_dup_edge");
          renderAll({ focusLabel: true });
          return;
        }
        pushHistory();
        const edge = { id: shortLocalId("e"), from: fromId, to: nid, label: "" };
        state.graph.edges.push(edge);
        dispatchInteraction({ type: "SELECT_ITEM", kind: "edge", id: edge.id }, "create_edge");
        renderAll({ focusLabel: true });
        notifyChanged();
        return;
      }

      if (state.interaction.mode === "dragging") return;

      ev.preventDefault();
      const pt = svgPointFromEvent(ev);
      const svg = q("[data-sd-svg]");
      dispatchInteraction(
        {
          type: "START_DRAG",
          nodeId: nid,
          pointerId: ev.pointerId,
          captureEl: svg || null,
          offsetX: pt.x - node.x,
          offsetY: pt.y - node.y,
          startX: node.x,
          startY: node.y,
          snapshot: cloneGraph(),
        },
        "pointer_down"
      );
      try {
        svg?.setPointerCapture?.(ev.pointerId);
      } catch {
        // Synthetic PointerEvents without active browser pointers may throw
      }
      renderAll();
    }

    function onPointerMove(ev) {
      if (state.interaction.mode !== "dragging" || ev.pointerId !== state.interaction.pointerId) return;
      const drag = state.interaction;
      const node = state.graph.nodes.find((n) => n.id === drag.nodeId);
      if (!node) return;
      const pt = svgPointFromEvent(ev);
      const nextX = clamp(snap(pt.x - drag.offsetX), 12, CANVAS_W - NODE_W - 12);
      const nextY = clamp(snap(pt.y - drag.offsetY), 12, CANVAS_H - NODE_H - 12);
      if (nextX !== node.x || nextY !== node.y) {
        node.x = nextX;
        node.y = nextY;
        dispatchInteraction({ type: "MOVE_DRAG", pointerId: ev.pointerId }, "pointer_move");
        renderCanvas();
      }
    }

    function onPointerUp(ev) {
      if (state.interaction.mode !== "dragging" || ev.pointerId !== state.interaction.pointerId) return;
      const drag = state.interaction;
      dispatchInteraction({ type: "COMMIT_DRAG", pointerId: ev.pointerId }, "pointer_up");
      releaseDragCapture(drag);
      if (drag.moved) {
        pushHistory(drag.snapshot);
        renderAll();
        notifyChanged();
      } else {
        renderAll();
      }
    }

    function onPointerCancel(ev) {
      cancelActiveDrag(ev.pointerId, { reason: "pointer_cancel" });
    }

    function onLostPointerCapture(ev) {
      if (state.interaction.mode !== "dragging" || ev.pointerId !== state.interaction.pointerId) return;
      cancelActiveDrag(ev.pointerId, { reason: "lost_pointer_capture" });
    }

    function onFocusOut(ev) {
      if (
        ev.target?.matches?.(
          "[data-sd-edit-label], [data-sd-edit-note], [data-sd-edit-edge-label]"
        )
      ) {
        dispatchInteraction({ type: "CLEAR_TEXT_DIRTY" }, "focus_out");
      }
    }

    function onInput(ev) {
      if (state.interaction.mode !== "selected") return;
      if (ev.target.matches("[data-sd-edit-label]") && state.interaction.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === state.interaction.id);
        if (n) {
          if (!state.interaction.textEditDirty) {
            pushHistory();
            dispatchInteraction({ type: "MARK_TEXT_DIRTY" }, "edit_label");
          }
          n.label = safeSliceChars(ev.target.value, 80);
          renderCanvas();
          notifyChanged();
        }
      } else if (ev.target.matches("[data-sd-edit-note]") && state.interaction.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === state.interaction.id);
        if (n) {
          if (!state.interaction.textEditDirty) {
            pushHistory();
            dispatchInteraction({ type: "MARK_TEXT_DIRTY" }, "edit_note");
          }
          n.note = safeSliceChars(ev.target.value, 140);
          renderCanvas();
          notifyChanged();
        }
      } else if (ev.target.matches("[data-sd-edit-edge-label]") && state.interaction.kind === "edge") {
        const e = state.graph.edges.find((x) => x.id === state.interaction.id);
        if (e) {
          if (!state.interaction.textEditDirty) {
            pushHistory();
            dispatchInteraction({ type: "MARK_TEXT_DIRTY" }, "edit_edge_label");
          }
          e.label = safeSliceChars(ev.target.value, 60);
          renderCanvas();
          notifyChanged();
        }
      }
    }

    function onChange(ev) {
      if (
        ev.target.matches("[data-sd-edit-type]") &&
        state.interaction.mode === "selected" &&
        state.interaction.kind === "node"
      ) {
        const n = state.graph.nodes.find((x) => x.id === state.interaction.id);
        const nextType = ev.target.value;
        if (n && NODE_TYPE_MAP.has(nextType)) {
          pushHistory();
          n.type = nextType;
          dispatchInteraction({ type: "CLEAR_TEXT_DIRTY" }, "edit_type");
          renderAll();
          notifyChanged();
        }
      } else if (ev.target.matches("[data-sd-req-idx]")) {
        const curQ = getCurrentQuestion();
        const reqs = Array.isArray(curQ?.core_requirements) ? curQ.core_requirements : [];
        const idx = Number(ev.target.getAttribute("data-sd-req-idx"));
        const reqText = reqs[idx];
        if (reqText) {
          if (ev.target.checked) state.checkedReqs.add(reqText);
          else state.checkedReqs.delete(reqText);
          renderPlaybook();
          notifyChanged();
        }
      }
    }

    function onKeyDown(ev) {
      if (ev.key === "Escape" && state.interaction.mode === "dragging") {
        ev.preventDefault();
        cancelActiveDrag(null, { reason: "escape_drag" });
        return;
      }

      const active = document.activeElement;
      // Only handle whiteboard shortcuts when focus is inside the studio container
      if (!active || !containerEl.contains(active)) return;

      const inEditable =
        active.tagName === "INPUT" ||
        active.tagName === "TEXTAREA" ||
        active.tagName === "SELECT" ||
        active.isContentEditable;

      if (ev.key === "Escape" && state.interaction.mode === "connecting") {
        ev.preventDefault();
        dispatchInteraction({ type: "CANCEL_CONNECT" }, "escape_connect");
        renderAll();
        return;
      }

      if (inEditable) return;

      if (ev.key === "Escape" && state.interaction.mode === "selected") {
        ev.preventDefault();
        dispatchInteraction({ type: "DESELECT" }, "escape_deselect");
        renderAll();
        return;
      }

      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") {
        ev.preventDefault();
        if (ev.shiftKey) redo();
        else undo();
        return;
      }

      if ((ev.key === "Delete" || ev.key === "Backspace") && state.interaction.mode === "selected") {
        ev.preventDefault();
        deleteSelected();
      }
    }

    buildShell();
    renderAll();

    containerEl.addEventListener("click", onClick);
    containerEl.addEventListener("dblclick", onDblClick);
    containerEl.addEventListener("pointerdown", onPointerDown);
    containerEl.addEventListener("lostpointercapture", onLostPointerCapture);
    containerEl.addEventListener("input", onInput);
    containerEl.addEventListener("focusout", onFocusOut);
    containerEl.addEventListener("change", onChange);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerCancel);
    window.addEventListener("keydown", onKeyDown);

    return {
      getWorkspaceContext() {
        const graph =
          state.interaction.mode === "dragging" && state.interaction.snapshot
            ? cloneGraph(state.interaction.snapshot)
            : cloneGraph();
        return {
          diagram_graph: graph,
          diagram_summary: serializeDiagramSummary(graph),
          checked_requirements: Array.from(state.checkedReqs).slice(0, 100),
        };
      },
      getStateSnapshot() {
        const inter = state.interaction;
        return {
          studio: "system_design",
          mode: inter.mode,
          tool: currentTool(),
          selected: currentSelected(),
          connectFromId: inter.mode === "connecting" ? inter.fromId : null,
          draggingNodeId: inter.mode === "dragging" ? inter.nodeId : null,
          nodeCount: state.graph.nodes.length,
          edgeCount: state.graph.edges.length,
          canUndo: inter.mode !== "dragging" && state.history.length > 0,
          canRedo: inter.mode !== "dragging" && state.future.length > 0,
          playbookOpen: state.playbookOpen,
          allowedActions: deriveDesignAllowedActions({
            interaction: inter,
            nodeCount: state.graph.nodes.length,
            edgeCount: state.graph.edges.length,
            historyLen: state.history.length,
            futureLen: state.future.length,
          }),
        };
      },
      updateSession(nextSession) {
        state.session = nextSession;
        renderPlaybook();
      },
      destroy() {
        state.destroyed = true;
        cancelActiveDrag(null, { render: false, reason: "destroy" });
        clientMachines?.unregister?.("designStudio");
        containerEl.removeEventListener("click", onClick);
        containerEl.removeEventListener("dblclick", onDblClick);
        containerEl.removeEventListener("pointerdown", onPointerDown);
        containerEl.removeEventListener("lostpointercapture", onLostPointerCapture);
        containerEl.removeEventListener("input", onInput);
        containerEl.removeEventListener("focusout", onFocusOut);
        containerEl.removeEventListener("change", onChange);
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        window.removeEventListener("pointercancel", onPointerCancel);
        window.removeEventListener("keydown", onKeyDown);
      },
    };
  };
}
