/* System Design Whiteboard Studio (Milestone 3)
 * 100% offline SVG architecture canvas registered as a Practice Studio plugin.
 * Ink & Paper aesthetic in Light & Dark modes, zero CDN dependencies.
 */

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

export function createSystemDesignStudioPlugin({ escapeHtml, showToast }) {
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
      tool: "select", // "select" | "connect"
      connectFromId: null,
      selected: null, // { kind: "node" | "edge", id: string } | null
      history: [],
      future: [],
      drag: null, // { nodeId, pointerId, captureEl, offsetX, offsetY, startX, startY, moved, snapshot }
      textEditDirty: false,
      playbookOpen: false,
      placeCursor: 0,
    };

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
      containerEl.querySelectorAll("[data-sd-tool]").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-sd-tool") === state.tool);
      });
      const undoBtn = q("[data-sd-undo]");
      const redoBtn = q("[data-sd-redo]");
      if (undoBtn) undoBtn.disabled = state.history.length === 0;
      if (redoBtn) redoBtn.disabled = state.future.length === 0;

      // Render edges
      const edgeMarkup = [];
      for (const e of edges) {
        const a = nodeMap.get(e.from);
        const b = nodeMap.get(e.to);
        if (!a || !b) continue;
        const cA = { x: a.x + NODE_W / 2, y: a.y + NODE_H / 2 };
        const cB = { x: b.x + NODE_W / 2, y: b.y + NODE_H / 2 };
        const hasReverse = edges.some((o) => o.from === e.to && o.to === e.from);

        const isSel = state.selected?.kind === "edge" && state.selected.id === e.id;
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
        const isSel = state.selected?.kind === "node" && state.selected.id === n.id;
        const isConnectSrc = state.tool === "connect" && state.connectFromId === n.id;
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

      if (state.tool === "connect") {
        const srcNode = state.connectFromId
          ? state.graph.nodes.find((n) => n.id === state.connectFromId)
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

      if (state.selected?.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === state.selected.id);
        if (n) {
          const typeOpts = NODE_TYPES.map(
            (t) =>
              `<option value="${escapeHtml(t.type)}" ${t.type === n.type ? "selected" : ""}>${escapeHtml(t.label)}</option>`
          ).join("");
          box.innerHTML = `
            <div class="sd-inspector-form">
              <select class="field-input sd-insp-select" data-sd-edit-type aria-label="Component type">${typeOpts}</select>
              <input type="text" class="field-input sd-insp-input" data-sd-edit-label value="${escapeHtml(n.label)}" maxlength="80" placeholder="Component name" aria-label="Component name" />
              <input type="text" class="field-input sd-insp-note" data-sd-edit-note value="${escapeHtml(n.note)}" maxlength="140" placeholder="Tech / scale notes (e.g. Redis · 50k RPS)" aria-label="Component notes" />
              <button type="button" class="btn btn-quiet btn-xs" data-sd-connect-from="${escapeHtml(n.id)}">Connect →</button>
              <button type="button" class="btn btn-quiet btn-xs sd-danger-btn" data-sd-delete-selected>Delete</button>
            </div>
          `;
          if (focusLabel) {
            const inp = box.querySelector("[data-sd-edit-label]");
            inp?.focus();
            inp?.select();
          }
          return;
        }
      }

      if (state.selected?.kind === "edge") {
        const e = state.graph.edges.find((x) => x.id === state.selected.id);
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
          Drag components to arrange · Double-click to rename · Use <strong>Connect Arrow</strong> to draw data flows · Press <strong>Delete</strong> to remove selected item
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
      state.tool = "select";
      state.connectFromId = null;
      state.selected = { kind: "node", id: node.id };
      state.textEditDirty = false;
      renderAll({ focusLabel: true });
      notifyChanged();
    }

    function deleteSelected() {
      if (!state.selected) return;
      pushHistory();
      if (state.selected.kind === "node") {
        const nid = state.selected.id;
        state.graph.nodes = state.graph.nodes.filter((n) => n.id !== nid);
        state.graph.edges = state.graph.edges.filter((e) => e.from !== nid && e.to !== nid);
      } else if (state.selected.kind === "edge") {
        const eid = state.selected.id;
        state.graph.edges = state.graph.edges.filter((e) => e.id !== eid);
      }
      state.selected = null;
      state.textEditDirty = false;
      renderAll();
      notifyChanged();
    }

    function undo() {
      if (!state.history.length) return;
      state.future.push(cloneGraph());
      state.graph = state.history.pop();
      state.selected = null;
      state.connectFromId = null;
      state.textEditDirty = false;
      renderAll();
      notifyChanged();
    }

    function redo() {
      if (!state.future.length) return;
      state.history.push(cloneGraph());
      state.graph = state.future.pop();
      state.selected = null;
      state.connectFromId = null;
      state.textEditDirty = false;
      renderAll();
      notifyChanged();
    }

    function loadStarterTemplate() {
      pushHistory();
      state.graph = createStarterGraph(getCurrentQuestion());
      state.selected = null;
      state.tool = "select";
      state.connectFromId = null;
      state.textEditDirty = false;
      renderAll();
      notifyChanged();
      showToast("Starter blueprint loaded — drag, edit, or add components freely!", 2400);
    }

    function clearCanvas() {
      if (!state.graph.nodes.length && !state.graph.edges.length) return;
      pushHistory();
      state.graph = { version: 1, nodes: [], edges: [] };
      state.selected = null;
      state.connectFromId = null;
      state.textEditDirty = false;
      renderAll();
      notifyChanged();
    }

    function exportPng() {
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
          const a = document.createElement("a");
          a.href = pngUrl;
          a.download = "mockery_system_design.png";
          a.click();
          showToast("Exported architecture whiteboard PNG.", 2200);
        } catch {
          showToast("Could not export PNG in this browser.", 2600);
        } finally {
          URL.revokeObjectURL(url);
        }
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        showToast("Could not render whiteboard PNG.", 2600);
      };
      img.src = url;
    }

    /* ---------------- Event Handlers ---------------- */

    function onClick(ev) {
      const addBtn = ev.target.closest("[data-sd-add]");
      if (addBtn) {
        addNode(addBtn.getAttribute("data-sd-add"));
        return;
      }

      const toolBtn = ev.target.closest("[data-sd-tool]");
      if (toolBtn) {
        state.tool = toolBtn.getAttribute("data-sd-tool") || "select";
        state.connectFromId = null;
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-cancel-connect]")) {
        state.tool = "select";
        state.connectFromId = null;
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
        state.tool = "connect";
        state.connectFromId = connectFromBtn.getAttribute("data-sd-connect-from");
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-delete-selected]")) {
        deleteSelected();
        return;
      }

      if (ev.target.closest("[data-sd-reverse-edge]") && state.selected?.kind === "edge") {
        const e = state.graph.edges.find((x) => x.id === state.selected.id);
        if (e) {
          const existingOpposite = state.graph.edges.find(
            (x) => x.id !== e.id && x.from === e.to && x.to === e.from
          );
          if (existingOpposite) {
            state.selected = { kind: "edge", id: existingOpposite.id };
            state.textEditDirty = false;
            renderAll();
            showToast("A flow in the opposite direction already exists — selected it.", 2400);
            return;
          }
          pushHistory();
          const tmp = e.from;
          e.from = e.to;
          e.to = tmp;
          state.textEditDirty = false;
          renderAll();
          notifyChanged();
        }
        return;
      }

      const edgeGroup = ev.target.closest("[data-sd-edge]");
      if (edgeGroup) {
        const eid = edgeGroup.getAttribute("data-sd-edge");
        state.tool = "select";
        state.connectFromId = null;
        state.selected = { kind: "edge", id: eid };
        state.textEditDirty = false;
        focusCanvas();
        renderAll();
        return;
      }

      if (ev.target.closest("[data-sd-bg]")) {
        state.selected = null;
        state.textEditDirty = false;
        if (state.tool === "connect") state.connectFromId = null;
        focusCanvas();
        renderAll();
      }
    }

    function onDblClick(ev) {
      const nodeEl = ev.target.closest("[data-sd-node]");
      if (nodeEl) {
        state.tool = "select";
        state.connectFromId = null;
        state.selected = { kind: "node", id: nodeEl.getAttribute("data-sd-node") };
        state.textEditDirty = false;
        renderAll({ focusLabel: true });
        return;
      }
      const edgeEl = ev.target.closest("[data-sd-edge]");
      if (edgeEl) {
        state.tool = "select";
        state.connectFromId = null;
        state.selected = { kind: "edge", id: edgeEl.getAttribute("data-sd-edge") };
        state.textEditDirty = false;
        renderAll({ focusLabel: true });
      }
    }

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

      if (state.tool === "connect") {
        ev.preventDefault();
        if (!state.connectFromId) {
          state.connectFromId = nid;
          renderAll();
          return;
        }
        if (state.connectFromId === nid) {
          state.connectFromId = null;
          renderAll();
          return;
        }
        if (state.graph.edges.length >= MAX_EDGES) {
          showToast(`Maximum ${MAX_EDGES} flows reached.`, 2200);
          return;
        }
        const dup = state.graph.edges.find((e) => e.from === state.connectFromId && e.to === nid);
        if (dup) {
          state.selected = { kind: "edge", id: dup.id };
          state.tool = "select";
          state.connectFromId = null;
          state.textEditDirty = false;
          renderAll({ focusLabel: true });
          return;
        }
        pushHistory();
        const edge = { id: shortLocalId("e"), from: state.connectFromId, to: nid, label: "" };
        state.graph.edges.push(edge);
        state.connectFromId = null;
        state.tool = "select";
        state.selected = { kind: "edge", id: edge.id };
        state.textEditDirty = false;
        renderAll({ focusLabel: true });
        notifyChanged();
        return;
      }

      if (state.drag) return;

      ev.preventDefault();
      const pt = svgPointFromEvent(ev);
      const svg = q("[data-sd-svg]");
      state.selected = { kind: "node", id: nid };
      state.textEditDirty = false;
      state.drag = {
        nodeId: nid,
        pointerId: ev.pointerId,
        captureEl: svg || null,
        offsetX: pt.x - node.x,
        offsetY: pt.y - node.y,
        startX: node.x,
        startY: node.y,
        moved: false,
        snapshot: cloneGraph(),
      };
      try {
        svg?.setPointerCapture?.(ev.pointerId);
      } catch {
        // Synthetic PointerEvents without active browser pointers may throw
      }
      renderAll();
    }

    function onPointerMove(ev) {
      if (!state.drag || ev.pointerId !== state.drag.pointerId) return;
      const node = state.graph.nodes.find((n) => n.id === state.drag.nodeId);
      if (!node) return;
      const pt = svgPointFromEvent(ev);
      const nextX = clamp(snap(pt.x - state.drag.offsetX), 12, CANVAS_W - NODE_W - 12);
      const nextY = clamp(snap(pt.y - state.drag.offsetY), 12, CANVAS_H - NODE_H - 12);
      if (nextX !== node.x || nextY !== node.y) {
        node.x = nextX;
        node.y = nextY;
        state.drag.moved = true;
        renderCanvas();
      }
    }

    function onPointerUp(ev) {
      if (!state.drag || ev.pointerId !== state.drag.pointerId) return;
      const drag = state.drag;
      state.drag = null;
      releaseDragCapture(drag);
      if (drag.moved) {
        pushHistory(drag.snapshot);
        renderCanvas();
        notifyChanged();
      }
    }

    function onPointerCancel(ev) {
      if (!state.drag || ev.pointerId !== state.drag.pointerId) return;
      const drag = state.drag;
      state.drag = null;
      releaseDragCapture(drag);
      const node = state.graph.nodes.find((n) => n.id === drag.nodeId);
      if (node) {
        node.x = drag.startX;
        node.y = drag.startY;
      }
      renderCanvas();
    }

    function onFocusOut(ev) {
      if (
        ev.target?.matches?.(
          "[data-sd-edit-label], [data-sd-edit-note], [data-sd-edit-edge-label]"
        )
      ) {
        state.textEditDirty = false;
      }
    }

    function onInput(ev) {
      if (ev.target.matches("[data-sd-edit-label]") && state.selected?.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === state.selected.id);
        if (n) {
          if (!state.textEditDirty) {
            pushHistory();
            state.textEditDirty = true;
          }
          n.label = safeSliceChars(ev.target.value, 80);
          renderCanvas();
          notifyChanged();
        }
      } else if (ev.target.matches("[data-sd-edit-note]") && state.selected?.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === state.selected.id);
        if (n) {
          if (!state.textEditDirty) {
            pushHistory();
            state.textEditDirty = true;
          }
          n.note = safeSliceChars(ev.target.value, 140);
          renderCanvas();
          notifyChanged();
        }
      } else if (ev.target.matches("[data-sd-edit-edge-label]") && state.selected?.kind === "edge") {
        const e = state.graph.edges.find((x) => x.id === state.selected.id);
        if (e) {
          if (!state.textEditDirty) {
            pushHistory();
            state.textEditDirty = true;
          }
          e.label = safeSliceChars(ev.target.value, 60);
          renderCanvas();
          notifyChanged();
        }
      }
    }

    function onChange(ev) {
      if (ev.target.matches("[data-sd-edit-type]") && state.selected?.kind === "node") {
        const n = state.graph.nodes.find((x) => x.id === state.selected.id);
        const nextType = ev.target.value;
        if (n && NODE_TYPE_MAP.has(nextType)) {
          pushHistory();
          n.type = nextType;
          state.textEditDirty = false;
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
      const active = document.activeElement;
      // Only handle whiteboard shortcuts when focus is inside the studio container
      if (!active || !containerEl.contains(active)) return;

      const inEditable =
        active.tagName === "INPUT" ||
        active.tagName === "TEXTAREA" ||
        active.tagName === "SELECT" ||
        active.isContentEditable;

      if (ev.key === "Escape" && state.tool === "connect") {
        state.tool = "select";
        state.connectFromId = null;
        renderAll();
        return;
      }

      if (inEditable) return;

      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") {
        ev.preventDefault();
        if (ev.shiftKey) redo();
        else undo();
        return;
      }

      if ((ev.key === "Delete" || ev.key === "Backspace") && state.selected) {
        ev.preventDefault();
        deleteSelected();
      }
    }

    buildShell();
    renderAll();

    containerEl.addEventListener("click", onClick);
    containerEl.addEventListener("dblclick", onDblClick);
    containerEl.addEventListener("pointerdown", onPointerDown);
    containerEl.addEventListener("input", onInput);
    containerEl.addEventListener("focusout", onFocusOut);
    containerEl.addEventListener("change", onChange);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerCancel);
    window.addEventListener("keydown", onKeyDown);

    return {
      getWorkspaceContext() {
        const graph = cloneGraph();
        return {
          diagram_graph: graph,
          diagram_summary: serializeDiagramSummary(graph),
          checked_requirements: Array.from(state.checkedReqs).slice(0, 100),
        };
      },
      updateSession(nextSession) {
        state.session = nextSession;
        renderPlaybook();
      },
      destroy() {
        if (state.drag) {
          releaseDragCapture(state.drag);
          state.drag = null;
        }
        containerEl.removeEventListener("click", onClick);
        containerEl.removeEventListener("dblclick", onDblClick);
        containerEl.removeEventListener("pointerdown", onPointerDown);
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
