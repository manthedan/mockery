/* System Design Whiteboard Studio
 * 100% offline SVG architecture canvas registered as a Practice Studio plugin.
 * Ink & Paper aesthetic in Light & Dark modes, zero CDN dependencies.
 */

import { createMachine } from "./state_machines.js";

const CANVAS_W = 920;
const CANVAS_H = 420;
// The canvas fills its pane; the 920×420 artboard stays centred inside it. The dot
// grid bleeds this far past the artboard so no pane shape ever shows a bare margin.
const CANVAS_BLEED = 2400;
const NODE_W = 148;
const NODE_H = 58;
const GRID_STEP = 12;
const MAX_NODES = 50;
const MAX_EDGES = 100;
const MAX_HISTORY = 40;
const MAX_SUMMARY_COMP_CHARS = 3400;
const MAX_SUMMARY_FLOW_CHARS = 2400;
const MAX_DESIGN_NOTES_CHARS = 16000;

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

// Models and older saves describe components loosely ("db", "kafka", "lb").
// Map the common spellings to a known type so a Postgres box never renders
// with a generic service badge. Anything unrecognised still falls back to
// "service".
const NODE_TYPE_ALIASES = new Map(
  Object.entries({
    database: ["db", "sql", "nosql", "postgres", "postgresql", "mysql", "datastore", "data_store", "store", "table", "rdbms", "dynamodb", "mongodb", "cassandra"],
    cache: ["redis", "memcached", "memcache", "kv", "kv_store"],
    queue: ["mq", "kafka", "stream", "streams", "pubsub", "pub_sub", "bus", "event_bus", "message_queue", "sqs", "topic"],
    gateway: ["lb", "load_balancer", "loadbalancer", "api_gateway", "apigateway", "proxy", "edge", "router", "ingress"],
    storage: ["cdn", "s3", "blob", "blob_store", "object_store", "object_storage", "bucket", "files"],
    worker: ["ai", "llm", "model", "ml", "job", "jobs", "cron", "consumer", "inference"],
    client: ["cli", "user", "users", "browser", "web", "mobile", "app", "frontend"],
    service: ["svc", "api", "server", "backend", "microservice"],
    note: ["comment", "annotation", "text", "label"],
  }).flatMap(([type, aliases]) => aliases.map((a) => [a, type]))
);

export function resolveNodeType(rawType) {
  const raw = typeof rawType === "string" ? rawType : "";
  if (NODE_TYPE_MAP.has(raw)) return raw;
  const key = raw.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (NODE_TYPE_MAP.has(key)) return key;
  return NODE_TYPE_ALIASES.get(key) || "service";
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
      const type = resolveNodeType(rawType);
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

/* ---------------- Edge routing & label placement ----------------

   Every flow leaves the perimeter of its source box and ends — arrowhead and all —
   at the perimeter of its target box. Aligned boxes get one straight line. Diagonal
   neighbours get a single rounded elbow, so two boxes that almost touch corner to
   corner are still joined by legs long enough to read (a centre-to-centre line
   between them is a stub that its own label swallows). Labels are placed last: on
   the line where a leg can carry them, beside it where it cannot, and never over a
   box, another label, or somebody else's line when a clear spot exists. */

const EDGE_TIP_OVERSHOOT = 3.1; // the arrowhead reaches this far past the end of its path
const EDGE_TIP_GAP = EDGE_TIP_OVERSHOOT + 1.4; // path end → box edge: tip rests just off the stroke
const EDGE_ARROW_ROOM = 8; // straight run the arrowhead needs before its path ends
const EDGE_MIN_LEG = 20; // an elbow leg shorter than this reads as a kink, not a route
const EDGE_CORNER_R = 8;
const EDGE_LANE = 7; // a flow and its reverse run this far either side of the shared axis
const EDGE_LABEL_H = 17;
const EDGE_LABEL_PAD_X = 7;
const EDGE_LABEL_FONT = '500 10px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif';

function nodeRect(n) {
  return {
    x1: n.x,
    y1: n.y,
    x2: n.x + NODE_W,
    y2: n.y + NODE_H,
    cx: n.x + NODE_W / 2,
    cy: n.y + NODE_H / 2,
  };
}

function rectsOverlap(r, o, pad = 0) {
  return r.x1 < o.x2 + pad && r.x2 > o.x1 - pad && r.y1 < o.y2 + pad && r.y2 > o.y1 - pad;
}

// Liang–Barsky: does the segment p→q pass through rect r?
function segmentHitsRect(p, q, r) {
  let t0 = 0;
  let t1 = 1;
  const clip = (den, num) => {
    if (den === 0) return num >= 0;
    const t = num / den;
    if (den < 0) {
      if (t > t1) return false;
      if (t > t0) t0 = t;
    } else {
      if (t < t0) return false;
      if (t < t1) t1 = t;
    }
    return true;
  };
  const dx = q.x - p.x;
  const dy = q.y - p.y;
  return (
    clip(-dx, p.x - r.x1) && clip(dx, r.x2 - p.x) && clip(-dy, p.y - r.y1) && clip(dy, r.y2 - p.y)
  );
}

/**
 * Route one flow between two components.
 * Returns the polyline it follows: two points for a straight run, three for an elbow.
 * `paired` means the reverse flow also exists, so the two must not sit on top of each other.
 * `obstacles` are the rects of the other components, used only to pick the clearer elbow.
 */
export function routeEdge(a, b, { paired = false, obstacles = [] } = {}) {
  const A = nodeRect(a);
  const B = nodeRect(b);
  const dx = B.cx - A.cx;
  const dy = B.cy - A.cy;
  const right = dx >= 0;
  const down = dy >= 0;
  const gapX = Math.abs(dx) - NODE_W; // clear space between the boxes; negative = overlap
  const gapY = Math.abs(dy) - NODE_H;
  const lane = paired ? EDGE_LANE : 0;

  const straightH = () => {
    const y =
      (Math.max(A.y1, B.y1) + Math.min(A.y2, B.y2)) / 2 +
      (right ? -1 : 1) * Math.min(lane, Math.max(0, -gapY / 2 - 4));
    return [
      { x: right ? A.x2 : A.x1, y },
      { x: right ? B.x1 - EDGE_TIP_GAP : B.x2 + EDGE_TIP_GAP, y },
    ];
  };
  const straightV = () => {
    const x =
      (Math.max(A.x1, B.x1) + Math.min(A.x2, B.x2)) / 2 +
      (down ? 1 : -1) * Math.min(lane, Math.max(0, -gapX / 2 - 4));
    return [
      { x, y: down ? A.y2 : A.y1 },
      { x, y: down ? B.y1 - EDGE_TIP_GAP : B.y2 + EDGE_TIP_GAP },
    ];
  };

  // Same row or same column: one straight line through the band the boxes share.
  if (gapX > 0 && -gapY >= NODE_H / 2) return straightH();
  if (gapY > 0 && -gapX >= NODE_W / 2) return straightV();

  // Diagonal: one elbow. Both orientations have the same leg lengths, so they are
  // either both drawable or neither is.
  const legX = Math.abs(dx) - NODE_W / 2;
  const legY = Math.abs(dy) - NODE_H / 2;
  if (legX - EDGE_TIP_GAP >= EDGE_MIN_LEG && legY - EDGE_TIP_GAP >= EDGE_MIN_LEG) {
    // Out through the top/bottom, in through the side.
    const viaSide = [
      { x: A.cx, y: down ? A.y2 : A.y1 },
      { x: A.cx, y: B.cy },
      { x: right ? B.x1 - EDGE_TIP_GAP : B.x2 + EDGE_TIP_GAP, y: B.cy },
    ];
    // A flow and its reverse both take this shape, which puts them on opposite corners.
    if (paired) return viaSide;
    // Out through the side, in through the top/bottom.
    const viaTop = [
      { x: right ? A.x2 : A.x1, y: A.cy },
      { x: B.cx, y: A.cy },
      { x: B.cx, y: down ? B.y1 - EDGE_TIP_GAP : B.y2 + EDGE_TIP_GAP },
    ];
    const others = obstacles.filter(
      (o) => !(o.x1 === A.x1 && o.y1 === A.y1) && !(o.x1 === B.x1 && o.y1 === B.y1)
    );
    // Count the boxes each elbow would run through (or graze: a little air counts too).
    const crossings = (pts) =>
      others.reduce((n, o) => {
        const air = { x1: o.x1 - 6, y1: o.y1 - 6, x2: o.x2 + 6, y2: o.y2 + 6 };
        return n + (segmentHitsRect(pts[0], pts[1], air) || segmentHitsRect(pts[1], pts[2], air) ? 1 : 0);
      }, 0);
    return crossings(viaTop) < crossings(viaSide) ? viaTop : viaSide;
  }

  // Too offset for the aligned case, too close for an elbow: use the sliver they share.
  if (gapY > 0 && -gapX >= 16) return straightV();
  if (gapX > 0 && -gapY >= 10) return straightH();

  // Last resort (boxes overlapping, or barely offset): clip the centre line to both perimeters.
  const clip = (from, box, pad) => {
    const ddx = box.cx - from.cx;
    const ddy = box.cy - from.cy;
    if (Math.abs(ddx) < 0.001 && Math.abs(ddy) < 0.001) return { x: box.cx, y: box.cy };
    const t = Math.min(
      1,
      Math.abs(ddx) > 0.001 ? (NODE_W / 2 + pad) / Math.abs(ddx) : Infinity,
      Math.abs(ddy) > 0.001 ? (NODE_H / 2 + pad) / Math.abs(ddy) : Infinity
    );
    return { x: box.cx - ddx * t, y: box.cy - ddy * t };
  };
  const dist = Math.hypot(dx, dy) || 1;
  const shift = { x: (dy / dist) * lane, y: (-dx / dist) * lane };
  const start = clip(B, A, 0);
  const end = clip(A, B, EDGE_TIP_GAP);
  return [
    { x: start.x + shift.x, y: start.y + shift.y },
    { x: end.x + shift.x, y: end.y + shift.y },
  ];
}

/** SVG path for a routed polyline, with elbows rounded and room left for the arrowhead. */
export function edgePathD(pts) {
  const f = (n) => n.toFixed(1);
  let d = `M ${f(pts[0].x)} ${f(pts[0].y)}`;
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i];
    const prev = pts[i - 1];
    const next = pts[i + 1];
    if (!next) {
      d += ` L ${f(p.x)} ${f(p.y)}`;
      break;
    }
    const inLen = Math.hypot(p.x - prev.x, p.y - prev.y) || 1;
    const outLen = Math.hypot(next.x - p.x, next.y - p.y) || 1;
    const outRoom = i === pts.length - 2 ? outLen - EDGE_ARROW_ROOM : outLen / 2;
    const r = Math.max(0, Math.min(EDGE_CORNER_R, inLen / 2, outRoom));
    const before = { x: p.x - ((p.x - prev.x) / inLen) * r, y: p.y - ((p.y - prev.y) / inLen) * r };
    const after = { x: p.x + ((next.x - p.x) / outLen) * r, y: p.y + ((next.y - p.y) / outLen) * r };
    d += ` L ${f(before.x)} ${f(before.y)} Q ${f(p.x)} ${f(p.y)} ${f(after.x)} ${f(after.y)}`;
  }
  return d;
}

/**
 * Choose where a w×h label pill sits for a routed flow. Returns the pill's centre.
 * `boxes` are all component rects, `labels` the pills already placed, and `lines`
 * the [p, q] segments of every other flow.
 */
export function placeEdgeLabel(pts, w, h, { boxes = [], labels = [], lines = [] } = {}) {
  const legs = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p = pts[i];
    const q = pts[i + 1];
    const len = Math.hypot(q.x - p.x, q.y - p.y);
    if (len < 1) continue;
    const ux = (q.x - p.x) / len;
    const uy = (q.y - p.y) / len;
    // Length of line a pill hides when centred on this leg.
    const cover = Math.min(
      Math.abs(ux) > 0.01 ? w / Math.abs(ux) : Infinity,
      Math.abs(uy) > 0.01 ? h / Math.abs(uy) : Infinity
    );
    // Keep a visible stub at each end — and the whole arrowhead on the final leg.
    const tail = i === pts.length - 2 ? EDGE_ARROW_ROOM + 5 : 5;
    const lo = cover / 2 + 5;
    legs.push({ p, ux, uy, len, lo, room: len - cover / 2 - tail - lo });
  }
  if (legs.length === 0) {
    const last = pts[pts.length - 1];
    return { x: (pts[0].x + last.x) / 2, y: (pts[0].y + last.y) / 2 };
  }
  legs.sort((s, t) => t.room - s.room);
  const at = (leg, along, off = 0) => ({
    x: leg.p.x + leg.ux * along - leg.uy * off,
    y: leg.p.y + leg.uy * along + leg.ux * off,
  });

  const candidates = [];
  // 1. On the line: centred between the leg's start and its arrowhead, then nudged along it.
  for (const leg of legs) {
    if (leg.room < 0) continue;
    for (const frac of [0.5, 0.15, 0.85]) candidates.push({ ...at(leg, leg.lo + leg.room * frac), away: 0 });
  }
  // 2. Beside the line, when no leg can carry the pill without swallowing the arrow —
  //    stepping further out when the gap the line runs through is narrower than the pill.
  //    Fine steps let the pill settle just clear of whatever blocks it, so it still
  //    reads as belonging to its arrow rather than floating above it.
  for (let step = 0; step <= 12; step++) {
    for (const leg of legs) {
      const off = (Math.abs(leg.uy) * w) / 2 + (Math.abs(leg.ux) * h) / 2 + 5 + step * 3;
      const sides = [at(leg, leg.len / 2, -off), at(leg, leg.len / 2, off)];
      sides.sort((m, n) => m.y - n.y || n.x - m.x); // above first, then to the right
      for (const side of sides) candidates.push({ ...side, away: step });
    }
  }

  let best = candidates[0];
  let bestCost = Infinity;
  candidates.forEach((c, idx) => {
    const r = { x1: c.x - w / 2, y1: c.y - h / 2, x2: c.x + w / 2, y2: c.y + h / 2 };
    const near = { x1: r.x1 - 2, y1: r.y1 - 2, x2: r.x2 + 2, y2: r.y2 + 2 };
    // Earlier candidates win ties; drifting away from the line costs more than a snug fit,
    // but far less than covering a box, a label, or another flow.
    let cost = idx * 0.005 + c.away;
    if (r.x1 < 2 || r.y1 < 2 || r.x2 > CANVAS_W - 2 || r.y2 > CANVAS_H - 2) cost += 100;
    for (const box of boxes) cost += rectsOverlap(r, box, 1) ? 100 : rectsOverlap(r, box, 3) ? 1 : 0;
    for (const lbl of labels) if (rectsOverlap(r, lbl, 2)) cost += 60;
    for (const [p, q] of lines) if (segmentHitsRect(p, q, near)) cost += 25;
    if (cost < bestCost) {
      best = c;
      bestCost = cost;
    }
  });
  return { x: best.x, y: best.y };
}

let edgeLabelMeasureCtx;

/** Width of a label pill: the real rendered text width plus padding. */
function edgeLabelWidth(text) {
  if (edgeLabelMeasureCtx === undefined) {
    try {
      edgeLabelMeasureCtx = document.createElement("canvas").getContext("2d");
      if (edgeLabelMeasureCtx) edgeLabelMeasureCtx.font = EDGE_LABEL_FONT;
    } catch {
      edgeLabelMeasureCtx = null;
    }
  }
  const textW = edgeLabelMeasureCtx
    ? edgeLabelMeasureCtx.measureText(text).width
    : Array.from(text).length * 5.9;
  return Math.ceil(textW) + EDGE_LABEL_PAD_X * 2;
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
  hasNotes = false,
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
  if (nodeCount > 0 || hasNotes) actions.push("ask_ai_review");
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
    const initialDesignNotes = safeSliceChars(
      safeStr(session?.workspace_context?.design_notes, ""),
      MAX_DESIGN_NOTES_CHARS
    );
    const state = {
      session,
      graph: normalizeDiagramGraph(existingGraph),
      designNotes: initialDesignNotes,
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
      notesOpen: Boolean(initialDesignNotes.trim()),
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
          `<button type="button" class="sd-palette-btn" data-sd-add="${escapeHtml(t.type)}" data-sd-kind="${escapeHtml(t.type)}" title="Add ${escapeHtml(t.label)}" aria-label="Add ${escapeHtml(t.label)}">
             <span class="sd-palette-tag">${escapeHtml(t.badge)}</span>
             <span class="sd-palette-name">${escapeHtml(t.label)}</span>
           </button>`
      ).join("");

      containerEl.innerHTML = `
        <div class="sd-studio" data-sd-root>
          <div class="sd-header">
            <div class="sd-header-left">
              <span class="practice-kicker">Whiteboard</span>
              <span class="sd-count-pill" data-sd-count>0 components · 0 flows</span>
            </div>
            <div class="sd-header-actions">
              <button type="button" class="sd-chip-btn" data-sd-notes-toggle aria-expanded="false">Notes · API &amp; Math</button>
              <button type="button" class="sd-chip-btn" data-sd-playbook-toggle>Requirements</button>
              <button type="button" class="btn btn-ink btn-xs sd-review-btn" data-sd-review title="Send your diagram and design notes to your interviewer for feedback">
                <svg class="sd-review-glyph" viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M8 1.6l1.5 4.1 4.1 1.5-4.1 1.5L8 12.8 6.5 8.7 2.4 7.2l4.1-1.5z" fill="currentColor"/></svg>
                <span>Ask AI to review</span>
              </button>
            </div>
          </div>

          <div class="sd-playbook hidden" data-sd-playbook></div>

          <div class="sd-notes-panel hidden" data-sd-notes-panel>
            <div class="sd-notes-head">
              <span class="practice-kicker">API · Schema · Capacity Math</span>
              <span class="sd-notes-meta" data-sd-notes-meta>Shared live with your interviewer</span>
            </div>
            <textarea
              class="sd-notes-textarea"
              data-sd-notes
              rows="5"
              maxlength="${MAX_DESIGN_NOTES_CHARS}"
              spellcheck="false"
              aria-label="System design notes for APIs, schemas, and capacity math"
              placeholder="APIs: POST /v1/...&#10;Schema: tables, partition keys, indexes&#10;Math: peak QPS, p99 budget, storage/day"
            ></textarea>
          </div>

          <div class="sd-toolbar">
            <div class="sd-mode-group" role="group" aria-label="Whiteboard tool">
              <button type="button" class="sd-tool-btn active" data-sd-tool="select" title="Move and select (V)" aria-label="Move and select">
                <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M3.2 2.2l9.4 4.6-4 1.3-1.4 4z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>
                <span>Select</span>
              </button>
              <button type="button" class="sd-tool-btn" data-sd-tool="connect" title="Draw a flow between two components" aria-label="Connect components">
                <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M2.5 12.5l9-9M7 3.2h4.8V8" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <span>Connect</span>
              </button>
            </div>
            <span class="sd-toolbar-rule" aria-hidden="true"></span>
            <div class="sd-palette" role="group" aria-label="Add architecture component">
              ${paletteButtons}
            </div>
            <span class="sd-toolbar-spacer" aria-hidden="true"></span>
            <div class="sd-history-group" role="group" aria-label="History">
              <button type="button" class="sd-icon-btn" data-sd-undo title="Undo (⌘/Ctrl+Z)" aria-label="Undo" disabled>
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M5.6 3.4L2.6 6.4l3 3M2.9 6.4h6.2a3.6 3.6 0 010 7.2H6.4" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </button>
              <button type="button" class="sd-icon-btn" data-sd-redo title="Redo (⇧⌘Z)" aria-label="Redo" disabled>
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M10.4 3.4l3 3-3 3M13.1 6.4H6.9a3.6 3.6 0 000 7.2h2.7" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </button>
            </div>
            <details class="sd-more" data-sd-pop>
              <summary class="sd-icon-btn" title="More whiteboard actions" aria-label="More whiteboard actions">
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><circle cx="3.5" cy="8" r="1.2" fill="currentColor"/><circle cx="8" cy="8" r="1.2" fill="currentColor"/><circle cx="12.5" cy="8" r="1.2" fill="currentColor"/></svg>
              </summary>
              <div class="sd-more-menu" role="menu">
                <button type="button" class="sd-menu-item" role="menuitem" data-sd-template title="Load a starter architecture blueprint">Start from a template</button>
                <button type="button" class="sd-menu-item" role="menuitem" data-sd-export title="Export whiteboard as PNG">Export as PNG</button>
                <span class="sd-menu-rule" aria-hidden="true"></span>
                <button type="button" class="sd-menu-item is-danger" role="menuitem" data-sd-clear title="Clear whiteboard">Clear whiteboard</button>
              </div>
            </details>
          </div>

          <div class="sd-canvas-wrap">
            <svg class="sd-svg" data-sd-svg tabindex="0" viewBox="0 0 ${CANVAS_W} ${CANVAS_H}" role="application" aria-label="System design architecture canvas">
              <defs>
                <pattern id="${gridId}" width="24" height="24" patternUnits="userSpaceOnUse">
                  <circle cx="1.5" cy="1.5" r="1" class="sd-grid-dot"></circle>
                </pattern>
                <marker id="${markerId}" viewBox="0 0 10 10" refX="7" refY="5" markerUnits="userSpaceOnUse" markerWidth="10.4" markerHeight="10.4" orient="auto-start-reverse">
                  <path d="M 0 1.2 L 10 5 L 0 8.8 z" class="sd-arrow-fill"></path>
                </marker>
                <marker id="${markerSelId}" viewBox="0 0 10 10" refX="7" refY="5" markerUnits="userSpaceOnUse" markerWidth="10.4" markerHeight="10.4" orient="auto-start-reverse">
                  <path d="M 0 1.2 L 10 5 L 0 8.8 z" class="sd-arrow-fill-sel"></path>
                </marker>
              </defs>
              <rect x="${-CANVAS_BLEED}" y="${-CANVAS_BLEED}" width="${CANVAS_W + CANVAS_BLEED * 2}" height="${CANVAS_H + CANVAS_BLEED * 2}" fill="url(#${gridId})" class="sd-bg-rect" data-sd-bg></rect>
              <rect x="0.5" y="0.5" width="${CANVAS_W - 1}" height="${CANVAS_H - 1}" rx="10" class="sd-artboard" aria-hidden="true"></rect>
              <g data-sd-edges></g>
              <g data-sd-nodes></g>
            </svg>
            <div class="sd-empty-hint hidden" data-sd-empty>
              <span class="sd-empty-title">Start from a blank whiteboard</span>
              <span class="sd-empty-sub">Clarify requirements first, add components above, or</span>
              <button type="button" class="chip-action" data-sd-template>load a starter template</button>
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
          ? `Requirements ${covered}/${reqs.length}`
          : "Requirements";
        toggleBtn.classList.toggle("is-open", !!state.playbookOpen);
        toggleBtn.classList.toggle("is-complete", reqs.length > 0 && covered === reqs.length);
        toggleBtn.setAttribute("aria-expanded", state.playbookOpen ? "true" : "false");
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
                  <div class="practice-kicker">Core requirements</div>
                  <div class="sd-req-list">
                    ${reqs
                      .map((r, idx) => {
                        const checked = state.checkedReqs.has(r) ? "checked" : "";
                        return `<label class="sd-req-item"><input type="checkbox" class="studio-check" data-sd-req-idx="${idx}" ${checked} /><span>${escapeHtml(r)}</span></label>`;
                      })
                      .join("")}
                  </div>
                </div>`
              : ""
          }
          ${
            dives.length
              ? `<div class="sd-playbook-col">
                  <details class="sd-dives-spoiler">
                    <summary class="practice-kicker sd-dives-summary">Expected deep dives (spoilers)</summary>
                    <ul class="sd-dive-list">
                      ${dives.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}
                    </ul>
                  </details>
                </div>`
              : ""
          }
        </div>
      `;
    }

    function renderNotes() {
      const panel = q("[data-sd-notes-panel]");
      const toggleBtn = q("[data-sd-notes-toggle]");
      const metaEl = q("[data-sd-notes-meta]");
      const ta = q("[data-sd-notes]");
      const trimmedLen = state.designNotes.trim().length;
      if (toggleBtn) {
        toggleBtn.textContent = trimmedLen > 0 ? "Notes · API & Math •" : "Notes · API & Math";
        toggleBtn.classList.toggle("is-open", !!state.notesOpen);
        toggleBtn.classList.toggle("is-complete", trimmedLen > 0);
        toggleBtn.setAttribute("aria-expanded", state.notesOpen ? "true" : "false");
      }
      if (panel) {
        panel.classList.toggle("hidden", !state.notesOpen);
      }
      if (ta && document.activeElement !== ta && ta.value !== state.designNotes) {
        ta.value = state.designNotes;
      }
      if (metaEl) {
        metaEl.textContent =
          trimmedLen > 0
            ? `${state.designNotes.length.toLocaleString()} / ${MAX_DESIGN_NOTES_CHARS.toLocaleString()} chars · synced live`
            : "Shared live with your interviewer";
      }
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
      // While a component is being moved, the drawable frame shows as a quiet guide.
      q("[data-sd-svg]")?.classList.toggle("is-dragging", isDragging);
      const undoBtn = q("[data-sd-undo]");
      const redoBtn = q("[data-sd-redo]");
      if (undoBtn) undoBtn.disabled = isDragging || state.history.length === 0;
      if (redoBtn) redoBtn.disabled = isDragging || state.future.length === 0;

      // Render edges: route every flow first, then place labels around all boxes and lines.
      const boxes = nodes.map(nodeRect);
      const routed = [];
      for (const e of edges) {
        const a = nodeMap.get(e.from);
        const b = nodeMap.get(e.to);
        if (!a || !b) continue;
        const paired = edges.some((o) => o.from === e.to && o.to === e.from);
        routed.push({ edge: e, pts: routeEdge(a, b, { paired, obstacles: boxes }) });
      }

      const lineMarkup = [];
      const labelMarkup = [];
      const placedLabels = [];
      for (const r of routed) {
        const e = r.edge;
        const isSel = selected?.kind === "edge" && selected.id === e.id;
        const groupCls = `sd-edge-group ${isSel ? "is-selected" : ""}`;
        const pathD = edgePathD(r.pts);
        lineMarkup.push(`
          <g class="${groupCls}" data-sd-edge="${xmlEscape(e.id)}">
            <path d="${pathD}" class="sd-edge-hit" fill="none"></path>
            <path d="${pathD}" class="sd-edge-line" fill="none" marker-end="url(#${isSel ? markerSelId : markerId})"></path>
          </g>
        `);

        const lbl = truncateText(e.label, 24);
        if (!lbl) continue;
        const w = edgeLabelWidth(lbl);
        const lines = [];
        for (const other of routed) {
          if (other === r) continue;
          for (let i = 0; i < other.pts.length - 1; i++) lines.push([other.pts[i], other.pts[i + 1]]);
        }
        const c = placeEdgeLabel(r.pts, w, EDGE_LABEL_H, { boxes, labels: placedLabels, lines });
        placedLabels.push({
          x1: c.x - w / 2,
          y1: c.y - EDGE_LABEL_H / 2,
          x2: c.x + w / 2,
          y2: c.y + EDGE_LABEL_H / 2,
        });
        labelMarkup.push(`
          <g class="${groupCls} sd-edge-label-group" data-sd-edge="${xmlEscape(e.id)}" transform="translate(${c.x.toFixed(1)}, ${c.y.toFixed(1)})">
            <rect x="${(-w / 2).toFixed(1)}" y="${-EDGE_LABEL_H / 2}" width="${w}" height="${EDGE_LABEL_H}" rx="${EDGE_LABEL_H / 2}" class="sd-edge-label-bg"></rect>
            <text x="0" y="3.5" text-anchor="middle" class="sd-edge-label">${xmlEscape(lbl)}</text>
          </g>
        `);
      }
      // Pills are drawn after every line, so no flow ever runs across another flow's label.
      edgesLayer.innerHTML = lineMarkup.join("") + labelMarkup.join("");

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
                ? `From <strong>${escapeHtml(srcNode.label)}</strong> — now click the component it flows to`
                : "Click a component to start a flow, then click where it goes"
            }</span>
            <button type="button" class="btn btn-quiet btn-xs" data-sd-cancel-connect>Done <kbd>esc</kbd></button>
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
          <span>Drag to arrange</span><span class="sd-hint-dot" aria-hidden="true">·</span><span>Double-click to rename</span><span class="sd-hint-dot" aria-hidden="true">·</span><span><kbd>⌫</kbd> removes the selection</span>
        </div>
      `;
    }

    function renderAll(opts = {}) {
      renderPlaybook();
      renderNotes();
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
      if (state.placeCursor < state.graph.nodes.length) {
        state.placeCursor = state.graph.nodes.length;
      }
      let x = 48;
      let y = 36;
      for (let attempt = 0; attempt < 24; attempt++) {
        const idx = state.placeCursor + attempt;
        const col = idx % 4;
        const row = Math.floor(idx / 4) % 4;
        const candX = clamp(snap(48 + col * 196 + ((idx * 12) % 24)), 12, CANVAS_W - NODE_W - 12);
        const candY = clamp(snap(36 + row * 88), 12, CANVAS_H - NODE_H - 12);
        const collides = state.graph.nodes.some(
          (n) => Math.abs(n.x - candX) < 36 && Math.abs(n.y - candY) < 28
        );
        x = candX;
        y = candY;
        if (!collides) {
          state.placeCursor = idx + 1;
          break;
        }
        if (attempt === 23) {
          state.placeCursor += 1;
        }
      }
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
        .sd-artboard { fill: none; stroke: none; }
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

    function closePopovers(exceptTarget = null) {
      containerEl.querySelectorAll("details[data-sd-pop][open]").forEach((d) => {
        if (exceptTarget && d.contains(exceptTarget) && exceptTarget.closest("summary")) return;
        d.open = false;
      });
    }

    function onClick(ev) {
      closePopovers(ev.target);
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

      if (ev.target.closest("[data-sd-notes-toggle]")) {
        state.notesOpen = !state.notesOpen;
        renderNotes();
        if (state.notesOpen) {
          q("[data-sd-notes]")?.focus();
        }
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
        const hasNodes = state.graph.nodes.length > 0;
        const hasNotes = state.designNotes.trim().length > 0;
        if (!hasNodes && !hasNotes) {
          showToast("Add a few components or jot down API/schema notes before asking for a review.", 2600);
          return;
        }
        notifyChanged();
        const artifact =
          hasNodes && hasNotes
            ? "whiteboard diagram and API/schema/capacity notes"
            : hasNodes
            ? "whiteboard diagram"
            : "API/schema/capacity design notes";
        onAskAiReview?.(
          `Please review my current System Design ${artifact} — highlight what looks solid and probe one key bottleneck, trade-off, or missing failure mode.`
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
        const now = Date.now();
        const isDblTap =
          ev.detail >= 2 ||
          (state.lastEdgeTap && state.lastEdgeTap.id === eid && now - state.lastEdgeTap.at <= 380);
        state.lastEdgeTap = isDblTap ? null : { id: eid, at: now };
        dispatchInteraction({ type: "SELECT_ITEM", kind: "edge", id: eid }, isDblTap ? "dblclick_edge" : "click_edge");
        if (isDblTap) {
          renderAll({ focusLabel: true });
        } else {
          focusCanvas();
          renderAll();
        }
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
        state.lastNodeTap = null;
        pushHistory(drag.snapshot);
        renderAll();
        notifyChanged();
      } else {
        const now = Date.now();
        const isDblTap =
          state.lastNodeTap &&
          state.lastNodeTap.id === drag.nodeId &&
          now - state.lastNodeTap.at <= 380;
        state.lastNodeTap = isDblTap ? null : { id: drag.nodeId, at: now };
        renderAll({ focusLabel: Boolean(isDblTap) });
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
      if (ev.target.matches("[data-sd-notes]")) {
        state.designNotes = safeSliceChars(ev.target.value, MAX_DESIGN_NOTES_CHARS);
        renderNotes();
        notifyChanged();
        return;
      }
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

      if (inEditable) {
        // Pressing Enter or Escape in the inline inspector commits the edit and returns
        // focus to the whiteboard canvas so keyboard shortcuts (like Backspace) work immediately.
        if (
          (ev.key === "Enter" || ev.key === "Escape") &&
          active.matches?.("[data-sd-edit-label], [data-sd-edit-note], [data-sd-edit-edge-label]")
        ) {
          ev.preventDefault();
          active.blur?.();
          focusCanvas();
        }
        return;
      }

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
        // Don't delete a node/edge when a toolbar button or interactive control has focus
        const tag = active.tagName;
        if (tag === "BUTTON" || tag === "A" || tag === "SUMMARY") return;
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
          design_notes: state.designNotes,
          checked_requirements: Array.from(state.checkedReqs).slice(0, 100),
        };
      },
      getStateSnapshot() {
        const inter = state.interaction;
        const hasNotes = state.designNotes.trim().length > 0;
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
          notesOpen: state.notesOpen,
          hasNotes,
          allowedActions: deriveDesignAllowedActions({
            interaction: inter,
            nodeCount: state.graph.nodes.length,
            edgeCount: state.graph.edges.length,
            historyLen: state.history.length,
            futureLen: state.future.length,
            hasNotes,
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
