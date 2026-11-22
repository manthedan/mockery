/* Sandboxed JavaScript Execution Worker for Coding Interview Studio (Milestone 4)
 * Served with strict Worker CSP (`default-src 'none'; script-src 'unsafe-eval'; connect-src 'none';
 * worker-src 'none'; child-src 'none'`). Spawned fresh per submission; host terminates on completion
 * or timeout. Tracks thrown values independently of truthiness and bounds output allocation.
 */

"use strict";

const _postMessage = self.postMessage.bind(self);
const _setTimeout = self.setTimeout.bind(self);
const _clearTimeout = self.clearTimeout.bind(self);
const _perfNow =
  self.performance && typeof self.performance.now === "function"
    ? self.performance.now.bind(self.performance)
    : () => Date.now();

const BLOCKED_GLOBALS = [
  "fetch",
  "XMLHttpRequest",
  "WebSocket",
  "EventSource",
  "importScripts",
  "indexedDB",
  "caches",
  "BroadcastChannel",
  "Worker",
  "SharedWorker",
  "MessageChannel",
  "navigator",
  "postMessage",
  "close",
  "localStorage",
  "sessionStorage",
  "FileReaderSync",
];

function disableHostApis() {
  for (const name of BLOCKED_GLOBALS) {
    try {
      Object.defineProperty(self, name, {
        value: undefined,
        writable: false,
        configurable: false,
      });
    } catch {
      try {
        self[name] = undefined;
      } catch {
        // Non-configurable built-ins are still blocked by Worker CSP + scope shadowing
      }
    }
  }
}

const MAX_LOG_LINES = 200;
const MAX_OUTPUT_CHARS = 16000;
const MAX_ERROR_CHARS = 8000;
const MAX_TIMERS_PER_RUN = 250;

function takeText(text, budget) {
  if (budget.remaining <= 0) {
    budget.truncated = true;
    return "";
  }
  if (text.length <= budget.remaining) {
    budget.remaining -= text.length;
    return text;
  }
  const slice = text.slice(0, budget.remaining);
  budget.remaining = 0;
  budget.truncated = true;
  return `${slice}…`;
}

function formatValue(val, budget, depth = 0, seen = new WeakSet()) {
  if (budget.remaining <= 0) {
    budget.truncated = true;
    return "…";
  }
  if (val === null) return takeText("null", budget);
  if (val === undefined) return takeText("undefined", budget);
  const t = typeof val;
  if (t === "string") {
    const raw = depth === 0 ? val : JSON.stringify(val.slice(0, budget.remaining + 4));
    return takeText(raw, budget);
  }
  if (t === "number" || t === "boolean" || t === "symbol") return takeText(String(val), budget);
  if (t === "bigint") return takeText(`${val}n`, budget);
  if (t === "function") return takeText(`[Function ${val.name || "anonymous"}]`, budget);
  if (val instanceof Error) {
    const errText = val.stack || `${val.name}: ${val.message}`;
    return takeText(String(errText), budget);
  }
  if (depth > 3) return takeText("[Object]", budget);
  if (seen.has(val)) return takeText("[Circular]", budget);
  seen.add(val);
  try {
    if (Array.isArray(val)) {
      const parts = [takeText("[", budget)];
      const limit = Math.min(val.length, 50);
      for (let i = 0; i < limit && budget.remaining > 0; i++) {
        if (i > 0) parts.push(takeText(", ", budget));
        parts.push(formatValue(val[i], budget, depth + 1, seen));
      }
      if (val.length > 50 && budget.remaining > 0) {
        parts.push(takeText(`, … +${val.length - 50} more`, budget));
      }
      parts.push(takeText("]", budget));
      return parts.join("");
    }
    if (val instanceof Map) {
      const parts = [takeText(`Map(${val.size}) { `, budget)];
      let count = 0;
      for (const [k, v] of val.entries()) {
        if (budget.remaining <= 0) break;
        if (count >= 30) {
          parts.push(takeText(`, … +${val.size - 30} more`, budget));
          break;
        }
        if (count > 0) parts.push(takeText(", ", budget));
        parts.push(formatValue(k, budget, depth + 1, seen));
        parts.push(takeText(" => ", budget));
        parts.push(formatValue(v, budget, depth + 1, seen));
        count += 1;
      }
      parts.push(takeText(" }", budget));
      return parts.join("");
    }
    if (val instanceof Set) {
      const parts = [takeText(`Set(${val.size}) { `, budget)];
      let count = 0;
      for (const v of val.values()) {
        if (budget.remaining <= 0) break;
        if (count >= 30) {
          parts.push(takeText(`, … +${val.size - 30} more`, budget));
          break;
        }
        if (count > 0) parts.push(takeText(", ", budget));
        parts.push(formatValue(v, budget, depth + 1, seen));
        count += 1;
      }
      parts.push(takeText(" }", budget));
      return parts.join("");
    }
    const keys = Object.keys(val);
    const parts = [takeText("{ ", budget)];
    const limit = Math.min(keys.length, 30);
    for (let i = 0; i < limit && budget.remaining > 0; i++) {
      if (i > 0) parts.push(takeText(", ", budget));
      parts.push(takeText(`${keys[i]}: `, budget));
      parts.push(formatValue(val[keys[i]], budget, depth + 1, seen));
    }
    if (keys.length > 30 && budget.remaining > 0) {
      parts.push(takeText(`, … +${keys.length - 30} more`, budget));
    }
    parts.push(takeText(" }", budget));
    return parts.join("");
  } catch {
    return takeText(Object.prototype.toString.call(val), budget);
  }
}

function formatThrownError(err) {
  const budget = { remaining: MAX_ERROR_CHARS, truncated: false };
  if (err instanceof Error) {
    const msg = `${err.name || "Error"}: ${err.message || ""}`.trim();
    const stack = typeof err.stack === "string" ? err.stack : "";
    if (!stack) return takeText(msg, budget);
    const cleaned = stack.split("\n").slice(0, 8).join("\n");
    const full = cleaned.includes(err.message) ? cleaned : `${msg}\n${cleaned}`;
    return takeText(full, budget);
  }
  return `Thrown: ${formatValue(err, budget)}`;
}

let hasRun = false;

self.addEventListener("message", async (ev) => {
  if (hasRun) return;
  const payload = ev.data || {};
  if (payload.type !== "run") return;
  hasRun = true;

  const runId = payload.runId || "";
  const code = typeof payload.code === "string" ? payload.code : "";
  const tests = typeof payload.tests === "string" ? payload.tests : "";

  const logs = [];
  const logBudget = { remaining: MAX_OUTPUT_CHARS, truncated: false };

  function appendLog(prefix, args) {
    if (logs.length >= MAX_LOG_LINES || logBudget.remaining <= 0) {
      logBudget.truncated = true;
      return;
    }
    const parts = [];
    if (prefix) parts.push(takeText(`${prefix} `, logBudget));
    for (let i = 0; i < args.length && logBudget.remaining > 0; i++) {
      if (i > 0) parts.push(takeText(" ", logBudget));
      parts.push(formatValue(args[i], logBudget));
    }
    logs.push(parts.join(""));
    logBudget.remaining = Math.max(0, logBudget.remaining - 1);
  }

  const sandboxConsole = Object.freeze({
    log: (...args) => appendLog("", args),
    info: (...args) => appendLog("", args),
    warn: (...args) => appendLog("[warn]", args),
    error: (...args) => appendLog("[error]", args),
    assert: (cond, ...args) => {
      if (!cond) {
        const b = { remaining: 500, truncated: false };
        const msg = args.length
          ? args.map((a) => formatValue(a, b)).join(" ")
          : "Assertion failed";
        throw new Error(msg);
      }
    },
  });

  let hasAsyncError = false;
  let asyncError = undefined;
  function recordAsyncError(err) {
    if (!hasAsyncError) {
      hasAsyncError = true;
      asyncError = err;
    }
  }

  self.addEventListener("unhandledrejection", (rejEv) => {
    if (typeof rejEv.preventDefault === "function") rejEv.preventDefault();
    recordAsyncError(rejEv.reason);
  });

  const pendingTimers = new Set();
  let timerCount = 0;
  let drainResolver = null;

  function checkDrained() {
    if (pendingTimers.size === 0 && drainResolver) {
      const r = drainResolver;
      drainResolver = null;
      r();
    }
  }

  function trackedSetTimeout(fn, delay = 0, ...args) {
    if (typeof fn !== "function") {
      throw new TypeError("setTimeout requires a function callback in the coding sandbox.");
    }
    timerCount += 1;
    if (timerCount > MAX_TIMERS_PER_RUN) {
      throw new Error(`Exceeded maximum timer limit (${MAX_TIMERS_PER_RUN}) in sandbox.`);
    }
    const safeDelay = Math.max(0, Math.min(Number(delay) || 0, 2500));
    const handle = _setTimeout(() => {
      pendingTimers.delete(handle);
      try {
        const res = fn(...args);
        if (res && typeof res.then === "function") {
          res.catch((err) => recordAsyncError(err));
        }
      } catch (err) {
        recordAsyncError(err);
      } finally {
        checkDrained();
      }
    }, safeDelay);
    pendingTimers.add(handle);
    return handle;
  }

  function trackedClearTimeout(handle) {
    if (pendingTimers.has(handle)) {
      pendingTimers.delete(handle);
      _clearTimeout(handle);
      checkDrained();
    }
  }

  function blockedSetInterval() {
    throw new Error("setInterval is disabled in the coding sandbox; use setTimeout or async/await.");
  }

  // Lock down worker globals before evaluating untrusted candidate code
  disableHostApis();

  const startedAt = _perfNow();
  let hasRunError = false;
  let runError = undefined;

  try {
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    const combinedSource = `"use strict";\n${code}\n;\n${tests}\n`;
    const shadowNames = [
      "console",
      "setTimeout",
      "clearTimeout",
      "setInterval",
      "clearInterval",
      "self",
      "globalThis",
      "window",
      ...BLOCKED_GLOBALS,
    ];
    const shadowValues = [
      sandboxConsole,
      trackedSetTimeout,
      trackedClearTimeout,
      blockedSetInterval,
      () => {},
      undefined,
      undefined,
      undefined,
      ...BLOCKED_GLOBALS.map(() => undefined),
    ];

    const fn = new AsyncFunction(...shadowNames, combinedSource);
    await fn(...shadowValues);

    // Wait for any remaining tracked timers / microtasks to settle
    for (let rounds = 0; rounds < 12; rounds++) {
      await Promise.resolve();
      if (pendingTimers.size > 0) {
        await new Promise((resolve) => {
          drainResolver = resolve;
          checkDrained();
        });
      }
      await new Promise((resolve) => _setTimeout(resolve, 0));
      if (pendingTimers.size === 0) break;
    }
  } catch (err) {
    hasRunError = true;
    runError = err;
  } finally {
    for (const h of pendingTimers) _clearTimeout(h);
    pendingTimers.clear();
  }

  const failed = hasRunError || hasAsyncError;
  const finalError = hasRunError ? runError : asyncError;
  const durationMs = Math.max(0, Math.round(_perfNow() - startedAt));
  if (logBudget.truncated) {
    logs.push("… [output truncated]");
  }

  _postMessage({
    type: "result",
    runId,
    ok: !failed,
    passed: !failed,
    stdout: logs.join("\n").slice(0, MAX_OUTPUT_CHARS + 64),
    error: failed ? formatThrownError(finalError).slice(0, MAX_ERROR_CHARS + 64) : "",
    duration_ms: durationMs,
  });
});
