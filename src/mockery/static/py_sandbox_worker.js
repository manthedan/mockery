/* Sandboxed Offline Python Execution Worker (Pyodide WASM — Milestone 4)
 * Served with strict Worker CSP restricting `script-src` and `connect-src` exclusively to
 * `/api/addons/pyodide/` (zero external CDN or localhost API access). Spawned fresh per
 * submission and terminated by the host on every completion/timeout so interpreter state,
 * modules, builtins, virtual FS, and event loops never leak across runs.
 */

"use strict";

const _postMessage = self.postMessage.bind(self);
const _perfNow =
  self.performance && typeof self.performance.now === "function"
    ? self.performance.now.bind(self.performance)
    : () => Date.now();

const LOCAL_PYODIDE_BASE = "/api/addons/pyodide/";
const MAX_ERROR_CHARS = 8000;

function disableRuntimeHostApis() {
  const blocked = [
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
    "postMessage",
    "close",
    "navigator",
  ];
  for (const name of blocked) {
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
        // Non-configurable built-ins remain restricted by Worker CSP
      }
    }
  }
}

const PY_SANDBOX_BOOTSTRAP = `
import ast
import contextlib
import io
import traceback

class _BoundedWriter(io.TextIOBase):
    def __init__(self, limit: int = 16000):
        self._limit = limit
        self._used = 0
        self._parts = []
        self.truncated = False

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        n = len(s)
        if n == 0:
            return 0
        rem = self._limit - self._used
        if rem <= 0:
            self.truncated = True
            return n
        if n > rem:
            self._parts.append(s[:rem])
            self._used += rem
            self.truncated = True
        else:
            self._parts.append(s)
            self._used += n
        return n

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        text = "".join(self._parts)
        if self.truncated:
            text += "\\n… [output truncated]"
        return text

class _TopLevelAsyncioRunRewriter(ast.NodeTransformer):
    """Rewrite only top-level asyncio.run(expr) calls in legacy test harnesses without touching nested functions/classes."""
    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
            and node.func.attr == "run"
            and len(node.args) == 1
            and not node.keywords
        ):
            return ast.copy_location(ast.Await(value=node.args[0]), node)
        return node

async def __mockery_run_sandbox(code_src: str, tests_src: str) -> dict:
    out_buf = _BoundedWriter(16000)
    err_buf = _BoundedWriter(8000)
    ns = {"__name__": "__main__"}
    ok = True
    error_text = ""
    try:
        # Compile & execute candidate solution completely untouched
        sol_compiled = compile((code_src or "") + "\\n", filename="<solution>", mode="exec")
        # Parse test suite and allow top-level await (plus top-level asyncio.run compatibility)
        test_tree = ast.parse((tests_src or "") + "\\n", filename="<tests>", mode="exec")
        test_tree = _TopLevelAsyncioRunRewriter().visit(test_tree)
        ast.fix_missing_locations(test_tree)
        test_compiled = compile(
            test_tree,
            filename="<tests>",
            mode="exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(sol_compiled, ns, ns)
            maybe_coro = eval(test_compiled, ns, ns)
            if maybe_coro is not None and hasattr(maybe_coro, "__await__"):
                await maybe_coro
    except BaseException as exc:
        ok = False
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=12)
        buf = []
        used = 0
        for part in tb_lines:
            rem = 8000 - used
            if rem <= 0:
                buf.append("\\n… [traceback truncated]")
                break
            if len(part) > rem:
                buf.append(part[:rem] + "\\n… [traceback truncated]")
                used = 8000
                break
            buf.append(part)
            used += len(part)
        error_text = "".join(buf)

    stdout = out_buf.getvalue()
    stderr = err_buf.getvalue()
    if stderr:
        stdout = (stdout + ("\\n" if stdout else "") + stderr)
    if len(stdout) > 16100:
        stdout = stdout[:16000] + "\\n… [output truncated]"
    return {"ok": ok, "stdout": stdout.rstrip("\\n"), "error": error_text.rstrip("\\n")}
`;

let hasRun = false;

self.addEventListener("message", async (ev) => {
  if (hasRun) return;
  const payload = ev.data || {};
  if (payload.type !== "run") return;
  hasRun = true;

  const runId = payload.runId || "";
  const code = typeof payload.code === "string" ? payload.code : "";
  const tests = typeof payload.tests === "string" ? payload.tests : "";

  try {
    _postMessage({ type: "status", runId, status: "loading" });
    importScripts(`${LOCAL_PYODIDE_BASE}pyodide.js`);
    if (typeof self.loadPyodide !== "function") {
      throw new Error("Local Pyodide bootstrap did not expose loadPyodide().");
    }
    const py = await self.loadPyodide({
      indexURL: LOCAL_PYODIDE_BASE,
    });
    // Lock down worker messaging, script loading, and network APIs before running untrusted Python
    disableRuntimeHostApis();

    await py.runPythonAsync(PY_SANDBOX_BOOTSTRAP);
    _postMessage({ type: "ready", runId });

    const startedAt = _perfNow();
    const runner = py.globals.get("__mockery_run_sandbox");
    const pyRes = await runner(code, tests);
    const res = pyRes.toJs({ dict_converter: Object.fromEntries });
    pyRes.destroy?.();
    runner.destroy?.();
    const durationMs = Math.max(0, Math.round(_perfNow() - startedAt));

    _postMessage({
      type: "result",
      runId,
      ok: Boolean(res.ok),
      passed: Boolean(res.ok),
      stdout: String(res.stdout || "").slice(0, 16100),
      error: String(res.error || "").slice(0, MAX_ERROR_CHARS + 64),
      duration_ms: durationMs,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    _postMessage({
      type: "result",
      runId,
      ok: false,
      passed: false,
      stdout: "",
      error: msg.slice(0, MAX_ERROR_CHARS),
      duration_ms: 0,
    });
  }
});
