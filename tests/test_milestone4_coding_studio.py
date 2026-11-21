from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from mockery.pyodide_pack import PYODIDE_CORE_FILES
from mockery.server import create_app


def test_coding_session_workspace_llm_context_and_markdown_export(tmp_path: Path, monkeypatch):
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    runtime.llm_config.provider = "none"
    runtime.llm_config.api_key = ""

    seen_ws: list[dict] = []
    orig_reply = runtime.llm.generate_practice_interviewer_reply

    async def _spy_reply(**kwargs):
        ws = kwargs.get("workspace_context") or {}
        if ws.get("code") or ws.get("test_output"):
            seen_ws.append(dict(ws))
        return await orig_reply(**kwargs)

    monkeypatch.setattr(runtime.llm, "generate_practice_interviewer_reply", _spy_reply)

    client = TestClient(app)

    # 1. Start a coding practice session pinned to code_ttl_lru_01
    start_res = client.post(
        "/api/practice/start",
        json={
            "role_title": "Staff Software Engineer",
            "company": "Acme AI",
            "interview_type": "coding",
            "question_source": "bank_only",
            "target_level": "L5",
            "selected_question_id": "code_ttl_lru_01",
        },
    )
    assert start_res.status_code == 200
    sess = start_res.json()["session"]
    sid = sess["id"]
    assert len(sess["questions"]) == 1
    q0 = sess["questions"][0]
    assert q0["bank_question_id"] == "code_ttl_lru_01"
    assert "TTLLRUCache" in q0["starter_code"]["javascript"]
    assert "TTLLRUCache" in q0["starter_code"]["python"]
    assert "ALL TESTS PASSED" in q0["tests"]["javascript"]
    assert len(q0["follow_ups"]) >= 2

    # 2. Invalid code_by_lang / tests_by_lang shapes rejected with 400
    bad_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={"workspace_context": {"code_by_lang": {"javascript": 123}}},
    )
    assert bad_res.status_code == 400

    # 3. Save coding workspace (including triple backticks in a comment to verify Markdown fence safety)
    solution_js = (
        "class TTLLRUCache {\n"
        "  // Example ``` fenced comment ```\n"
        "  constructor(capacity) { this.capacity = capacity; this.map = new Map(); }\n"
        "}\n"
    )
    test_output = "[PASSED] JavaScript (8 ms)\n\nSTDOUT:\nALL TESTS PASSED"
    ws_res = client.post(
        f"/api/practice/{sid}/workspace",
        json={
            "workspace_context": {
                "language": "javascript",
                "code": solution_js,
                "test_output": test_output,
                "code_by_lang": {"javascript": solution_js, "python": q0["starter_code"]["python"]},
                "tests_by_lang": {"javascript": q0["tests"]["javascript"]},
            }
        },
    )
    assert ws_res.status_code == 200

    # 4. Reopen session and confirm workspace persisted
    loaded = client.get(f"/api/practice/sessions/{sid}")
    assert loaded.status_code == 200
    lws = loaded.json()["session"]["workspace_context"]
    assert lws["language"] == "javascript"
    assert lws["code"] == solution_js
    assert lws["test_output"] == test_output

    # 5. Ask AI to review code via /turn -> code & test_output reach LLM
    turn_res = client.post(
        f"/api/practice/{sid}/turn",
        json={
            "text": "Please review my current code approach.",
            "expected_question_id": q0["id"],
        },
    )
    assert turn_res.status_code == 200
    assert seen_ws
    assert "TTLLRUCache" in seen_ws[-1]["code"]
    assert "ALL TESTS PASSED" in seen_ws[-1]["test_output"]

    # 6. Debrief + Markdown export includes Coding Studio Solution with safe dynamic fences
    debrief_res = client.post(f"/api/practice/{sid}/debrief")
    assert debrief_res.status_code == 200
    md_res = client.get(f"/api/practice/{sid}/markdown")
    assert md_res.status_code == 200
    assert "## Coding Studio Solution (javascript)" in md_res.text
    assert "````javascript\n" in md_res.text
    assert "**Latest Test Output:**" in md_res.text
    assert "ALL TESTS PASSED" in md_res.text

    # 7. coding_studio add-on flag toggles cleanly
    off_res = client.post("/api/addons/modules", json={"modules": {"coding_studio": False}})
    assert off_res.status_code == 200
    assert off_res.json()["modules_map"]["coding_studio"] is False


def test_sandbox_worker_csp_headers_and_pyodide_integrity_symlink_and_etag(tmp_path: Path):
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime
    client = TestClient(app)

    # 1. Both sandbox workers are served with strict Worker Content-Security-Policy headers
    js_w = client.get("/static/js_sandbox_worker.js")
    assert js_w.status_code == 200
    js_csp = js_w.headers.get("content-security-policy", "")
    assert "default-src 'none'" in js_csp
    assert "connect-src 'none'" in js_csp
    assert "worker-src 'none'" in js_csp
    assert "script-src 'unsafe-eval'" in js_csp

    py_w = client.get("/static/py_sandbox_worker.js")
    assert py_w.status_code == 200
    py_csp = py_w.headers.get("content-security-policy", "")
    assert "default-src 'none'" in py_csp
    assert "worker-src 'none'" in py_csp
    assert "/api/addons/pyodide/" in py_csp
    assert "wasm-unsafe-eval" in py_csp

    # 2. Initially available=True, installed=False, asset route returns 404
    addons_res = client.get("/api/addons")
    assert addons_res.status_code == 200
    packs = {p["id"]: p for p in addons_res.json()["packs"]}
    assert packs["pyodide_python"]["available"] is True
    assert packs["pyodide_python"]["installed"] is False
    assert packs["pyodide_python"]["state"] == "not_installed"
    assert client.get("/api/addons/pyodide/pyodide.js").status_code == 404

    # 3. Populate verified core files (using synthetic expected hashes for fast offline test)
    pack_dir = runtime.pyodide.pack_dir
    pack_dir.mkdir(parents=True, exist_ok=True)
    test_hashes: dict[str, str] = {}
    for name in PYODIDE_CORE_FILES:
        content = b"\x00asm\x01\x00\x00\x00" if name.endswith(".wasm") else f"// {name}\n".encode()
        (pack_dir / name).write_bytes(content)
        test_hashes[name] = hashlib.sha256(content).hexdigest()
    runtime.pyodide.expected_hashes = test_hashes

    # Without manifest.json, unverified files must NOT count as installed
    assert runtime.pyodide.is_installed() is False
    assert client.get("/api/addons/pyodide/pyodide.js").status_code == 404

    # Write verified manifest -> now installed=True
    runtime.pyodide.write_verified_manifest()
    assert runtime.pyodide.is_installed() is True

    (pack_dir.parent / "secret.js").write_text("SECRET_DO_NOT_SERVE", encoding="utf-8")
    (pack_dir / "extra.js").write_text("NOT_IN_ALLOWLIST", encoding="utf-8")

    # 4. Verified WASM/JS serve with ETag + Cache-Control: no-cache + 304 revalidation
    wasm_res = client.get("/api/addons/pyodide/pyodide.asm.wasm")
    assert wasm_res.status_code == 200
    assert wasm_res.headers["content-type"].startswith("application/wasm")
    assert wasm_res.headers["cache-control"] == "no-cache"
    assert wasm_res.headers["x-content-type-options"] == "nosniff"
    etag = wasm_res.headers["etag"]
    assert wasm_res.content == b"\x00asm\x01\x00\x00\x00"

    reval_res = client.get("/api/addons/pyodide/pyodide.asm.wasm", headers={"If-None-Match": etag})
    assert reval_res.status_code == 304

    # 5. Exact allowlist + path traversal + unknown file rejection
    assert client.get("/api/addons/pyodide/..%2Fsecret.js").status_code == 400
    assert client.get("/api/addons/pyodide/extra.js").status_code == 400
    assert client.get("/api/addons/pyodide/manifest.json").status_code == 400
    for bad_rel in ("", "../secret.js", "/etc/passwd", "a\\b.js", "bad\x00.js", "extra.js"):
        try:
            runtime.pyodide.open_asset(bad_rel)
            raise AssertionError(f"Expected ValueError for {bad_rel!r}")
        except ValueError:
            pass

    # 6. Symlinked pack_dir is strictly rejected (never trusted as root!)
    real_elsewhere = tmp_path / "outside_dir"
    real_elsewhere.mkdir()
    runtime.pyodide.uninstall()
    pack_dir.symlink_to(real_elsewhere)
    assert runtime.pyodide.is_installed() is False
    assert runtime.pyodide.open_asset("pyodide.js") is None
    assert client.get("/api/addons/pyodide/pyodide.js").status_code == 404
    runtime.pyodide.uninstall()
    assert not pack_dir.exists() and real_elsewhere.exists()


def test_addon_manager_serializes_concurrent_install_and_uninstall(tmp_path: Path):
    app = create_app(project_root=tmp_path, preload_asr=False)
    runtime = app.state.runtime

    installed = {"flag": False}
    install_calls = {"count": 0}
    events: list[str] = []

    async def slow_install():
        install_calls["count"] += 1
        events.append("install_start")
        await asyncio.sleep(0.05)
        installed["flag"] = True
        events.append("install_end")

    async def fast_uninstall():
        events.append("uninstall_start")
        installed["flag"] = False
        events.append("uninstall_end")
        return True

    runtime.addons.register_pack_handler(
        "pyodide_python",
        is_installed=lambda: installed["flag"],
        install=slow_install,
        uninstall=fast_uninstall,
        available=True,
    )

    async def _run():
        # Two overlapping installs deduplicate and both await completion with state == installed
        t1 = asyncio.create_task(runtime.addons.install_pack("pyodide_python"))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(runtime.addons.install_pack("pyodide_python"))
        # Uninstall launched mid-install must wait until install finishes, never interleaved
        t3 = asyncio.create_task(runtime.addons.uninstall_pack("pyodide_python"))
        s1, s2, s3 = await asyncio.gather(t1, t2, t3)
        return s1, s2, s3

    _s1, _s2, s3 = asyncio.run(_run())
    assert install_calls["count"] == 1
    assert events == ["install_start", "install_end", "uninstall_start", "uninstall_end"]
    packs3 = {p["id"]: p for p in s3["packs"]}
    assert packs3["pyodide_python"]["installed"] is False


def test_coding_studio_helpers_and_real_bank_suites_in_js_and_python_sandboxes():
    repo_root = Path(__file__).resolve().parents[1]
    bank = json.loads((repo_root / "src/mockery/banks/coding.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in bank["questions"]}
    async_q = by_id["code_async_pool_01"]

    # 1. Node test: coding_studio helpers + real js_sandbox_worker.js against real bank suite & falsy throws
    script = r"""
import fs from 'node:fs';
import vm from 'node:vm';
import {
  normalizeCodingWorkspace,
  safeClipText,
  hasExecutableTests,
  formatTestRunSummary,
  isPyodidePackInstalled,
} from './src/mockery/static/coding_studio.js';

const bank = JSON.parse(fs.readFileSync('./src/mockery/banks/coding.json', 'utf8'));
const asyncQ = bank.questions.find((q) => q.id === 'code_async_pool_01');

// 1a. UTF-8 byte clipping without splitting multibyte emojis
const emojiStr = 'a' + '🚀'.repeat(10); // 1 + 40 = 41 UTF-8 bytes
const clipped = safeClipText(emojiStr, 7); // 'a' (1) + 1 emoji (4) = 5 bytes; 2nd emoji would be 9 > 7
if (Buffer.byteLength(clipped, 'utf8') > 7) throw new Error('UTF-8 byte limit exceeded');
if (clipped !== 'a🚀') throw new Error('Unexpected clipped emoji string: ' + JSON.stringify(clipped));

// 1b. hasExecutableTests detects empty/comment-only suites
if (hasExecutableTests('   // only comment\n/* block */\n', 'javascript')) {
  throw new Error('Comment-only JS suite should not be executable');
}
if (hasExecutableTests('  # only python comment\n\n', 'python')) {
  throw new Error('Comment-only Python suite should not be executable');
}
if (!hasExecutableTests(asyncQ.tests.javascript, 'javascript')) {
  throw new Error('Real curated JS suite should be executable');
}

// 1c. Default language selection & summary formatting
const ws = normalizeCodingWorkspace(asyncQ, {}, { packs: [{ id: 'pyodide_python', installed: false }] });
if (ws.language !== 'javascript') throw new Error('Expected javascript default without Pyodide');
const wsPy = normalizeCodingWorkspace(asyncQ, {}, { packs: [{ id: 'pyodide_python', installed: true }] });
if (wsPy.language !== 'python' || !isPyodidePackInstalled({ packs: [{ id: 'pyodide_python', installed: true }] })) {
  throw new Error('Expected python default when Pyodide installed');
}
const formatted = formatTestRunSummary({ language: 'javascript', ok: true, stdout: 'ALL TESTS PASSED', error: '', durationMs: 9 });
if (!formatted.includes('[PASSED]')) throw new Error('Expected [PASSED] header');

// 2. Run real js_sandbox_worker.js inside isolated VM context
const workerSource = fs.readFileSync('./src/mockery/static/js_sandbox_worker.js', 'utf8');

function runInWorker(code, tests) {
  return new Promise((resolve, reject) => {
    const listeners = { message: [], unhandledrejection: [] };
    const ctx = {
      setTimeout,
      clearTimeout,
      performance: { now: () => performance.now() },
      fetch: () => { throw new Error('HOST_FETCH_REACHED'); },
      XMLHttpRequest: function() {},
      WebSocket: function() {},
      importScripts: () => {},
      postMessage: (msg) => {
        if (msg && msg.type === 'result') resolve(msg);
      },
      addEventListener: (type, fn) => {
        if (listeners[type]) listeners[type].push(fn);
      },
    };
    ctx.self = ctx;
    vm.createContext(ctx);
    vm.runInContext(workerSource, ctx);
    for (const fn of listeners.message) {
      Promise.resolve(fn({ data: { type: 'run', runId: 'r1', code, tests } })).catch(reject);
    }
  });
}

// 2a. Real curated bank test for code_async_pool_01 (including fail-fast worker rejection!)
const goodPoolJs = `
async function runWithLimit(items, worker, limit) {
  const out = new Array(items.length);
  let idx = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (idx < items.length) {
      const cur = idx++;
      out[cur] = await worker(items[cur]);
    }
  });
  await Promise.all(runners);
  return out;
}
`;
const rAsync = await runInWorker(goodPoolJs, asyncQ.tests.javascript);
if (!rAsync.ok || !rAsync.stdout.includes('ALL TESTS PASSED')) {
  throw new Error('Curated async pool JS suite failed: ' + JSON.stringify(rAsync));
}

// 2b. Falsy thrown values (throw false, throw 0, throw "") MUST fail (ok: false, passed: false)
for (const expr of ['throw false;', 'throw 0;', 'throw "";', 'throw null;']) {
  const r = await runInWorker('', expr);
  if (r.ok || r.passed) {
    throw new Error('Falsy throw falsely passed: ' + expr + ' -> ' + JSON.stringify(r));
  }
}

// 2c. Neutered APIs + circular object + bounded massive stdout/error
const rSec = await runInWorker(
  `const cyc = { a: 1 }; cyc.self = cyc; console.log(cyc);
   for (let i = 0; i < 500; i++) console.log('x'.repeat(200));
   if (typeof fetch !== 'undefined' || typeof XMLHttpRequest !== 'undefined' || typeof WebSocket !== 'undefined' || typeof importScripts !== 'undefined') {
     throw new Error('Host API leaked');
   }`,
  `console.log('DONE');`
);
if (!rSec.ok || !rSec.stdout.includes('[Circular]') || rSec.stdout.length > 17000) {
  throw new Error('Security or output cap failed: len=' + rSec.stdout.length);
}

console.log(JSON.stringify({ ok: true }));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert json.loads(proc.stdout.strip().splitlines()[-1])["ok"] is True

    # 2. Extract and run the real Python bootstrap from py_sandbox_worker.js
    py_worker_src = (repo_root / "src/mockery/static/py_sandbox_worker.js").read_text(encoding="utf-8")
    m = re.search(r"const PY_SANDBOX_BOOTSTRAP = `([\s\S]*?)`;", py_worker_src)
    assert m, "Could not locate PY_SANDBOX_BOOTSTRAP in py_sandbox_worker.js"
    # Unescape JS template literal escaped backslashes
    py_bootstrap = m.group(1).replace("\\\\n", "\\n")
    ns: dict = {}
    exec(compile(py_bootstrap, "<py_sandbox_bootstrap>", "exec"), ns, ns)
    run_sandbox = ns["__mockery_run_sandbox"]

    # 2a. Curated Python async pool suite passes (including fail-fast exception propagation!)
    good_pool_py = """
import asyncio

async def run_with_limit(items, worker, limit: int):
    sem = asyncio.Semaphore(max(1, limit))
    async def guarded(item):
        async with sem:
            return await worker(item)
    return await asyncio.gather(*(guarded(x) for x in items))

# Sync helper containing asyncio.run(...) MUST NOT break compilation!
def unused_sync_helper():
    return asyncio.run(asyncio.sleep(0))
"""
    res_py = asyncio.run(run_sandbox(good_pool_py, async_q["tests"]["python"]))
    assert res_py["ok"] is True, f"Python sandbox failed: {res_py}"
    assert "ALL TESTS PASSED" in res_py["stdout"]

    # 2b. Huge print loop is bounded without blowing up memory
    res_big = asyncio.run(run_sandbox("for _ in range(5000): print('A' * 100)", "assert True"))
    assert res_big["ok"] is True
    assert len(res_big["stdout"]) <= 16150
    assert "truncated" in res_big["stdout"]
