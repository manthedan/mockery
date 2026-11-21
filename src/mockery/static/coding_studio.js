/* Coding Interview Studio (Milestone 4)
 * Offline-first code editor + sandboxed JS/Python test runner plugin for Practice Studio.
 * Zero external CDN dependencies: both JS and Python execute in isolated single-use Web Workers
 * served with strict Worker CSP headers, and Python loads strictly from `/api/addons/pyodide/`.
 */

import { createMachine, createPackOpTracker, formatApiDetail } from "./state_machines.js";

const JS_TIMEOUT_MS = 3500;
const PY_BOOT_TIMEOUT_MS = 20000;
const PY_RUN_TIMEOUT_MS = 4000;
// Bounded in UTF-8 bytes so all 3 code buffers + 2 test buffers + active copy + test output
// always stay well beneath the server's 200 KB workspace payload limit even for multibyte text.
const MAX_CODE_UTF8_BYTES = 24000;
const MAX_TEST_OUTPUT_UTF8_BYTES = 4000;

export const CODING_LANGUAGES = [
  { id: "javascript", label: "JavaScript", badge: "JS · Built-in", indent: "  " },
  { id: "python", label: "Python", badge: "PY · Pyodide", indent: "    " },
  { id: "pseudocode", label: "Whiteboard", badge: "Pseudocode", indent: "  " },
];

const VALID_LANG_IDS = new Set(CODING_LANGUAGES.map((l) => l.id));

function stripInvalidSurrogates(input) {
  const s = typeof input === "string" ? input : "";
  if (!s) return "";
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = i + 1 < s.length ? s.charCodeAt(i + 1) : 0;
      if (next >= 0xdc00 && next <= 0xdfff) {
        out += s[i] + s[i + 1];
        i += 1;
      }
      continue;
    }
    if (code >= 0xdc00 && code <= 0xdfff) continue;
    out += s[i];
  }
  return out;
}

function utf8ByteLengthOfCodePoint(cp) {
  if (cp <= 0x7f) return 1;
  if (cp <= 0x7ff) return 2;
  if (cp <= 0xffff) return 3;
  return 4;
}

export function safeClipText(text, maxUtf8Bytes) {
  const clean = stripInvalidSurrogates(typeof text === "string" ? text : "");
  const limit = Math.max(0, Number(maxUtf8Bytes) || 0);
  if (!clean || limit === 0) return "";
  // Fast path: even if every UTF-16 code unit took 3 UTF-8 bytes, it fits.
  if (clean.length * 3 <= limit) return clean;
  let bytes = 0;
  let endIdx = 0;
  for (let i = 0; i < clean.length; ) {
    const cp = clean.codePointAt(i);
    const step = cp > 0xffff ? 2 : 1;
    const b = utf8ByteLengthOfCodePoint(cp);
    if (bytes + b > limit) break;
    bytes += b;
    i += step;
    endIdx = i;
  }
  return clean.slice(0, endIdx);
}

export function hasExecutableTests(testsText, language) {
  const raw = typeof testsText === "string" ? testsText : "";
  if (!raw.trim()) return false;
  if (language === "python") {
    const stripped = raw
      .split("\n")
      .map((line) => line.replace(/#.*$/, "").trim())
      .filter(Boolean)
      .join("\n");
    return stripped.length > 0;
  }
  if (language === "javascript") {
    const noBlock = raw.replace(/\/\*[\s\S]*?\*\//g, "");
    const stripped = noBlock
      .split("\n")
      .map((line) => line.replace(/\/\/.*$/, "").trim())
      .filter(Boolean)
      .join("\n");
    return stripped.length > 0;
  }
  return false;
}

export function isPyodidePackInstalled(addons) {
  const packs = Array.isArray(addons?.packs) ? addons.packs : [];
  const py = packs.find((p) => p && p.id === "pyodide_python");
  return Boolean(py && (py.installed || py.state === "installed"));
}

export function detectDefaultLanguage(question, savedWorkspace, addons) {
  const savedLang = typeof savedWorkspace?.language === "string" ? savedWorkspace.language : "";
  if (VALID_LANG_IDS.has(savedLang)) return savedLang;
  const starter = question?.starter_code && typeof question.starter_code === "object" ? question.starter_code : {};
  if (isPyodidePackInstalled(addons) && typeof starter.python === "string" && starter.python.trim()) {
    return "python";
  }
  if (typeof starter.javascript === "string" && starter.javascript.trim()) {
    return "javascript";
  }
  if (typeof starter.python === "string" && starter.python.trim()) {
    return "python";
  }
  return "javascript";
}

export function normalizeCodingWorkspace(question, savedWorkspace, addons) {
  const starter =
    question?.starter_code && typeof question.starter_code === "object" ? question.starter_code : {};
  const tests = question?.tests && typeof question.tests === "object" ? question.tests : {};
  const saved = savedWorkspace && typeof savedWorkspace === "object" ? savedWorkspace : {};
  const savedCodeMap = saved.code_by_lang && typeof saved.code_by_lang === "object" ? saved.code_by_lang : {};
  const savedTestsMap =
    saved.tests_by_lang && typeof saved.tests_by_lang === "object" ? saved.tests_by_lang : {};

  const language = detectDefaultLanguage(question, saved, addons);

  const codeByLang = {
    javascript: safeClipText(
      typeof savedCodeMap.javascript === "string"
        ? savedCodeMap.javascript
        : typeof starter.javascript === "string"
        ? starter.javascript
        : "// Write your JavaScript solution here\n",
      MAX_CODE_UTF8_BYTES
    ),
    python: safeClipText(
      typeof savedCodeMap.python === "string"
        ? savedCodeMap.python
        : typeof starter.python === "string"
        ? starter.python
        : "# Write your Python solution here\n",
      MAX_CODE_UTF8_BYTES
    ),
    pseudocode: safeClipText(
      typeof savedCodeMap.pseudocode === "string"
        ? savedCodeMap.pseudocode
        : "// Outline your algorithm, invariants, and time/space complexity here\n",
      MAX_CODE_UTF8_BYTES
    ),
  };

  if (typeof saved.code === "string" && saved.code.trim() && !(language in savedCodeMap)) {
    codeByLang[language] = safeClipText(saved.code, MAX_CODE_UTF8_BYTES);
  }

  const testsByLang = {
    javascript: safeClipText(
      typeof savedTestsMap.javascript === "string"
        ? savedTestsMap.javascript
        : typeof tests.javascript === "string"
        ? tests.javascript
        : "// Add test assertions here\n",
      MAX_CODE_UTF8_BYTES
    ),
    python: safeClipText(
      typeof savedTestsMap.python === "string"
        ? savedTestsMap.python
        : typeof tests.python === "string"
        ? tests.python
        : "# Add Python assert statements here\n",
      MAX_CODE_UTF8_BYTES
    ),
    pseudocode: "",
  };

  return {
    language,
    codeByLang,
    testsByLang,
    testOutput: safeClipText(
      typeof saved.test_output === "string" ? saved.test_output : "",
      MAX_TEST_OUTPUT_UTF8_BYTES
    ),
  };
}

export function formatTestRunSummary({ language, ok, stdout, error, durationMs, timedOut = false }) {
  const langLabel = language === "python" ? "Python (Pyodide)" : "JavaScript";
  const status = timedOut ? "TIMEOUT" : ok ? "PASSED" : "FAILED";
  const msText = Number.isFinite(durationMs) ? ` (${Math.max(0, Math.round(durationMs))} ms)` : "";
  const parts = [`[${status}] ${langLabel}${msText}`];
  const cleanOut = safeClipText(stdout || "", 2400).trim();
  const cleanErr = safeClipText(error || "", 1200).trim();
  if (cleanOut) parts.push(`STDOUT:\n${cleanOut}`);
  if (cleanErr) parts.push(`ERROR:\n${cleanErr}`);
  return safeClipText(parts.join("\n\n"), MAX_TEST_OUTPUT_UTF8_BYTES);
}

function shortRunId() {
  return `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

/* ---------------- Canonical Coding Runner State Derivation ---------------- */

export function deriveCodingRunnerState({
  language = "javascript",
  activeRun = null,
  matchingResult = null,
  cancelled = false,
  staleResult = false,
  testsPresent = false,
  pyodideInstalled = false,
  pyodideInstalling = false,
  pyodideUninstalling = false,
} = {}) {
  const baseActions = ["switch_language", "edit_code", "reset_starter", "ask_ai_review"];

  if (activeRun) {
    const runLang = activeRun.language || "javascript";
    const runLangLabel = runLang === "python" ? "Python" : "JS";
    if (runLang === language) {
      if (language === "python" && !activeRun.booted) {
        return {
          status: "booting_pyodide",
          pillText: activeRun.phase === "loading" ? "Loading Python…" : "Starting Python…",
          pillTone: "running",
          canRun: false,
          canCancel: true,
          runButtonText: "Running…",
          consoleMeta: "Starting isolated local Pyodide runtime…",
          consoleBodyDefault: "Booting local Python WASM runtime…",
          consoleTone: "neutral",
          allowedActions: ["cancel_run", ...baseActions],
        };
      }
      return {
        status: "running",
        pillText: "Running…",
        pillTone: "running",
        canRun: false,
        canCancel: true,
        runButtonText: "Running…",
        consoleMeta: "Running inside isolated sandbox…",
        consoleBodyDefault: "Executing tests…",
        consoleTone: "neutral",
        allowedActions: ["cancel_run", ...baseActions],
      };
    }
    const otherActions = ["cancel_run", ...baseActions];
    if (language === "python" && !pyodideInstalled && !pyodideInstalling && !pyodideUninstalling) {
      otherActions.splice(1, 0, "install_pyodide");
    }
    return {
      status: "running_other_lang",
      pillText: `Running ${runLangLabel}…`,
      pillTone: "running",
      canRun: false,
      canCancel: true,
      runButtonText: "Running…",
      consoleMeta: `${runLangLabel} test run is still in progress — cancel or wait to run here`,
      consoleBodyDefault: `Waiting for active ${runLangLabel} sandbox run to finish (or click Cancel).`,
      consoleTone: "neutral",
      allowedActions: otherActions,
    };
  }

  if (language === "pseudocode") {
    return {
      status: "pseudocode",
      pillText: "Whiteboard Mode",
      pillTone: "neutral",
      canRun: false,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Whiteboard pseudocode mode (no test runner)",
      consoleBodyDefault: "Talk through your approach and click “Ask AI to review code” for feedback.",
      consoleTone: "neutral",
      allowedActions: baseActions,
    };
  }

  if (language === "python" && pyodideUninstalling) {
    return {
      status: "pyodide_uninstalling",
      pillText: "Removing Python…",
      pillTone: "running",
      canRun: false,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Offline Python runner is being removed…",
      consoleBodyDefault: "Offline Python runner is currently being uninstalled.",
      consoleTone: "neutral",
      allowedActions: baseActions,
    };
  }

  if (language === "python" && pyodideInstalling) {
    return {
      status: "pyodide_installing",
      pillText: "Installing Python…",
      pillTone: "running",
      canRun: false,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Downloading local Pyodide WASM pack…",
      consoleBodyDefault: "Installing offline Python runtime (~14 MB)…",
      consoleTone: "neutral",
      allowedActions: baseActions,
    };
  }

  if (language === "python" && !pyodideInstalled) {
    return {
      status: "pyodide_missing",
      pillText: "Python Not Installed",
      pillTone: "neutral",
      canRun: false,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Install Offline Python above to run tests locally",
      consoleBodyDefault: "Offline Python runner is not installed.",
      consoleTone: "neutral",
      allowedActions: ["install_pyodide", ...baseActions],
    };
  }

  if (!testsPresent) {
    return {
      status: "no_tests",
      pillText: "No Tests",
      pillTone: "neutral",
      canRun: false,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "No tests to run",
      consoleBodyDefault: "Add executable assertions in the Test Suite tab to run tests.",
      consoleTone: "neutral",
      allowedActions: baseActions,
    };
  }

  if (cancelled || matchingResult?.status === "cancelled") {
    return {
      status: "cancelled",
      pillText: "Cancelled",
      pillTone: "neutral",
      canRun: true,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Test run cancelled — press Run Tests to run again",
      consoleBodyDefault: "Test execution was cancelled before completion.",
      consoleTone: "neutral",
      allowedActions: ["run_tests", ...baseActions],
    };
  }

  if (matchingResult?.status === "passed") {
    const msPill = Number.isFinite(matchingResult.durationMs) ? ` · ${matchingResult.durationMs} ms` : "";
    const msMeta = Number.isFinite(matchingResult.durationMs) ? ` in ${matchingResult.durationMs} ms` : "";
    return {
      status: "passed",
      pillText: `Tests Passed${msPill}`,
      pillTone: "pass",
      canRun: true,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: `All assertions passed${msMeta}`,
      consoleBodyDefault: "Completed with no output.",
      consoleTone: "pass",
      allowedActions: ["run_tests", ...baseActions],
    };
  }

  if (matchingResult?.status === "timeout") {
    return {
      status: "timeout",
      pillText: "Timed Out",
      pillTone: "fail",
      canRun: true,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Execution terminated (timeout)",
      consoleBodyDefault: "Execution timed out.",
      consoleTone: "fail",
      allowedActions: ["run_tests", ...baseActions],
    };
  }

  if (matchingResult?.status === "failed") {
    const msMeta = Number.isFinite(matchingResult.durationMs) ? ` after ${matchingResult.durationMs} ms` : "";
    return {
      status: "failed",
      pillText: "Tests Failed",
      pillTone: "fail",
      canRun: true,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: `Test failure${msMeta}`,
      consoleBodyDefault: "Tests failed.",
      consoleTone: "fail",
      allowedActions: ["run_tests", ...baseActions],
    };
  }

  if (matchingResult?.status === "saved") {
    return {
      status: "saved",
      pillText: "Ready",
      pillTone: "neutral",
      canRun: true,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Saved output",
      consoleBodyDefault: "Saved output.",
      consoleTone: "neutral",
      allowedActions: ["run_tests", ...baseActions],
    };
  }

  if (staleResult) {
    return {
      status: "stale",
      pillText: "Modified",
      pillTone: "neutral",
      canRun: true,
      canCancel: false,
      runButtonText: "Run Tests",
      consoleMeta: "Buffer edited since last run — press Run Tests to re-verify",
      consoleBodyDefault: "Previous test output cleared because the code or test suite changed.",
      consoleTone: "neutral",
      allowedActions: ["run_tests", ...baseActions],
    };
  }

  return {
    status: "ready",
    pillText: "Ready",
    pillTone: "neutral",
    canRun: true,
    canCancel: false,
    runButtonText: "Run Tests",
    consoleMeta: "Press Run Tests or ⌘/Ctrl+Enter",
    consoleBodyDefault: "No test runs yet for this language.",
    consoleTone: "neutral",
    allowedActions: ["run_tests", ...baseActions],
  };
}

export function createCodingStudioPlugin({
  escapeHtml,
  showToast,
  clientMachines = null,
  getAddons,
  onAddonsUpdated,
  packOps = null,
}) {
  const sharedPackOps = packOps || createPackOpTracker();
  return function mountCodingStudio({
    containerEl,
    session,
    onWorkspaceChange,
    onAskAiReview,
  }) {
    function getCurrentQuestion(sess = state.session) {
      const qs = sess?.questions || [];
      if (!qs.length) return null;
      const idx = Math.min(sess?.current_question_index || 0, qs.length - 1);
      return qs[idx] || qs[0] || null;
    }

    const initialQ = getCurrentQuestion(session);
    const initialAddons = getAddons?.() || null;
    const initialWs = normalizeCodingWorkspace(initialQ, session?.workspace_context, initialAddons);

    const initialResultsByLang = { javascript: null, python: null };
    if (
      initialWs.testOutput &&
      (initialWs.language === "javascript" || initialWs.language === "python")
    ) {
      initialResultsByLang[initialWs.language] = {
        status: initialWs.testOutput.startsWith("[PASSED]")
          ? "passed"
          : initialWs.testOutput.startsWith("[TIMEOUT]")
          ? "timeout"
          : initialWs.testOutput.startsWith("[FAILED]")
          ? "failed"
          : "saved",
        durationMs: null,
        stdout: initialWs.testOutput,
        error: "",
        codeSnap: initialWs.codeByLang[initialWs.language],
        testsSnap: initialWs.testsByLang[initialWs.language],
        summary: initialWs.testOutput,
      };
    }

    const state = {
      session,
      addons: initialAddons,
      language: initialWs.language,
      activeTab: "solution", // "solution" | "tests"
      codeByLang: initialWs.codeByLang,
      testsByLang: initialWs.testsByLang,
      resultsByLang: initialResultsByLang,
      cancelledByLang: { javascript: null, python: null },
      installingPyodide: false,
      hintsOpen: false,
      revealedHints: 0,
      lastLineCount: 0,
      tabEscaped: false,
      destroyed: false,
    };

    const runnerMachine = createMachine({
      name: "codingStudio",
      initial: "idle",
      transitions: {
        idle: ["booting_pyodide", "running"],
        booting_pyodide: ["running", "idle"],
        running: ["idle"],
      },
      registry: clientMachines,
    });

    // Unified single-use worker handle for the currently running test (JS or Python)
    // { id, worker, timer, booted, phase, language, codeSnap, testsSnap } | null
    let activeRun = null;

    function q(sel) {
      return containerEl.querySelector(sel);
    }

    function notifyChanged() {
      onWorkspaceChange?.();
    }

    function getMatchingResult(lang = state.language) {
      if (lang !== "javascript" && lang !== "python") return null;
      const rec = state.resultsByLang[lang];
      if (!rec) return null;
      const matches =
        rec.codeSnap === state.codeByLang[lang] && rec.testsSnap === state.testsByLang[lang];
      return matches ? rec : null;
    }

    function isCancelledForCurrentBuffer(lang = state.language) {
      if (lang !== "javascript" && lang !== "python") return false;
      const rec = state.cancelledByLang[lang];
      if (!rec) return false;
      return rec.codeSnap === state.codeByLang[lang] && rec.testsSnap === state.testsByLang[lang];
    }

    function hasStaleResult(lang = state.language) {
      if (lang !== "javascript" && lang !== "python") return false;
      const rec = state.resultsByLang[lang];
      if (!rec) return false;
      return rec.codeSnap !== state.codeByLang[lang] || rec.testsSnap !== state.testsByLang[lang];
    }

    function getPyodidePackFlags() {
      const pyPack = (state.addons?.packs || []).find((p) => p && p.id === "pyodide_python");
      const installing = sharedPackOps.isInstalling("pyodide_python") || pyPack?.state === "installing";
      const uninstalling =
        sharedPackOps.isUninstalling("pyodide_python") || pyPack?.state === "uninstalling";
      return {
        installed: isPyodidePackInstalled(state.addons),
        installing,
        uninstalling,
      };
    }

    function getDerivedRunnerState() {
      const pyFlags = getPyodidePackFlags();
      return deriveCodingRunnerState({
        language: state.language,
        activeRun,
        matchingResult: getMatchingResult(state.language),
        cancelled: isCancelledForCurrentBuffer(state.language),
        staleResult: hasStaleResult(state.language),
        testsPresent: hasExecutableTests(state.testsByLang[state.language], state.language),
        pyodideInstalled: pyFlags.installed,
        pyodideInstalling: pyFlags.installing,
        pyodideUninstalling: pyFlags.uninstalling,
      });
    }

    function terminateActiveRun(reason = "terminate") {
      if (!activeRun) return null;
      const run = activeRun;
      activeRun = null;
      if (run.timer) {
        clearTimeout(run.timer);
      }
      if (run.worker) {
        run.worker.onmessage = null;
        run.worker.onerror = null;
        try {
          run.worker.terminate();
        } catch {
          /* ignore */
        }
      }
      clientMachines?.endOp?.("coding_test_run");
      if (runnerMachine.state !== "idle") {
        runnerMachine.transition("idle", { event: reason });
      }
      return run;
    }

    function cancelActiveRun() {
      const run = terminateActiveRun("cancel_run");
      if (!run) return;
      state.cancelledByLang[run.language] = {
        codeSnap: run.codeSnap,
        testsSnap: run.testsSnap,
      };
      if (!state.destroyed) {
        renderStatusPill();
        renderConsole();
        showToast("Cancelled running tests.", 2000);
      }
    }

    function activeEditorText() {
      if (state.activeTab === "tests") {
        return state.testsByLang[state.language] ?? "";
      }
      return state.codeByLang[state.language] ?? "";
    }

    function setActiveEditorText(val) {
      const clipped = safeClipText(val, MAX_CODE_UTF8_BYTES);
      if (state.language === "javascript" || state.language === "python") {
        state.cancelledByLang[state.language] = null;
      }
      if (state.activeTab === "tests") {
        state.testsByLang[state.language] = clipped;
      } else {
        state.codeByLang[state.language] = clipped;
      }
    }

    function buildShell() {
      const langBtns = CODING_LANGUAGES.map(
        (l) =>
          `<button type="button" class="cs-lang-btn" data-cs-lang="${escapeHtml(l.id)}">
             <span>${escapeHtml(l.label)}</span>
           </button>`
      ).join("");

      containerEl.innerHTML = `
        <div class="cs-studio" data-cs-root>
          <div class="cs-header">
            <div class="cs-header-left">
              <span class="practice-kicker">Coding Interview Studio</span>
              <span class="cs-status-pill" data-cs-status role="status" aria-live="polite">Ready</span>
            </div>
            <div class="cs-header-actions">
              <button type="button" class="btn btn-quiet btn-xs" data-cs-hints-toggle>Hints</button>
              <button type="button" class="btn btn-quiet btn-xs" data-cs-reset title="Restore starter template for this language">Reset Starter</button>
              <button type="button" class="btn btn-quiet btn-xs" data-cs-run title="Run unit tests (⌘/Ctrl+Enter)">Run Tests</button>
              <button type="button" class="btn btn-quiet btn-xs cs-cancel-btn hidden" data-cs-cancel title="Cancel active sandbox run">Cancel</button>
              <button type="button" class="btn btn-ink btn-xs" data-cs-review>Ask AI to review code</button>
            </div>
          </div>

          <div class="cs-hints-panel hidden" data-cs-hints></div>

          <div class="cs-toolbar">
            <div class="cs-lang-group" role="group" aria-label="Programming language">
              ${langBtns}
            </div>
            <div class="cs-tab-group" role="tablist" aria-label="Editor view">
              <button type="button" role="tab" class="cs-tab-btn active" data-cs-tab="solution">Solution</button>
              <button type="button" role="tab" class="cs-tab-btn" data-cs-tab="tests">Test Suite</button>
            </div>
          </div>

          <div class="cs-py-banner hidden" data-cs-py-banner></div>

          <div class="cs-editor-wrap">
            <pre class="cs-gutter" data-cs-gutter aria-hidden="true">1</pre>
            <textarea
              class="cs-textarea"
              data-cs-editor
              spellcheck="false"
              autocomplete="off"
              autocapitalize="off"
              aria-label="Code editor (press Escape then Tab to move focus out of the editor)"
            ></textarea>
          </div>
          <div class="cs-editor-hint" data-cs-editor-hint aria-live="polite">Tab indents · Press <strong>Esc</strong> then <strong>Tab</strong> to leave editor · <strong>⌘/Ctrl+Enter</strong> runs tests</div>

          <div class="cs-console" data-cs-console>
            <div class="cs-console-head">
              <span class="practice-kicker">Test Runner Output</span>
              <span class="cs-console-meta" data-cs-console-meta role="status" aria-live="polite">Press Run Tests or ⌘/Ctrl+Enter</span>
            </div>
            <pre class="cs-console-body" data-cs-console-body></pre>
          </div>
        </div>
      `;
    }

    function syncGutter(text) {
      const gutter = q("[data-cs-gutter]");
      if (!gutter) return;
      const lines = Math.max(1, (text.match(/\n/g) || []).length + 1);
      if (lines === state.lastLineCount) return;
      state.lastLineCount = lines;
      const nums = [];
      for (let i = 1; i <= lines; i++) nums.push(i);
      gutter.textContent = nums.join("\n");
    }

    function renderEditorHint() {
      const hint = q("[data-cs-editor-hint]");
      if (!hint) return;
      hint.classList.toggle("is-escaped", state.tabEscaped);
      if (state.tabEscaped) {
        hint.innerHTML =
          "Escape armed — press <strong>Tab</strong> or <strong>Shift+Tab</strong> to move focus out of editor (type any key to resume indenting)";
      } else {
        hint.innerHTML =
          "Tab indents · Press <strong>Esc</strong> then <strong>Tab</strong> to leave editor · <strong>⌘/Ctrl+Enter</strong> runs tests";
      }
    }

    function setTabEscaped(next) {
      const val = Boolean(next);
      if (state.tabEscaped === val) return;
      state.tabEscaped = val;
      renderEditorHint();
    }

    function renderStatusPill() {
      const pill = q("[data-cs-status]");
      const runBtn = q("[data-cs-run]");
      const cancelBtn = q("[data-cs-cancel]");
      const derived = getDerivedRunnerState();

      if (pill) {
        pill.classList.remove("is-pass", "is-fail", "is-running");
        pill.textContent = derived.pillText;
        if (derived.pillTone === "running") pill.classList.add("is-running");
        else if (derived.pillTone === "pass") pill.classList.add("is-pass");
        else if (derived.pillTone === "fail") pill.classList.add("is-fail");
      }

      if (runBtn) {
        runBtn.disabled = !derived.canRun;
        runBtn.textContent = derived.runButtonText;
      }

      if (cancelBtn) {
        cancelBtn.classList.toggle("hidden", !derived.canCancel);
        cancelBtn.disabled = !derived.canCancel;
      }
    }

    function renderHints() {
      const box = q("[data-cs-hints]");
      const btn = q("[data-cs-hints-toggle]");
      if (!box) return;
      const curQ = getCurrentQuestion();
      const hints = Array.isArray(curQ?.follow_ups) ? curQ.follow_ups.filter(Boolean) : [];

      if (btn) {
        btn.textContent = hints.length
          ? `Hints (${Math.min(state.revealedHints, hints.length)}/${hints.length})`
          : "Hints";
      }
      box.classList.toggle("hidden", !state.hintsOpen);
      if (!state.hintsOpen) return;

      if (!hints.length) {
        box.innerHTML = `<div class="cs-hints-empty">Start by stating your invariant & complexity, then test a small edge case by hand.</div>`;
        return;
      }

      const visible = hints.slice(0, state.revealedHints);
      const hasMore = state.revealedHints < hints.length;
      box.innerHTML = `
        <div class="cs-hints-inner">
          <div class="cs-hints-head">
            <span class="practice-kicker">Progressive Coaching Hints</span>
            ${
              hasMore
                ? `<button type="button" class="btn btn-quiet btn-xs" data-cs-reveal-hint>Reveal Hint ${state.revealedHints + 1} of ${hints.length}</button>`
                : `<span class="cs-hints-done">All ${hints.length} hints revealed</span>`
            }
          </div>
          ${
            visible.length
              ? `<ol class="cs-hints-list">${visible.map((h) => `<li>${escapeHtml(h)}</li>`).join("")}</ol>`
              : `<div class="cs-hints-empty">Try sketching your approach first — reveal a hint only when you want a nudge.</div>`
          }
        </div>
      `;
    }

    function renderPyBanner() {
      const banner = q("[data-cs-py-banner]");
      if (!banner) return;
      const { installed, installing, uninstalling } = getPyodidePackFlags();
      const busy = installing || uninstalling;
      const needsPyodide = state.language === "python" && (!installed || busy);
      banner.classList.toggle("hidden", !needsPyodide);
      if (!needsPyodide) {
        banner.innerHTML = "";
        return;
      }
      banner.innerHTML = `
        <div class="cs-py-banner-text">
          <strong>Offline Python runner ${uninstalling ? "is being removed…" : "not installed."}</strong>
          You can write Python and have the AI review it right now, or install the verified local Pyodide WASM pack (~14 MB) to execute Python unit tests offline.
        </div>
        <div class="cs-py-banner-actions">
          <button type="button" class="btn btn-ink btn-xs" data-cs-install-pyodide ${busy ? "disabled" : ""}>
            ${installing ? "Downloading Pyodide…" : uninstalling ? "Removing Pyodide…" : "Install Offline Python (~14 MB)"}
          </button>
          <button type="button" class="btn btn-quiet btn-xs" data-cs-switch-js ${busy ? "disabled" : ""}>
            Switch to JavaScript
          </button>
        </div>
      `;
    }

    function renderToolbarAndEditor({ preserveCursor = false } = {}) {
      containerEl.querySelectorAll("[data-cs-lang]").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-cs-lang") === state.language);
      });
      const testsTab = q('[data-cs-tab="tests"]');
      if (testsTab) {
        const isPseudo = state.language === "pseudocode";
        testsTab.classList.toggle("hidden", isPseudo);
        if (isPseudo && state.activeTab === "tests") {
          state.activeTab = "solution";
        }
      }
      containerEl.querySelectorAll("[data-cs-tab]").forEach((btn) => {
        const isActive = btn.getAttribute("data-cs-tab") === state.activeTab;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
      });

      const editor = q("[data-cs-editor]");
      if (editor) {
        const nextText = activeEditorText();
        if (!preserveCursor || editor.value !== nextText) {
          editor.value = nextText;
        }
        editor.placeholder =
          state.activeTab === "tests"
            ? "Edit or add unit test assertions here…"
            : state.language === "pseudocode"
            ? "Outline your algorithm, data structures, edge cases, and Big-O complexity…"
            : "Write your solution here…";
        syncGutter(editor.value);
      }
    }

    function renderConsole() {
      const meta = q("[data-cs-console-meta]");
      const body = q("[data-cs-console-body]");
      if (!body || !meta) return;

      body.classList.remove("is-pass", "is-fail");
      const derived = getDerivedRunnerState();
      meta.textContent = derived.consoleMeta;

      if (derived.consoleTone === "pass") body.classList.add("is-pass");
      else if (derived.consoleTone === "fail") body.classList.add("is-fail");

      if (
        derived.status === "passed" ||
        derived.status === "failed" ||
        derived.status === "timeout" ||
        derived.status === "saved"
      ) {
        const currentRes = getMatchingResult(state.language);
        const parts = [];
        if (currentRes?.stdout) parts.push(currentRes.stdout);
        if (currentRes?.error) parts.push(currentRes.error);
        body.textContent =
          parts.join("\n\n").trim() || currentRes?.summary || derived.consoleBodyDefault;
        return;
      }

      body.textContent = derived.consoleBodyDefault;
    }

    function renderAll(opts = {}) {
      renderStatusPill();
      renderHints();
      renderPyBanner();
      renderToolbarAndEditor(opts);
      renderEditorHint();
      renderConsole();
    }

    /* ---------------- Sandboxed Test Execution (Single-Use Isolated Workers) ---------------- */

    function finishRun(runRef, { ok, stdout, error, durationMs, timedOut = false }) {
      if (state.destroyed || activeRun !== runRef) return;
      terminateActiveRun(timedOut ? "timeout" : ok ? "passed" : "failed");

      const cleanStdout = safeClipText(stdout || "", 3000);
      const cleanError = safeClipText(error || "", 1800);
      const summary = formatTestRunSummary({
        language: runRef.language,
        ok,
        stdout: cleanStdout,
        error: cleanError,
        durationMs,
        timedOut,
      });

      state.cancelledByLang[runRef.language] = null;
      state.resultsByLang[runRef.language] = {
        status: timedOut ? "timeout" : ok ? "passed" : "failed",
        durationMs: Number.isFinite(durationMs) ? Math.max(0, Math.round(durationMs)) : null,
        stdout: cleanStdout,
        error: cleanError,
        codeSnap: runRef.codeSnap,
        testsSnap: runRef.testsSnap,
        summary,
      };

      renderStatusPill();
      renderConsole();
      notifyChanged();
    }

    function launchIsolatedWorker(language, code, tests) {
      terminateActiveRun("restart");
      state.cancelledByLang[language] = null;

      const runId = shortRunId();
      const workerUrl =
        language === "python" ? "/static/py_sandbox_worker.js" : "/static/js_sandbox_worker.js";
      let worker;
      try {
        worker = new Worker(workerUrl);
      } catch (err) {
        const fallbackRef = {
          id: runId,
          worker: null,
          timer: null,
          booted: true,
          phase: "running",
          language,
          codeSnap: code,
          testsSnap: tests,
        };
        activeRun = fallbackRef;
        finishRun(fallbackRef, {
          ok: false,
          stdout: "",
          error: `Could not launch sandbox worker: ${err.message || err}`,
          durationMs: 0,
        });
        return;
      }

      const runRef = {
        id: runId,
        worker,
        timer: null,
        booted: language === "javascript",
        phase: language === "python" ? "starting" : "running",
        language,
        codeSnap: code,
        testsSnap: tests,
      };
      activeRun = runRef;
      clientMachines?.beginOp?.("coding_test_run", { language, runId });
      runnerMachine.transition(language === "python" ? "booting_pyodide" : "running", {
        event: "launch_worker",
        reason: language,
      });

      function armTimer(ms, timeoutMsg) {
        if (runRef.timer) clearTimeout(runRef.timer);
        runRef.timer = setTimeout(() => {
          finishRun(runRef, {
            ok: false,
            stdout: "",
            error: timeoutMsg,
            durationMs: ms,
            timedOut: true,
          });
        }, ms);
      }

      if (language === "javascript") {
        armTimer(
          JS_TIMEOUT_MS,
          `Execution timed out after ${(JS_TIMEOUT_MS / 1000).toFixed(1)}s (possible infinite loop or unresolved promise).`
        );
      } else {
        armTimer(
          PY_BOOT_TIMEOUT_MS,
          "Local Pyodide WASM runtime took too long to boot. Try reinstalling Offline Python in Settings → Add-Ons."
        );
      }

      worker.onmessage = (ev) => {
        if (state.destroyed || activeRun !== runRef) return;
        const msg = ev.data || {};
        if (msg.runId !== runId) return;

        if (language === "python" && msg.type === "status") {
          if (!runRef.booted && typeof msg.status === "string" && msg.status.trim()) {
            runRef.phase = msg.status.trim();
            renderStatusPill();
            renderConsole();
          }
          return;
        }

        if (language === "python" && msg.type === "ready") {
          // One-time transition from boot timeout to strict execution timeout
          if (runRef.booted) return;
          runRef.booted = true;
          runRef.phase = "running";
          runnerMachine.transition("running", { event: "pyodide_ready" });
          renderStatusPill();
          renderConsole();
          armTimer(
            PY_RUN_TIMEOUT_MS,
            `Python execution timed out after ${(PY_RUN_TIMEOUT_MS / 1000).toFixed(1)}s (possible infinite loop).`
          );
          return;
        }

        if (msg.type === "result") {
          finishRun(runRef, {
            ok: Boolean(msg.ok),
            stdout: msg.stdout || "",
            error: msg.error || "",
            durationMs: msg.duration_ms,
          });
        }
      };

      worker.onerror = (err) => {
        if (state.destroyed || activeRun !== runRef) return;
        finishRun(runRef, {
          ok: false,
          stdout: "",
          error: err.message || "Sandbox worker runtime error.",
          durationMs: 0,
        });
      };

      renderStatusPill();
      renderConsole();
      worker.postMessage({ type: "run", runId, code, tests });
    }

    function runTests() {
      if (activeRun) return;
      if (state.language === "pseudocode") {
        showToast("Switch to JavaScript or Python to execute unit tests.", 2400);
        return;
      }
      if (state.language === "python") {
        const pyFlags = getPyodidePackFlags();
        if (pyFlags.installing || pyFlags.uninstalling) {
          showToast(
            pyFlags.uninstalling
              ? "Offline Python runner is currently being removed."
              : "Offline Python runner is still installing.",
            2400
          );
          return;
        }
        if (!pyFlags.installed) {
          showToast("Install Offline Python (~14 MB) above to run Python tests locally.", 2800);
          return;
        }
      }
      const code = state.codeByLang[state.language] || "";
      const tests = state.testsByLang[state.language] || "";
      if (!hasExecutableTests(tests, state.language)) {
        showToast("No tests to run — add assertions in the Test Suite tab first.", 2600);
        return;
      }
      launchIsolatedWorker(state.language, code, tests);
    }

    async function installPyodideInline() {
      if (!sharedPackOps.begin("pyodide_python", "installing")) return;
      state.installingPyodide = true;
      renderPyBanner();
      renderStatusPill();
      renderConsole();
      let refreshedFromResponse = false;
      try {
        const res = await fetch("/api/addons/pyodide_python/install", { method: "POST" });
        const data = await res.json().catch(() => ({}));
        if (data?.addons) {
          refreshedFromResponse = true;
          onAddonsUpdated?.(data.addons);
          if (!state.destroyed) {
            state.addons = data.addons;
          }
        }
        if (!res.ok) {
          throw new Error(formatApiDetail(data.detail, "Pyodide installation failed."));
        }
        if (state.destroyed) return;
        if (isPyodidePackInstalled(data?.addons || state.addons)) {
          showToast(
            "Offline Python runner (Pyodide) installed! You can now run Python tests offline.",
            3200
          );
        } else {
          showToast("Offline Python installation is still finalizing — please check Add-Ons.", 3000);
        }
      } catch (err) {
        if (!state.destroyed) {
          showToast(String(err.message || err), 3600);
        }
      } finally {
        sharedPackOps.end("pyodide_python");
        state.installingPyodide = false;
        if (!refreshedFromResponse) {
          try {
            const freshRes = await fetch("/api/addons");
            if (freshRes.ok) {
              const freshAddons = await freshRes.json();
              onAddonsUpdated?.(freshAddons);
              if (!state.destroyed) state.addons = freshAddons;
            }
          } catch {
            /* ignore */
          }
        }
        if (!state.destroyed) {
          renderAll({ preserveCursor: true });
        }
      }
    }

    function resetStarterForActiveTab() {
      const curQ = getCurrentQuestion();
      if (state.language === "javascript" || state.language === "python") {
        state.cancelledByLang[state.language] = null;
      }
      if (state.activeTab === "tests") {
        const curated = curQ?.tests?.[state.language];
        state.testsByLang[state.language] = safeClipText(
          typeof curated === "string" ? curated : "",
          MAX_CODE_UTF8_BYTES
        );
        showToast("Restored curated test suite.", 2000);
      } else {
        const curated = curQ?.starter_code?.[state.language];
        state.codeByLang[state.language] = safeClipText(
          typeof curated === "string"
            ? curated
            : state.language === "python"
            ? "# Write your Python solution here\n"
            : "// Write your solution here\n",
          MAX_CODE_UTF8_BYTES
        );
        showToast("Restored starter code.", 2000);
      }
      renderAll();
      notifyChanged();
    }

    /* ---------------- Event Handlers ---------------- */

    function onClick(ev) {
      const langBtn = ev.target.closest("[data-cs-lang]");
      if (langBtn) {
        const nextLang = langBtn.getAttribute("data-cs-lang");
        if (VALID_LANG_IDS.has(nextLang) && nextLang !== state.language) {
          state.language = nextLang;
          if (nextLang === "pseudocode") state.activeTab = "solution";
          renderAll();
          notifyChanged();
        }
        return;
      }

      const tabBtn = ev.target.closest("[data-cs-tab]");
      if (tabBtn) {
        const nextTab = tabBtn.getAttribute("data-cs-tab");
        if ((nextTab === "solution" || nextTab === "tests") && nextTab !== state.activeTab) {
          state.activeTab = nextTab;
          renderToolbarAndEditor();
        }
        return;
      }

      if (ev.target.closest("[data-cs-hints-toggle]")) {
        state.hintsOpen = !state.hintsOpen;
        renderHints();
        return;
      }

      if (ev.target.closest("[data-cs-reveal-hint]")) {
        const hints = Array.isArray(getCurrentQuestion()?.follow_ups)
          ? getCurrentQuestion().follow_ups
          : [];
        if (state.revealedHints < hints.length) {
          state.revealedHints += 1;
          renderHints();
        }
        return;
      }

      if (ev.target.closest("[data-cs-reset]")) {
        resetStarterForActiveTab();
        return;
      }

      if (ev.target.closest("[data-cs-run]")) {
        runTests();
        return;
      }

      if (ev.target.closest("[data-cs-cancel]")) {
        cancelActiveRun();
        return;
      }

      if (ev.target.closest("[data-cs-install-pyodide]")) {
        installPyodideInline();
        return;
      }

      if (ev.target.closest("[data-cs-switch-js]")) {
        state.language = "javascript";
        renderAll();
        notifyChanged();
        return;
      }

      if (ev.target.closest("[data-cs-review]")) {
        const code = (state.codeByLang[state.language] || "").trim();
        if (!code) {
          showToast("Write a bit of code or pseudocode first before asking for a review.", 2400);
          return;
        }
        notifyChanged();
        const curRes = getMatchingResult(state.language);
        const statusNote =
          curRes?.status === "passed"
            ? "My tests are currently passing — please review my solution for edge cases, time/space complexity, and code clarity."
            : curRes?.status === "failed" || curRes?.status === "timeout"
            ? "I hit a test failure — please give me a gentle, non-spoiler hint on what might be going wrong in my code."
            : "Please review my current code approach — highlight what looks solid and probe one edge case or complexity trade-off.";
        onAskAiReview?.(statusNote);
      }
    }

    function onInput(ev) {
      if (!ev.target.matches("[data-cs-editor]")) return;
      setTabEscaped(false);
      setActiveEditorText(ev.target.value);
      syncGutter(ev.target.value);
      renderStatusPill();
      renderConsole();
      notifyChanged();
    }

    function onScroll(ev) {
      if (!ev.target.matches("[data-cs-editor]")) return;
      const gutter = q("[data-cs-gutter]");
      if (gutter) gutter.scrollTop = ev.target.scrollTop;
    }

    function onFocusOut(ev) {
      if (ev.target?.matches?.("[data-cs-editor]")) {
        setTabEscaped(false);
      }
    }

    function onKeyDown(ev) {
      if (!ev.target?.matches?.("[data-cs-editor]")) return;
      const ta = ev.target;

      // WCAG 2.1.2 keyboard-trap escape: press Escape, then Tab/Shift+Tab navigates normally
      if (ev.key === "Escape") {
        setTabEscaped(true);
        return;
      }

      // Preserve armed escape across modifier keys so Esc then Shift+Tab works cleanly
      if (
        ev.key === "Shift" ||
        ev.key === "Control" ||
        ev.key === "Alt" ||
        ev.key === "Meta"
      ) {
        return;
      }

      // Cmd/Ctrl+Enter inside the editor runs the unit test suite
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        setTabEscaped(false);
        runTests();
        return;
      }

      const langMeta = CODING_LANGUAGES.find((l) => l.id === state.language) || CODING_LANGUAGES[0];
      const indent = langMeta.indent;

      if (ev.key === "Tab") {
        if (state.tabEscaped) {
          // Let browser perform standard keyboard focus traversal out of the textarea
          setTabEscaped(false);
          return;
        }
        ev.preventDefault();
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const val = ta.value;

        if (!ev.shiftKey && start === end) {
          ta.value = val.slice(0, start) + indent + val.slice(end);
          ta.selectionStart = ta.selectionEnd = start + indent.length;
        } else {
          const lineStart = val.lastIndexOf("\n", Math.max(0, start - 1)) + 1;
          const block = val.slice(lineStart, end);
          const lines = block.split("\n");
          let deltaStart = 0;
          let deltaEnd = 0;
          const updated = lines.map((line, idx) => {
            if (!ev.shiftKey) {
              if (idx === 0) deltaStart = indent.length;
              deltaEnd += indent.length;
              return indent + line;
            }
            let remove = 0;
            if (line.startsWith(indent)) remove = indent.length;
            else if (line.startsWith("\t")) remove = 1;
            else {
              const spaces = line.match(/^ +/)?.[0]?.length || 0;
              remove = Math.min(spaces, indent.length);
            }
            if (idx === 0) deltaStart = -Math.min(remove, start - lineStart);
            deltaEnd -= remove;
            return line.slice(remove);
          });
          ta.value = val.slice(0, lineStart) + updated.join("\n") + val.slice(end);
          ta.selectionStart = Math.max(lineStart, start + deltaStart);
          ta.selectionEnd = Math.max(ta.selectionStart, end + deltaEnd);
        }
        setActiveEditorText(ta.value);
        syncGutter(ta.value);
        renderStatusPill();
        renderConsole();
        notifyChanged();
        return;
      }

      setTabEscaped(false);

      if (ev.key === "Enter") {
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const val = ta.value;
        const lineStart = val.lastIndexOf("\n", Math.max(0, start - 1)) + 1;
        const curLine = val.slice(lineStart, start);
        const leading = curLine.match(/^[ \t]*/)?.[0] || "";
        const extra =
          (state.language === "python" && /:\s*$/.test(curLine)) ||
          (state.language !== "python" && /\{\s*$/.test(curLine))
            ? indent
            : "";
        if (leading || extra) {
          ev.preventDefault();
          const insert = `\n${leading}${extra}`;
          ta.value = val.slice(0, start) + insert + val.slice(end);
          ta.selectionStart = ta.selectionEnd = start + insert.length;
          setActiveEditorText(ta.value);
          syncGutter(ta.value);
          renderStatusPill();
          renderConsole();
          notifyChanged();
        }
      }
    }

    buildShell();
    renderAll();

    const editorEl = q("[data-cs-editor]");
    containerEl.addEventListener("click", onClick);
    containerEl.addEventListener("input", onInput);
    containerEl.addEventListener("keydown", onKeyDown);
    containerEl.addEventListener("focusout", onFocusOut);
    editorEl?.addEventListener("scroll", onScroll);
    const unsubscribePackOps = sharedPackOps.subscribe((packId) => {
      if (state.destroyed || packId !== "pyodide_python") return;
      renderPyBanner();
      renderStatusPill();
      renderConsole();
    });

    return {
      getWorkspaceContext() {
        const currentRes = getMatchingResult(state.language);
        return {
          language: state.language,
          code: safeClipText(state.codeByLang[state.language] || "", MAX_CODE_UTF8_BYTES),
          test_output: safeClipText(currentRes?.summary || "", MAX_TEST_OUTPUT_UTF8_BYTES),
          code_by_lang: {
            javascript: safeClipText(state.codeByLang.javascript || "", MAX_CODE_UTF8_BYTES),
            python: safeClipText(state.codeByLang.python || "", MAX_CODE_UTF8_BYTES),
            pseudocode: safeClipText(state.codeByLang.pseudocode || "", MAX_CODE_UTF8_BYTES),
          },
          tests_by_lang: {
            javascript: safeClipText(state.testsByLang.javascript || "", MAX_CODE_UTF8_BYTES),
            python: safeClipText(state.testsByLang.python || "", MAX_CODE_UTF8_BYTES),
          },
        };
      },
      getStateSnapshot() {
        const derived = getDerivedRunnerState();
        return {
          studio: "coding",
          language: state.language,
          activeTab: state.activeTab,
          runnerState: runnerMachine.state,
          derivedStatus: derived.status,
          activeRunLanguage: activeRun?.language || null,
          activeRunPhase: activeRun?.phase || null,
          tabEscaped: state.tabEscaped,
          hintsOpen: state.hintsOpen,
          revealedHints: state.revealedHints,
          pyodideInstalled: isPyodidePackInstalled(state.addons),
          canRun: derived.canRun,
          canCancel: derived.canCancel,
          allowedActions: derived.allowedActions,
        };
      },
      updateSession(nextSession) {
        state.session = nextSession;
        renderHints();
      },
      updateAddons(nextAddons) {
        const wasInstalled = isPyodidePackInstalled(state.addons);
        state.addons = nextAddons;
        const nowInstalled = isPyodidePackInstalled(nextAddons);
        if (wasInstalled && !nowInstalled && activeRun?.language === "python") {
          terminateActiveRun("pyodide_uninstalled");
        }
        renderStatusPill();
        renderPyBanner();
        renderConsole();
      },
      destroy() {
        state.destroyed = true;
        unsubscribePackOps?.();
        terminateActiveRun("destroy");
        clientMachines?.unregister?.("codingStudio");
        containerEl.removeEventListener("click", onClick);
        containerEl.removeEventListener("input", onInput);
        containerEl.removeEventListener("keydown", onKeyDown);
        containerEl.removeEventListener("focusout", onFocusOut);
        editorEl?.removeEventListener("scroll", onScroll);
      },
    };
  };
}
