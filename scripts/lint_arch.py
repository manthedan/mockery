#!/usr/bin/env python3
"""Actionable architecture, boundary, DOM-ID, and state-chart verifier for Mockery.

Run via:
  uv run python scripts/lint_arch.py
  uv run python scripts/lint_arch.py --write-charts
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src" / "mockery"
STATIC_DIR = SRC_DIR / "static"
CHARTS_DOC = ROOT / "docs" / "state-charts.md"

FORBIDDEN_RUNTIME_IMPORTS = frozenset({"evals", "gepa", "litellm"})
FORBIDDEN_PROMPT_IMPORTS = frozenset(
    {
        "fastapi",
        "httpx",
        "openai",
        "uvicorn",
        "websockets",
        "mockery",
    }
)
HTTP_ROUTE_METHODS = frozenset({"get", "post", "put", "patch", "delete"})


def _iter_imported_roots(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append((node.lineno, "mockery"))
            elif node.module:
                found.append((node.lineno, node.module.split(".")[0]))
    return found


def check_runtime_isolation() -> list[str]:
    """ARCH001 & ARCH002: Enforce runtime/eval isolation and prompt purity."""
    errors: list[str] = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        rel = py_file.relative_to(ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(rel))
        for lineno, root_mod in _iter_imported_roots(tree):
            if root_mod in FORBIDDEN_RUNTIME_IMPORTS:
                errors.append(
                    f"[ARCH001] {rel}:{lineno}: Forbidden runtime import '{root_mod}'.\n"
                    f"  Why: Production code under src/mockery/ must not depend on optional offline eval/GEPA packages.\n"
                    f"  Fix: Move eval or optimization logic into evals/ and keep src/mockery/ zero-dependency on {root_mod}."
                )
            if py_file.name == "prompts.py" and root_mod in FORBIDDEN_PROMPT_IMPORTS:
                errors.append(
                    f"[ARCH002] {rel}:{lineno}: Forbidden import '{root_mod}' in prompts.py.\n"
                    f"  Why: prompts.py is the shared pure-Python single source of truth imported by both runtime and offline evals.\n"
                    f"  Fix: Keep prompts.py stdlib-only and pass runtime context as plain template arguments."
                )
    return errors


def _is_http_route_decorator(dec: ast.expr) -> bool:
    if not isinstance(dec, ast.Call):
        return False
    func = dec.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in ("app", "router")
        and func.attr in HTTP_ROUTE_METHODS
    )


def _is_runtime_llm_call(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Await):
        return None
    call = node.value
    if not isinstance(call, ast.Call):
        return None
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "runtime"
        and func.value.attr == "llm"
    ):
        return func.attr
    return None


def check_wrapped_llm_routes() -> list[str]:
    """ARCH003: Ensure HTTP route calls to `await runtime.llm.*` sit inside try/except."""
    route_files = [SRC_DIR / "server.py"]
    routers_dir = SRC_DIR / "routers"
    if routers_dir.is_dir():
        route_files.extend(sorted(routers_dir.glob("*.py")))

    errors: list[str] = []

    def walk_body(node: ast.AST, in_try: bool, route_name: str, rel: Path) -> None:
        if isinstance(node, ast.Try):
            has_except = bool(node.handlers)
            for stmt in node.body:
                walk_body(stmt, in_try or has_except, route_name, rel)
            for handler in node.handlers:
                walk_body(handler, in_try, route_name, rel)
            for stmt in node.orelse:
                walk_body(stmt, in_try, route_name, rel)
            for stmt in node.finalbody:
                walk_body(stmt, in_try, route_name, rel)
            return

        llm_method = _is_runtime_llm_call(node) if isinstance(node, ast.expr) else None
        if llm_method and not in_try:
            errors.append(
                f"[ARCH003] {rel}:{node.lineno}: Unwrapped `await runtime.llm.{llm_method}(...)` in route `{route_name}`.\n"
                f"  Why: Uncaught LLM/provider exceptions escape FastAPI as plain-text 500s and break client JSON parsing.\n"
                f"  Fix: Wrap `await runtime.llm.{llm_method}(...)` in `try ... except Exception as exc:` and raise "
                f"`HTTPException(status_code=503, detail=...)`."
            )
            return

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            walk_body(child, in_try, route_name, rel)

    for py_file in route_files:
        rel = py_file.relative_to(ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(_is_http_route_decorator(dec) for dec in node.decorator_list):
                    for stmt in node.body:
                        walk_body(stmt, False, node.name, rel)
    return errors


def check_router_slice_isolation() -> list[str]:
    """ARCH004: Domain routers under src/mockery/routers/ must stay isolated from siblings and server.py."""
    routers_dir = SRC_DIR / "routers"
    if not routers_dir.is_dir():
        return []
    errors: list[str] = []
    for py_file in sorted(routers_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(ROOT)
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 1 and mod:
                    errors.append(
                        f"[ARCH004] {rel}:{node.lineno}: Router imports sibling router `.{mod}`.\n"
                        f"  Why: Domain routers must be independent vertical slices to prevent circular coupling.\n"
                        f"  Fix: Share domain logic via src/mockery/<domain>.py or pass helpers through runtime."
                    )
                elif mod == "server" or mod.startswith("mockery.server") or mod.startswith("mockery.routers"):
                    errors.append(
                        f"[ARCH004] {rel}:{node.lineno}: Router imports `{mod}`.\n"
                        f"  Why: Routers must not depend back on the server composition root or sibling routers.\n"
                        f"  Fix: Inject needed runtime callables into `build_*_router(runtime, ...)`."
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mockery.server") or alias.name.startswith("mockery.routers"):
                        errors.append(
                            f"[ARCH004] {rel}:{node.lineno}: Router imports `{alias.name}`.\n"
                            f"  Why: Routers must remain independent vertical slices.\n"
                            f"  Fix: Inject needed dependencies into `build_*_router(runtime, ...)`."
                        )
    return errors


_HTML_ID_RE = re.compile(r"""\bid\s*=\s*["']([A-Za-z_][A-Za-z0-9_-]*)["']""")
_JS_ASSIGN_ID_RE = re.compile(r"""\.id\s*=\s*["']([A-Za-z_][A-Za-z0-9_-]*)["']""")
_JS_GET_BY_ID_RE = re.compile(r"""(?:document\.getElementById|\$)\(\s*["']([A-Za-z_][A-Za-z0-9_-]*)["']\s*\)""")


def check_dom_id_integrity() -> list[str]:
    """DOM001: Verify literal getElementById("...") calls match real IDs in index.html or JS templates."""
    index_html = STATIC_DIR / "index.html"
    known_ids: set[str] = set(_HTML_ID_RE.findall(index_html.read_text(encoding="utf-8")))
    js_files = sorted(STATIC_DIR.glob("*.js"))
    js_texts: dict[Path, str] = {}

    for js_file in js_files:
        text = js_file.read_text(encoding="utf-8")
        js_texts[js_file] = text
        known_ids.update(_HTML_ID_RE.findall(text))
        known_ids.update(_JS_ASSIGN_ID_RE.findall(text))

    errors: list[str] = []
    for js_file, text in js_texts.items():
        rel = js_file.relative_to(ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for match in _JS_GET_BY_ID_RE.finditer(line):
                dom_id = match.group(1)
                if dom_id not in known_ids:
                    errors.append(
                        f"[DOM001] {rel}:{lineno}: Unknown DOM id '{dom_id}' referenced via getElementById/$().\n"
                        f"  Why: Referencing a missing or renamed DOM id silently returns null and breaks UI flows.\n"
                        f"  Fix: Add id=\"{dom_id}\" in src/mockery/static/index.html (or its template) or remove the stale reference."
                    )
    return errors


def render_state_charts_markdown() -> str:
    """Generate canonical Mermaid state charts directly from Python + JS state machines."""
    sys.path.insert(0, str(ROOT / "src"))
    from mockery.state_machines import export_machine_specs

    specs = export_machine_specs()["machines"]

    # Introspect JS client FSMs via Node so frontend charts are also code-generated.
    node_probe = r"""
import * as sm from "./src/mockery/static/state_machines.js";
const factories = [
  ["practice_mic", () => sm.createPracticeMicMachine()],
  ["practice_op", () => sm.createPracticeOpMachine()],
  ["practice_retry_timer", () => sm.createRetryTimerMachine()],
  ["practice_workspace_save", () => sm.createWorkspaceSaveMachine()],
];
const out = {};
for (const [name, make] of factories) {
  const m = make();
  const initial = m.state;
  const visited = new Set([initial]);
  const queue = [initial];
  const edges = {};
  while (queue.length > 0) {
    const s = queue.shift();
    m.reset(s, { event: "probe" });
    const nexts = m.allowedNext().slice().sort();
    edges[s] = nexts;
    for (const n of nexts) {
      if (!visited.has(n)) {
        visited.add(n);
        queue.push(n);
      }
    }
  }
  out[name] = { initial, states: Array.from(visited).sort(), transitions: edges };
}
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    js_specs = json.loads(proc.stdout)

    lines: list[str] = [
        "# Mockery State Charts (Auto-Generated)",
        "",
        "Generated directly from `src/mockery/state_machines.py` and `src/mockery/static/state_machines.js`.",
        "Do not edit manually — regenerate with `uv run python scripts/lint_arch.py --write-charts`.",
        "",
        "## Backend State Machines (`src/mockery/state_machines.py`)",
        "",
    ]

    for name, spec in specs.items():
        transitions = spec.get("transitions")
        mermaid = spec.get("mermaid")
        if not transitions and not mermaid:
            continue
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```mermaid")
        if mermaid:
            lines.append(mermaid.rstrip())
        else:
            lines.append("stateDiagram-v2")
            states = spec.get("states") or []
            if states:
                lines.append(f"  [*] --> {states[0]}")
            for src_state, targets in transitions.items():
                for dst_state in targets:
                    lines.append(f"  {src_state} --> {dst_state}")
        lines.append("```")
        lines.append("")

    lines.append("## Frontend Client State Machines (`src/mockery/static/state_machines.js`)")
    lines.append("")
    for name, spec in js_specs.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```mermaid")
        lines.append("stateDiagram-v2")
        lines.append(f"  [*] --> {spec['initial']}")
        for src_state in sorted(spec["transitions"]):
            for dst_state in spec["transitions"][src_state]:
                lines.append(f"  {src_state} --> {dst_state}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def check_state_charts_freshness(write: bool = False) -> list[str]:
    """FSM001: Keep docs/state-charts.md strictly in sync with code."""
    expected = render_state_charts_markdown()
    if write:
        CHARTS_DOC.parent.mkdir(parents=True, exist_ok=True)
        CHARTS_DOC.write_text(expected, encoding="utf-8")
        return []
    if not CHARTS_DOC.exists():
        return [
            f"[FSM001] Missing {CHARTS_DOC.relative_to(ROOT)}.\n"
            f"  Why: Visual state charts are a checked-in verification artifact derived from the FSM code.\n"
            f"  Fix: Run `uv run python scripts/lint_arch.py --write-charts`."
        ]
    actual = CHARTS_DOC.read_text(encoding="utf-8")
    if actual != expected:
        return [
            f"[FSM001] {CHARTS_DOC.relative_to(ROOT)} is out of sync with state_machines.py / state_machines.js.\n"
            f"  Why: State charts must never drift from the live transition tables.\n"
            f"  Fix: Run `uv run python scripts/lint_arch.py --write-charts` and commit the updated diagram."
        ]
    return []


_INNERHTML_TPL_RE = re.compile(r"""\.innerHTML\s*=\s*`([^`]*)`""")
_TPL_EXPR_RE = re.compile(r"""\$\{([^}]+)\}""")
_SAFE_HTML_HELPERS = (
    "escapeHtml(",
    "formatWithNewlines(",
    "highlightCode(",
    "dimFillersHtml(",
    "renderMarkdownSafe(",
    "render",
)
_SAFE_KNOWN_IDENTIFIERS = frozenset(
    {
        "you",
        "aiTone",
        "title",
        "body",
        "action",
        "langBtns",
        "paletteButtons",
        "dragDisabled",
        "typeOpts",
        "gridId",
        "markerId",
        "markerSelId",
        "rowsHtml",
        "traceHtml",
    }
)


def _all_ternary_leaves_are_literals(expr: str) -> bool:
    s = expr.strip()
    if re.fullmatch(r"""(?:"[^"]*"|'[^']*')""", s):
        return True
    if "?" not in s or ":" not in s:
        return False
    after_q = s.split("?", 1)[1]
    left, right = after_q.split(":", 1)
    return _all_ternary_leaves_are_literals(left) and _all_ternary_leaves_are_literals(right)


def _is_safe_innerhtml_expr(expr: str) -> bool:
    e = expr.strip()
    if any(e.startswith(prefix) for prefix in _SAFE_HTML_HELPERS):
        return True
    if e in _SAFE_KNOWN_IDENTIFIERS:
        return True
    if e.endswith("Html") or e.endswith("HTML"):
        return True
    # Pure numeric / constant expressions (e.g. CANVAS_W - 1, -CANVAS_BLEED)
    if re.fullmatch(r"[\s0-9A-Z_+\-*/()]+", e):
        return True
    # Single or chained ternary whose leaf branches are all string literals
    if _all_ternary_leaves_are_literals(e):
        return True
    return False


def check_innerhtml_xss_safety() -> list[str]:
    r"""DOM002: Flag raw unescaped dynamic interpolations inside `.innerHTML = `...`` templates."""
    errors: list[str] = []
    for js_file in sorted(STATIC_DIR.glob("*.js")):
        rel = js_file.relative_to(ROOT)
        text = js_file.read_text(encoding="utf-8")
        for match in _INNERHTML_TPL_RE.finditer(text):
            lineno = text[: match.start()].count("\n") + 1
            for expr_match in _TPL_EXPR_RE.finditer(match.group(1)):
                expr = expr_match.group(1).strip()
                if not _is_safe_innerhtml_expr(expr):
                    errors.append(
                        f"[DOM002] {rel}:{lineno}: Potentially unescaped interpolation `${{{expr}}}` inside `.innerHTML = ...`.\n"
                        f"  Why: Interpolating dynamic text directly into `.innerHTML` creates an XSS vector.\n"
                        f"  Fix: Wrap dynamic strings with `escapeHtml(...)` or assign via `.textContent` instead."
                    )
    return errors


_STALE_TAG_RE = re.compile(r"\bMilestone\s+\d|scratch/triage-lab|#\s*noqa:\s*F401")


def check_comment_hygiene() -> list[str]:
    """HYG001: Block stale build-plan/prototype tags and `# noqa: F401` runtime probes in src/mockery/."""
    errors: list[str] = []
    exts = {".py", ".js", ".css", ".html"}
    for path in sorted(SRC_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in exts:
            continue
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            m = _STALE_TAG_RE.search(line)
            if m:
                errors.append(
                    f"[HYG001] {rel}:{lineno}: Disallowed stale tag or probe `{m.group(0)}`.\n"
                    f"  Why: Build-plan milestones, deleted scratch references, and `# noqa: F401` import probes rot quickly.\n"
                    f"  Fix: Use plain domain names in comments and `importlib.util.find_spec(...)` for package probes."
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Mockery architecture & boundary linter")
    parser.add_argument(
        "--write-charts",
        action="store_true",
        help="Regenerate docs/state-charts.md from live Python and JS state machines.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    errors.extend(check_runtime_isolation())
    errors.extend(check_wrapped_llm_routes())
    errors.extend(check_router_slice_isolation())
    errors.extend(check_dom_id_integrity())
    errors.extend(check_innerhtml_xss_safety())
    errors.extend(check_comment_hygiene())
    errors.extend(check_state_charts_freshness(write=args.write_charts))

    if errors:
        print(f"Architecture lint failed with {len(errors)} issue(s):\n", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
            print("", file=sys.stderr)
        return 1

    print("Architecture lint passed: boundaries, routers, LLM guards, DOM IDs/XSS, and state charts are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
