#!/usr/bin/env python3
"""Single-command Inner Loop verifier for Mockery.

Usage:
  uv run python scripts/check.py          # Fast static + unit + FSM + fuzz (<30s)
  uv run python scripts/check.py --ui     # Also run headless Chrome CDP UAT + property-based UI explorer
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, cmd: list[str]) -> float:
    print(f"\n==> {label}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {label} ({elapsed:.2f}s)", file=sys.stderr)
        raise SystemExit(proc.returncode)
    print(f"[OK]   {label} ({elapsed:.2f}s)")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mockery's deterministic Inner Loop verification suite.")
    parser.add_argument("--ui", action="store_true", help="Also run Browser UAT and Bombadil-style UI invariant explorer.")
    parser.add_argument("--explore-steps", type=int, default=35, help="Number of steps for UI explorer when --ui is set.")
    parser.add_argument("--explore-seed", type=int, default=42, help="Deterministic RNG seed for UI explorer.")
    args = parser.parse_args()

    t_start = time.perf_counter()

    _run("Ruff static analysis", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"])
    _run("Architecture, boundary, DOM-ID & state-chart lint", [sys.executable, "scripts/lint_arch.py"])

    js_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src" / "mockery" / "static").glob("*.js"))
    for js_file in js_files:
        proc = subprocess.run(["node", "--check", js_file], cwd=ROOT)
        if proc.returncode != 0:
            print(f"[FAIL] node --check {js_file}", file=sys.stderr)
            return proc.returncode
    print(f"\n[OK]   Node syntax check ({len(js_files)} static JS files)")

    _run("Pytest (unit + state machines + FSM verification + property fuzzing)", [sys.executable, "-m", "pytest", "-q"])

    if args.ui:
        _run("Headless Chrome CDP Browser UAT", [sys.executable, "scripts/verify_ui.py", "uat"])
        _run(
            f"Property-Based UI Explorer (seed={args.explore_seed}, steps={args.explore_steps})",
            [
                sys.executable,
                "scripts/verify_ui.py",
                "explore",
                "--steps",
                str(args.explore_steps),
                "--seed",
                str(args.explore_seed),
            ],
        )

    total = time.perf_counter() - t_start
    print(f"\nAll Inner Loop checks passed in {total:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
