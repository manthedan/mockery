"""Lightweight formal verification of Python + JS state machines and cross-language parity.

Checks structural graph invariants over every transition table in:
  - src/mockery/state_machines.py
  - src/mockery/static/state_machines.js
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
from enum import StrEnum
from pathlib import Path

from mockery.state_machines import (
    ADDON_PACK_TRANSITIONS,
    ASR_MODEL_TRANSITIONS,
    DIARIZER_ENROLLMENT_TRANSITIONS,
    PRACTICE_PHASE_TRANSITIONS,
    SESSION_PHASE_TRANSITIONS,
    VAD_TURN_TRANSITIONS,
    AddonPackState,
    ASRModelState,
    DiarizerEnrollmentState,
    PracticePhase,
    SessionPhase,
    VADTurnState,
    export_machine_specs,
)

ROOT = Path(__file__).resolve().parent.parent


def _bfs_reachable(initial: StrEnum, graph: dict[StrEnum, set[StrEnum]]) -> set[StrEnum]:
    seen = {initial}
    q: deque[StrEnum] = deque([initial])
    while q:
        cur = q.popleft()
        for nxt in graph.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _can_reach_any(start: StrEnum, targets: set[StrEnum], graph: dict[StrEnum, set[StrEnum]]) -> bool:
    return bool(_bfs_reachable(start, graph) & targets)


def test_python_fsm_completeness_reachability_and_no_unintended_deadlocks() -> None:
    """Every Python FSM must cover all enum members, be reachable, and allow recovery from transient states."""
    cases: list[tuple[str, type[StrEnum], dict[StrEnum, set[StrEnum]], StrEnum, set[StrEnum], set[StrEnum]]] = [
        (
            "vad_turn",
            VADTurnState,
            VAD_TURN_TRANSITIONS,
            VADTurnState.SILENCE,
            set(),
            {VADTurnState.SILENCE},
        ),
        (
            "asr_model",
            ASRModelState,
            ASR_MODEL_TRANSITIONS,
            ASRModelState.IDLE,
            set(),
            {ASRModelState.READY},
        ),
        (
            "session_phase",
            SessionPhase,
            SESSION_PHASE_TRANSITIONS,
            SessionPhase.READY,
            set(),
            {SessionPhase.CALIBRATED, SessionPhase.CAPTURING},
        ),
        (
            "practice_phase",
            PracticePhase,
            PRACTICE_PHASE_TRANSITIONS,
            PracticePhase.ACTIVE,
            set(),
            {PracticePhase.COMPLETED},
        ),
        (
            "addon_pack",
            AddonPackState,
            ADDON_PACK_TRANSITIONS,
            AddonPackState.NOT_INSTALLED,
            {AddonPackState.COMING_SOON},  # feature-flagged static state
            {AddonPackState.NOT_INSTALLED, AddonPackState.INSTALLED},
        ),
        (
            "diarizer_enrollment",
            DiarizerEnrollmentState,
            DIARIZER_ENROLLMENT_TRANSITIONS,
            DiarizerEnrollmentState.UNLOADED,
            set(),
            {DiarizerEnrollmentState.COLD_START, DiarizerEnrollmentState.MANUAL_CALIBRATED},
        ),
    ]

    for name, enum_cls, transitions, initial, intentionally_isolated, recovery_targets in cases:
        members = set(enum_cls)
        assert set(transitions.keys()) == members, f"[{name}] missing enum keys in transition table"
        for src, dsts in transitions.items():
            assert dsts <= members, f"[{name}] unknown target state(s) from {src}: {dsts - members}"

        reachable = _bfs_reachable(initial, transitions)
        expected_reachable = members - intentionally_isolated
        assert reachable == expected_reachable, (
            f"[{name}] unreachable states from {initial}: {expected_reachable - reachable}"
        )

        # Every reachable state must have a path to a stable/recovery target (zero unintended traps/deadlocks)
        for st in expected_reachable:
            assert _can_reach_any(st, recovery_targets, transitions), (
                f"[{name}] state '{st.value}' cannot reach any stable recovery state {recovery_targets}"
            )

    # Every transient/in-flight state must explicitly permit recovery (not stuck forever)
    assert PracticePhase.ACTIVE in PRACTICE_PHASE_TRANSITIONS[PracticePhase.DEBRIEFING]
    assert PracticePhase.AWAITING_DEBRIEF in PRACTICE_PHASE_TRANSITIONS[PracticePhase.DEBRIEFING]
    assert PracticePhase.COMPLETED in PRACTICE_PHASE_TRANSITIONS[PracticePhase.DEBRIEFING]
    assert AddonPackState.ERROR in ADDON_PACK_TRANSITIONS[AddonPackState.INSTALLING]
    assert AddonPackState.INSTALLING in ADDON_PACK_TRANSITIONS[AddonPackState.ERROR]


def test_export_machine_specs_matches_live_enums() -> None:
    specs = export_machine_specs()["machines"]
    assert specs["practice_phase"]["states"] == [s.value for s in PracticePhase]
    assert specs["session_phase"]["states"] == [s.value for s in SessionPhase]
    assert specs["addon_pack"]["states"] == [s.value for s in AddonPackState]
    assert specs["vad_turn"]["states"] == [s.value for s in VADTurnState]
    assert specs["asr_model"]["states"] == [s.value for s in ASRModelState]
    assert specs["diarizer_enrollment"]["states"] == [s.value for s in DiarizerEnrollmentState]


def test_js_client_fsms_reachability_recovery_and_python_parity() -> None:
    """Verify all JS client FSMs via Node and confirm PracticePhase parity with Python."""
    py_practice_phases = [s.value for s in PracticePhase]
    node_script = f"""
import * as sm from "./src/mockery/static/state_machines.js";

const factories = [
  ["practice_mic", () => sm.createPracticeMicMachine()],
  ["practice_op", () => sm.createPracticeOpMachine()],
  ["practice_retry_timer", () => sm.createRetryTimerMachine()],
  ["practice_workspace_save", () => sm.createWorkspaceSaveMachine()],
];

const report = {{ machines: {{}}, phaseActions: {{}} }};

for (const [name, make] of factories) {{
  const m = make();
  const initial = m.state;
  const visited = new Set([initial]);
  const queue = [initial];
  const graph = {{}};
  while (queue.length > 0) {{
    const s = queue.shift();
    m.reset(s, {{ event: "verify" }});
    const nexts = m.allowedNext();
    graph[s] = nexts;
    for (const n of nexts) {{
      if (!visited.has(n)) {{
        visited.add(n);
        queue.push(n);
      }}
    }}
  }}

  // Every reachable state must have a path back to `initial` (no stuck client UI states)
  const stuck = [];
  for (const st of visited) {{
    if (st === initial) continue;
    const seen = new Set([st]);
    const q = [st];
    let canReturn = false;
    while (q.length > 0) {{
      const cur = q.shift();
      for (const nxt of (graph[cur] || [])) {{
        if (nxt === initial) {{
          canReturn = true;
          break;
        }}
        if (!seen.has(nxt)) {{
          seen.add(nxt);
          q.push(nxt);
        }}
      }}
      if (canReturn) break;
    }}
    if (!canReturn) stuck.push(st);
  }}

  // Strict mode rejects unknown transition
  const strictMachine = sm.createMachine({{
    name: name + "_strict",
    initial,
    transitions: graph,
    strict: true,
  }});
  let rejectedIllegal = false;
  try {{
    strictMachine.transition("__impossible_state__");
  }} catch {{
    rejectedIllegal = true;
  }}

  report.machines[name] = {{
    initial,
    states: Array.from(visited),
    stuck,
    rejectedIllegal,
  }};
}}

// Cross-language PracticePhase parity: deriveAppAllowedActions handles every Python PracticePhase
for (const phase of {json.dumps(py_practice_phases)}) {{
  report.phaseActions[phase] = sm.deriveAppAllowedActions({{
    route: "practice",
    modalState: "none",
    practice: {{ phase, opState: "idle", micState: "idle" }},
  }});
}}

process.stdout.write(JSON.stringify(report));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(proc.stdout)

    for name, info in report["machines"].items():
        assert len(info["states"]) >= 3, f"JS machine '{name}' has unexpectedly few states: {info['states']}"
        assert info["stuck"] == [], f"JS machine '{name}' has stuck state(s) that cannot return to '{info['initial']}': {info['stuck']}"
        assert info["rejectedIllegal"] is True, f"JS machine '{name}' failed to reject illegal transition in strict mode"

    # Cross-language phase action parity assertions
    assert "submit_practice_turn" in report["phaseActions"]["active"]
    assert "finish_practice_debrief" in report["phaseActions"]["active"]
    assert "finish_practice_debrief" in report["phaseActions"]["awaiting_debrief"]
    assert "submit_practice_turn" not in report["phaseActions"]["awaiting_debrief"]
    assert "submit_practice_turn" not in report["phaseActions"]["debriefing"]
    assert "restart_practice" in report["phaseActions"]["completed"]
    assert "submit_practice_turn" not in report["phaseActions"]["completed"]
