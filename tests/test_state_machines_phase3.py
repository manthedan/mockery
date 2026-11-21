"""Phase 3 State Machines Audit tests: Design Studio FSM, Coding Studio runner FSM, unified AX snapshot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "src" / "mockery" / "static"


def _run_node_esm(script: str) -> dict:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return json.loads(proc.stdout.strip())


def test_design_studio_discriminated_interaction_fsm():
    mod_url = (STATIC_DIR / "design_studio.js").as_uri()
    res = _run_node_esm(
        f"""
        import {{
          createDesignInteractionState,
          reduceDesignInteraction,
          getSelectedFromInteraction,
          deriveDesignAllowedActions,
        }} from {json.dumps(mod_url)};

        let s = createDesignInteractionState();
        const initialMode = s.mode;

        // Select node -> mark dirty -> switch item resets dirty
        s = reduceDesignInteraction(s, {{ type: "SELECT_ITEM", kind: "node", id: "n1" }});
        s = reduceDesignInteraction(s, {{ type: "MARK_TEXT_DIRTY" }});
        const dirtyAfterMark = s.textEditDirty;
        s = reduceDesignInteraction(s, {{ type: "SELECT_ITEM", kind: "edge", id: "e1" }});
        const dirtyAfterSwitch = s.textEditDirty;
        const edgeActions = deriveDesignAllowedActions({{
          interaction: s,
          nodeCount: 2,
          edgeCount: 1,
          historyLen: 1,
          futureLen: 0,
        }});

        // Start drag from selected -> blocks connect/select/tool switching
        const snap = {{ version: 1, nodes: [{{ id: "n1", x: 48, y: 48 }}], edges: [] }};
        s = reduceDesignInteraction(s, {{
          type: "START_DRAG",
          nodeId: "n1",
          pointerId: 7,
          startX: 48,
          startY: 48,
          snapshot: snap,
        }});
        const modeDuringDrag = s.mode;
        const selDuringDrag = getSelectedFromInteraction(s);
        const dragActions = deriveDesignAllowedActions({{
          interaction: s,
          nodeCount: 2,
          edgeCount: 1,
          historyLen: 3,
          futureLen: 2,
        }});

        // Mutating mode switches ignored while dragging
        const blockedConnect = reduceDesignInteraction(s, {{ type: "START_CONNECT", fromId: "n1" }});
        const blockedSelect = reduceDesignInteraction(s, {{ type: "SELECT_ITEM", kind: "edge", id: "e1" }});
        const wrongPointerMove = reduceDesignInteraction(s, {{ type: "MOVE_DRAG", pointerId: 99 }});
        const movedState = reduceDesignInteraction(s, {{ type: "MOVE_DRAG", pointerId: 7 }});

        // Escape / unconditional cancel drag (pointerId: null) returns to selected node
        const afterEscDrag = reduceDesignInteraction(movedState, {{ type: "CANCEL_DRAG", pointerId: null }});

        // Connecting mode blocks START_DRAG and toggles source on same node
        let c = reduceDesignInteraction(afterEscDrag, {{ type: "START_CONNECT", fromId: null }});
        const blockedDragInConnect = reduceDesignInteraction(c, {{ type: "START_DRAG", nodeId: "n1", pointerId: 1 }});
        c = reduceDesignInteraction(c, {{ type: "SET_CONNECT_SOURCE", fromId: "n1" }});
        const connectSrcSet = c.fromId;
        c = reduceDesignInteraction(c, {{ type: "SET_CONNECT_SOURCE", fromId: "n1" }});
        const connectSrcToggled = c.fromId;
        c = reduceDesignInteraction(c, {{ type: "SET_CONNECT_SOURCE", fromId: "n2" }});
        c = reduceDesignInteraction(c, {{ type: "DESELECT" }});
        const connectAfterBgClick = {{ mode: c.mode, fromId: c.fromId }};
        c = reduceDesignInteraction(c, {{ type: "CANCEL_CONNECT" }});
        const afterCancelConnect = c.mode;

        console.log(JSON.stringify({{
          initialMode,
          dirtyAfterMark,
          dirtyAfterSwitch,
          edgeActions,
          modeDuringDrag,
          selDuringDrag,
          dragActions,
          blockedConnectMode: blockedConnect.mode,
          blockedSelectMode: blockedSelect.mode,
          wrongPointerMoved: wrongPointerMove.moved,
          rightPointerMoved: movedState.moved,
          afterEscDrag,
          blockedDragInConnectMode: blockedDragInConnect.mode,
          connectSrcSet,
          connectSrcToggled,
          connectAfterBgClick,
          afterCancelConnect,
        }}));
        """
    )

    assert res["initialMode"] == "idle"
    assert res["dirtyAfterMark"] is True
    assert res["dirtyAfterSwitch"] is False
    assert "reverse_edge" in res["edgeActions"]
    assert "undo" in res["edgeActions"]
    assert "redo" not in res["edgeActions"]
    assert res["modeDuringDrag"] == "dragging"
    assert res["selDuringDrag"] == {"kind": "node", "id": "n1"}
    assert res["dragActions"] == ["move_node", "commit_drag", "cancel_drag"]
    assert res["blockedConnectMode"] == "dragging"
    assert res["blockedSelectMode"] == "dragging"
    assert res["wrongPointerMoved"] is False
    assert res["rightPointerMoved"] is True
    assert res["afterEscDrag"] == {"mode": "selected", "kind": "node", "id": "n1", "textEditDirty": False}
    assert res["blockedDragInConnectMode"] == "connecting"
    assert res["connectSrcSet"] == "n1"
    assert res["connectSrcToggled"] is None
    assert res["connectAfterBgClick"] == {"mode": "connecting", "fromId": None}
    assert res["afterCancelConnect"] == "idle"


def test_coding_studio_canonical_runner_state_derivation():
    mod_url = (STATIC_DIR / "coding_studio.js").as_uri()
    res = _run_node_esm(
        f"""
        import {{ deriveCodingRunnerState }} from {json.dumps(mod_url)};

        const jsRunningInJs = deriveCodingRunnerState({{
          language: "javascript",
          activeRun: {{ language: "javascript", booted: true, phase: "running" }},
          testsPresent: true,
        }});

        // Key UX bug fixed in Phase 3: switching to Python while JS is still running
        const jsRunningViewedInPy = deriveCodingRunnerState({{
          language: "python",
          activeRun: {{ language: "javascript", booted: true, phase: "running" }},
          testsPresent: true,
          pyodideInstalled: true,
        }});

        // Python booting phases (starting vs loading vs ready)
        const pyStarting = deriveCodingRunnerState({{
          language: "python",
          activeRun: {{ language: "python", booted: false, phase: "starting" }},
          testsPresent: true,
          pyodideInstalled: true,
        }});
        const pyLoading = deriveCodingRunnerState({{
          language: "python",
          activeRun: {{ language: "python", booted: false, phase: "loading" }},
          testsPresent: true,
          pyodideInstalled: true,
        }});
        const pyRunning = deriveCodingRunnerState({{
          language: "python",
          activeRun: {{ language: "python", booted: true, phase: "running" }},
          testsPresent: true,
          pyodideInstalled: true,
        }});

        // JS run active while viewing Python with Pyodide missing still advertises install_pyodide
        const jsRunningViewedInPyMissing = deriveCodingRunnerState({{
          language: "python",
          activeRun: {{ language: "javascript", booted: true, phase: "running" }},
          testsPresent: true,
          pyodideInstalled: false,
        }});

        // Cancelling a rerun after a prior pass displays Cancelled while keeping matchingResult intact
        const cancelled = deriveCodingRunnerState({{
          language: "javascript",
          activeRun: null,
          matchingResult: {{ status: "passed", durationMs: 12 }},
          cancelled: true,
          testsPresent: true,
        }});

        const pyInstalling = deriveCodingRunnerState({{
          language: "python",
          testsPresent: true,
          pyodideInstalled: false,
          pyodideInstalling: true,
        }});

        const pyUninstalling = deriveCodingRunnerState({{
          language: "python",
          testsPresent: true,
          pyodideInstalled: true,
          pyodideUninstalling: true,
        }});

        const pyMissing = deriveCodingRunnerState({{
          language: "python",
          testsPresent: true,
          pyodideInstalled: false,
        }});

        const passed = deriveCodingRunnerState({{
          language: "javascript",
          matchingResult: {{ status: "passed", durationMs: 42 }},
          testsPresent: true,
        }});

        const stale = deriveCodingRunnerState({{
          language: "javascript",
          staleResult: true,
          testsPresent: true,
        }});

        const pseudo = deriveCodingRunnerState({{
          language: "pseudocode",
          testsPresent: false,
        }});

        console.log(JSON.stringify({{
          jsRunningInJs,
          jsRunningViewedInPy,
          jsRunningViewedInPyMissing,
          pyStarting,
          pyLoading,
          pyRunning,
          cancelled,
          pyInstalling,
          pyUninstalling,
          pyMissing,
          passed,
          stale,
          pseudo,
        }}));
        """
    )

    assert res["jsRunningInJs"]["status"] == "running"
    assert res["jsRunningInJs"]["canRun"] is False
    assert res["jsRunningInJs"]["canCancel"] is True
    assert "cancel_run" in res["jsRunningInJs"]["allowedActions"]
    assert "run_tests" not in res["jsRunningInJs"]["allowedActions"]

    assert res["jsRunningViewedInPy"]["status"] == "running_other_lang"
    assert res["jsRunningViewedInPy"]["pillText"] == "Running JS…"
    assert res["jsRunningViewedInPy"]["pillTone"] == "running"
    assert res["jsRunningViewedInPy"]["canRun"] is False
    assert res["jsRunningViewedInPy"]["canCancel"] is True
    assert "install_pyodide" not in res["jsRunningViewedInPy"]["allowedActions"]

    assert res["jsRunningViewedInPyMissing"]["status"] == "running_other_lang"
    assert "install_pyodide" in res["jsRunningViewedInPyMissing"]["allowedActions"]

    assert res["pyStarting"]["status"] == "booting_pyodide"
    assert res["pyStarting"]["pillText"] == "Starting Python…"
    assert res["pyLoading"]["status"] == "booting_pyodide"
    assert res["pyLoading"]["pillText"] == "Loading Python…"
    assert res["pyRunning"]["status"] == "running"
    assert res["pyRunning"]["pillText"] == "Running…"

    assert res["cancelled"]["status"] == "cancelled"
    assert res["cancelled"]["pillText"] == "Cancelled"
    assert res["cancelled"]["canRun"] is True
    assert res["cancelled"]["canCancel"] is False

    assert res["pyInstalling"]["status"] == "pyodide_installing"
    assert res["pyInstalling"]["canRun"] is False
    assert res["pyUninstalling"]["status"] == "pyodide_uninstalling"
    assert res["pyUninstalling"]["canRun"] is False
    assert res["pyMissing"]["status"] == "pyodide_missing"
    assert "install_pyodide" in res["pyMissing"]["allowedActions"]

    assert res["passed"]["status"] == "passed"
    assert res["passed"]["pillText"] == "Tests Passed · 42 ms"
    assert res["passed"]["pillTone"] == "pass"
    assert res["passed"]["consoleTone"] == "pass"

    assert res["stale"]["status"] == "stale"
    assert res["stale"]["pillText"] == "Modified"
    assert res["stale"]["canRun"] is True

    assert res["pseudo"]["status"] == "pseudocode"
    assert res["pseudo"]["canRun"] is False


def test_unified_app_allowed_actions_derivation():
    mod_url = (STATIC_DIR / "state_machines.js").as_uri()
    res = _run_node_esm(
        f"""
        import {{ deriveAppAllowedActions }} from {json.dumps(mod_url)};

        const settingsOpen = deriveAppAllowedActions({{
          route: "practice",
          modalState: "settings",
        }});

        const launchpadPractice = deriveAppAllowedActions({{
          route: "launchpad",
          modalState: "none",
          captureState: "idle",
          practice: {{ practiceEnabled: true, homeMode: "practice", opState: "idle" }},
        }});

        const launchpadCopilotPracticeDisabled = deriveAppAllowedActions({{
          route: "launchpad",
          modalState: "none",
          captureState: "idle",
          practice: {{ practiceEnabled: false, homeMode: "copilot", opState: "idle" }},
        }});

        // Real captureState during Copilot recording is "streaming"
        const copilotStreaming = deriveAppAllowedActions({{
          route: "copilot_workspace",
          modalState: "none",
          captureState: "streaming",
          wsState: "connected",
          serverAllowedEvents: ["stop_capture", "dive_deeper", "force_new_card"],
        }});

        const practiceWithCoding = deriveAppAllowedActions({{
          route: "practice",
          modalState: "none",
          practice: {{
            phase: "active",
            opState: "idle",
            micState: "idle",
            retryTimerState: "running",
            workspaceSaveState: "error",
            mountedStudio: {{
              studio: "coding",
              allowedActions: ["cancel_run", "switch_language"],
            }},
          }},
        }});

        const practiceAwaitingDebrief = deriveAppAllowedActions({{
          route: "practice",
          modalState: "none",
          practice: {{
            phase: "awaiting_debrief",
            opState: "idle",
            micState: "idle",
            retryTimerState: "idle",
          }},
        }});

        const practiceCompletedWarning = deriveAppAllowedActions({{
          route: "practice",
          modalState: "none",
          practice: {{
            phase: "completed",
            opState: "idle",
            micState: "idle",
            retryTimerState: "warning",
          }},
        }});

        const practiceExiting = deriveAppAllowedActions({{
          route: "practice",
          modalState: "none",
          practice: {{
            phase: "active",
            opState: "exiting",
            micState: "idle",
          }},
        }});

        console.log(JSON.stringify({{
          settingsOpen,
          launchpadPractice,
          launchpadCopilotPracticeDisabled,
          copilotStreaming,
          practiceWithCoding,
          practiceAwaitingDebrief,
          practiceCompletedWarning,
          practiceExiting,
        }}));
        """
    )

    assert res["settingsOpen"] == ["close_modal", "save_settings"]

    # Launchpad only advertises the active mode's start CTA and respects practiceEnabled
    assert "start_practice" in res["launchpadPractice"]
    assert "start_copilot_capture" not in res["launchpadPractice"]
    assert "switch_home_copilot" in res["launchpadPractice"]

    assert "start_copilot_capture" in res["launchpadCopilotPracticeDisabled"]
    assert "start_practice" not in res["launchpadCopilotPracticeDisabled"]
    assert "switch_home_practice" not in res["launchpadCopilotPracticeDisabled"]

    # Copilot streaming exposes stop_copilot_capture
    assert "stop_copilot_capture" in res["copilotStreaming"]
    assert "dive_deeper" in res["copilotStreaming"]
    assert "go_home" not in res["copilotStreaming"]

    assert "submit_practice_turn" in res["practiceWithCoding"]
    assert "start_practice_mic" in res["practiceWithCoding"]
    assert "cancel_practice_retry" in res["practiceWithCoding"]
    assert "retry_workspace_save" in res["practiceWithCoding"]
    assert "studio:cancel_run" in res["practiceWithCoding"]
    assert "studio:switch_language" in res["practiceWithCoding"]

    assert "finish_practice_debrief" in res["practiceAwaitingDebrief"]
    assert "submit_practice_turn" not in res["practiceAwaitingDebrief"]

    assert "restart_practice" in res["practiceCompletedWarning"]
    assert "cancel_practice_retry" in res["practiceCompletedWarning"]

    assert "exit_practice" not in res["practiceExiting"]
    assert "submit_practice_turn" not in res["practiceExiting"]


def test_phase3_frontend_contracts():
    design_js = (STATIC_DIR / "design_studio.js").read_text(encoding="utf-8")
    coding_js = (STATIC_DIR / "coding_studio.js").read_text(encoding="utf-8")
    practice_js = (STATIC_DIR / "practice_studio.js").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    # Design Studio FSM contracts
    assert 'name: "designStudio"' in design_js
    assert 'unregister?.("designStudio")' in design_js
    assert "cancelActiveDrag" in design_js
    assert 'ev.key === "Escape" && state.interaction.mode === "dragging"' in design_js
    assert "if (state.destroyed) return;" in design_js
    assert "lostpointercapture" in design_js
    assert "dragDisabled" in design_js
    assert "getStateSnapshot()" in design_js

    # Coding Studio FSM contracts
    assert 'name: "codingStudio"' in coding_js
    assert 'unregister?.("codingStudio")' in coding_js
    assert "data-cs-cancel" in coding_js
    assert 'msg.type === "status"' in coding_js
    assert "cancelledByLang" in coding_js
    assert "data-cs-editor-hint" in coding_js
    assert 'ev.key === "Shift"' in coding_js
    assert "getStateSnapshot()" in coding_js

    # Practice Studio exposes practiceEnabled for launchpad gating
    assert "practiceEnabled:" in practice_js

    # Unified AX snapshot preserves existing keys + adds allowedActions
    for key in [
        "wsState:",
        "captureState:",
        "modalState:",
        "rightTab:",
        "wideCopilot:",
        "selectedQaId:",
        "themePref:",
        "theme:",
        "asr:",
        "diarizer:",
        "addons:",
        "profileSummary:",
        "practiceActive:",
        "serverMachines",
        "route:",
        "practice:",
        "profileAddons:",
        "activeOps:",
        "recentTransitions:",
        "clientMachines:",
        "allowedActions:",
    ]:
        assert key in app_js, f"Missing expected __MOCKERY_STATE__ key: {key}"
