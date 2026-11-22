"""Phase 2 State Machines Audit tests: client FSMs, modal races, redaction epochs, pack ops."""

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


def test_all_static_js_syntax_valid():
    for js_file in sorted(STATIC_DIR.glob("*.js")):
        proc = subprocess.run(
            ["node", "--check", str(js_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Syntax error in {js_file.name}:\n{proc.stderr}"


def test_client_machine_and_registry_transitions():
    mod_url = (STATIC_DIR / "state_machines.js").as_uri()
    res = _run_node_esm(
        f"""
        import {{
          createMachineRegistry,
          createMachine,
          createPracticeMicMachine,
          createPracticeOpMachine,
          createRetryTimerMachine,
          createWorkspaceSaveMachine,
          formatApiDetail,
        }} from {json.dumps(mod_url)};

        const reg = createMachineRegistry({{ maxTransitions: 5 }});
        const mic = createPracticeMicMachine(reg);
        const op = createPracticeOpMachine(reg);
        const timer = createRetryTimerMachine(reg);
        const save = createWorkspaceSaveMachine(reg);

        // Illegal direct jump (skipping mic permission) rejected cleanly without mutating state
        const illegalMic = mic.transition("recording", {{ event: "bad" }});
        // Direct idle -> transcribing is allowed for ASR clip retry
        const okRetryTx = mic.transition("transcribing", {{ event: "retry_transcribe" }});
        const okRetryIdle = mic.transition("idle", {{ event: "retry_done" }});
        const okReq = mic.transition("requesting_mic", {{ event: "request" }});
        const okRec = mic.transition("recording", {{ event: "granted" }});
        const okTx = mic.transition("transcribing", {{ event: "stop" }});
        const okIdle = mic.transition("idle", {{ event: "done" }});

        // Strict mode throws on illegal transition
        let strictThrew = false;
        const strictM = createMachine({{
          name: "strict_test",
          initial: "a",
          transitions: {{ a: ["b"], b: ["a"] }},
          strict: true,
        }});
        try {{
          strictM.transition("c");
        }} catch {{
          strictThrew = true;
        }}

        // Practice op mutual exclusion + exiting preemption
        const opStart = op.transition("submitting_turn");
        const opDouble = op.transition("debriefing");
        const opExit = op.transition("exiting");
        const opDone = op.transition("idle");

        // Retry timer transitions
        const tRun = timer.transition("running");
        const tWarn = timer.transition("warning");
        const tExp = timer.transition("expired");
        const tRestart = timer.transition("running");

        // Workspace save transitions
        const sDirty = save.transition("dirty");
        const sSaving = save.transition("saving");
        const sErr = save.transition("error");
        const sRetry = save.transition("saving");
        const sSaved = save.transition("saved");

        // Ring buffer cap & op tracking
        reg.beginOp("test_op", {{ k: 1 }});
        const opsWhileActive = reg.listOps();
        reg.endOp("test_op");
        const snap = reg.snapshot();

        const detailStr = formatApiDetail("plain msg", "fb");
        const detailObj = formatApiDetail({{ code: "transition_rejected", message: "Locked" }}, "fb");
        const detailRem = formatApiDetail({{ remediation: "Retry later" }}, "fb");
        const detailFb = formatApiDetail(null, "fallback");

        console.log(JSON.stringify({{
          illegalMic, okRetryTx, okRetryIdle, okReq, okRec, okTx, okIdle,
          strictThrew,
          opStart, opDouble, opExit, opDone,
          tRun, tWarn, tExp, tRestart,
          sDirty, sSaving, sErr, sRetry, sSaved,
          opsWhileActiveLen: opsWhileActive.length,
          opsAfterEndLen: snap.activeOps.length,
          recentLen: snap.recentTransitions.length,
          registeredMachines: Object.keys(snap.machines).sort(),
          detailStr, detailObj, detailRem, detailFb,
        }}));
        """
    )
    assert res["illegalMic"] is False
    assert res["okRetryTx"] is True
    assert res["okRetryIdle"] is True
    assert res["okReq"] is True
    assert res["okRec"] is True
    assert res["okTx"] is True
    assert res["okIdle"] is True
    assert res["strictThrew"] is True
    assert res["opStart"] is True
    assert res["opDouble"] is False
    assert res["opExit"] is True
    assert res["opDone"] is True
    assert res["tRun"] is True
    assert res["tWarn"] is True
    assert res["tExp"] is True
    assert res["tRestart"] is True
    assert res["sDirty"] is True
    assert res["sSaving"] is True
    assert res["sErr"] is True
    assert res["sRetry"] is True
    assert res["sSaved"] is True
    assert res["opsWhileActiveLen"] == 1
    assert res["opsAfterEndLen"] == 0
    assert res["recentLen"] == 5  # ring buffer capped at maxTransitions=5
    assert res["registeredMachines"] == [
        "practice_mic",
        "practice_op",
        "practice_retry_timer",
        "practice_workspace_save",
    ]
    assert res["detailStr"] == "plain msg"
    assert res["detailObj"] == "Locked"
    assert res["detailRem"] == "Retry later"
    assert res["detailFb"] == "fallback"


def test_derive_app_route_and_modal_token_races():
    mod_url = (STATIC_DIR / "state_machines.js").as_uri()
    res = _run_node_esm(
        f"""
        import {{
          deriveAppRoute,
          createModalTokenController,
          createMachineRegistry,
        }} from {json.dumps(mod_url)};

        const routes = {{
          launchpad: deriveAppRoute({{}}),
          practice: deriveAppRoute({{ practiceActive: true, forceWorkspace: true }}),
          capturing: deriveAppRoute({{ captureState: "recording" }}),
          content: deriveAppRoute({{ hasCopilotContent: true }}),
          forced: deriveAppRoute({{ forceWorkspace: true }}),
        }};

        const reg = createMachineRegistry();
        const modal = createModalTokenController({{ registry: reg }});

        // Race 1: User clicks History (async token 1), then clicks Settings (async token 2) before History resolves
        const tHist = modal.beginRequest();
        const tSet = modal.beginRequest();
        const setApplied = modal.set("settings", {{ reqToken: tSet, event: "open_settings" }});
        const staleHistApplied = modal.set("history", {{ reqToken: tHist, event: "open_history" }});
        const afterRace1 = modal.state;

        // Race 2: Onboarding opens with token, user switches to Settings before 450ms closeIf timer fires
        const tOnb = modal.beginRequest();
        modal.set("onboarding", {{ reqToken: tOnb, event: "open_onboarding" }});
        modal.set("settings", {{ event: "sync_open_settings" }});
        const staleTimerClosed = modal.closeIf("onboarding", tOnb);
        const afterRace2 = modal.state;

        // Happy closeIf when still on onboarding with matching token
        const tOnb2 = modal.beginRequest();
        modal.set("onboarding", {{ reqToken: tOnb2 }});
        const validTimerClosed = modal.closeIf("onboarding", tOnb2);
        const afterValidClose = modal.state;

        console.log(JSON.stringify({{
          routes,
          setApplied,
          staleHistApplied,
          afterRace1,
          staleTimerClosed,
          afterRace2,
          validTimerClosed,
          afterValidClose,
        }}));
        """
    )
    assert res["routes"] == {
        "launchpad": "launchpad",
        "practice": "practice",
        "capturing": "copilot_workspace",
        "content": "copilot_workspace",
        "forced": "copilot_workspace",
    }
    assert res["setApplied"] is True
    assert res["staleHistApplied"] is False
    assert res["afterRace1"] == "settings"
    assert res["staleTimerClosed"] is False
    assert res["afterRace2"] == "settings"
    assert res["validTimerClosed"] is True
    assert res["afterValidClose"] == "none"


def test_redaction_queue_epoch_invalidation_and_pack_op_tracker():
    mod_url = (STATIC_DIR / "state_machines.js").as_uri()
    res = _run_node_esm(
        f"""
        import {{
          createRedactionQueueCoordinator,
          createPackOpTracker,
          createMachineRegistry,
        }} from {json.dumps(mod_url)};

        const coord = createRedactionQueueCoordinator();

        // Coalesce consecutive prefs while keeping toggles in order
        coord.enqueue({{ type: "prefs", payload: {{ a: 1 }} }});
        coord.enqueue({{ type: "prefs", payload: {{ a: 2 }} }});
        coord.enqueue({{ type: "toggle", spanId: "s1" }});
        coord.enqueue({{ type: "prefs", payload: {{ a: 3 }} }});

        const executed = [];
        let staleSawCurrent = null;
        await coord.drain(async (op, isCurrent) => {{
          executed.push(op.type === "prefs" ? `prefs:${{op.payload.a}}` : `toggle:${{op.spanId}}`);
          if (op.type === "toggle") {{
            // Simulate resume delete mid-flight while toggle is awaiting network
            coord.invalidate();
            staleSawCurrent = isCurrent();
          }}
        }});

        // Shared pack op tracker prevents concurrent install/uninstall across panels
        const reg = createMachineRegistry();
        const packOps = createPackOpTracker(reg);
        const events = [];
        const unsub = packOps.subscribe((packId, op) => events.push(`${{packId}}:${{op}}`));

        const b1 = packOps.begin("pyodide_python", "installing");
        const b2 = packOps.begin("pyodide_python", "uninstalling");
        const snapDuring = packOps.snapshot();
        packOps.end("pyodide_python");
        const b3 = packOps.begin("pyodide_python", "uninstalling");
        packOps.end("pyodide_python");
        unsub();

        console.log(JSON.stringify({{
          executed,
          staleSawCurrent,
          pendingAfter: coord.pendingCount,
          b1, b2, b3,
          snapDuring,
          snapAfter: packOps.snapshot(),
          events,
        }}));
        """
    )
    # prefs:1 was coalesced into prefs:2; toggle:s1 invalidated epoch so prefs:3 was discarded!
    assert res["executed"] == ["prefs:2", "toggle:s1"]
    assert res["staleSawCurrent"] is False
    assert res["pendingAfter"] == 0
    assert res["b1"] is True
    assert res["b2"] is False
    assert res["b3"] is True
    assert res["snapDuring"] == {"pyodide_python": "installing"}
    assert res["snapAfter"] == {}
    assert res["events"] == [
        "pyodide_python:installing",
        "pyodide_python:idle",
        "pyodide_python:uninstalling",
        "pyodide_python:idle",
    ]


def test_phase2_ui_contracts_present():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    practice_js = (STATIC_DIR / "practice_studio.js").read_text(encoding="utf-8")
    profile_js = (STATIC_DIR / "profile_addons.js").read_text(encoding="utf-8")

    assert 'id="practiceWorkspaceSaveStatus"' in html
    assert "deriveAppRoute" in app_js
    assert "createModalTokenController" in app_js
    assert "createPackOpTracker" in app_js
    assert "recentTransitions" in app_js
    assert "createPracticeOpMachine" in practice_js
    assert "createWorkspaceSaveMachine" in practice_js
    assert "createRedactionQueueCoordinator" in profile_js
    assert "data-retry-banks" in profile_js


def test_phase2_codex_regression_guards():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    practice_js = (STATIC_DIR / "practice_studio.js").read_text(encoding="utf-8")
    profile_js = (STATIC_DIR / "profile_addons.js").read_text(encoding="utf-8")
    coding_js = (STATIC_DIR / "coding_studio.js").read_text(encoding="utf-8")

    # P2-1: Retry pill must call pushWorkspaceSnapshot() with no snapshot override
    assert "pushWorkspaceSnapshot({ force: true })" not in practice_js
    # P2-2: Exit and debrief preserve edits on failed save
    assert "exit_blocked_unsaved" in practice_js
    assert "Could not save your latest workspace edits" in practice_js
    # P2-3: Exit blocks mic & studio remount
    assert "mic_aborted_during_exit" in practice_js
    # P2-4: Bidirectional Practice start/open vs Copilot capture mutual exclusion
    assert "isPracticeStartingOrActive" in practice_js
    assert "isPracticeStartingOrActive" in app_js
    # P2-5: Transcription tagged with question ID
    assert "transcribe_discarded_question_changed" in practice_js
    # P2-6: Failed pack install refreshes /api/addons in both profile_addons and coding_studio
    assert "refreshedFromResponse" in profile_js
    assert "refreshedFromResponse" in coding_js
    # P3-2: Profile redraw respects banksState === "ready"
    assert 'local.banksState === "ready"' in profile_js
    # P3-3: Settings save closes only matching modal token
    assert 'modalCtrl.closeIf("settings", settingsOpenToken)' in app_js
    # P3-4: WS reconnect avoids duplicate start_capture when flushed from outbox
    assert "flushedStart" in app_js
    # P3-5: Module toggle patches in place to keep focus
    assert "canPatchInPlace" in profile_js


def test_inference_latency_indicators_contract():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    practice_js = (STATIC_DIR / "practice_studio.js").read_text(encoding="utf-8")

    # Voice preparing vs speaking barge-in button markup
    assert 'id="practiceBargeInLabel"' in html
    assert 'class="speaking-bars"' in html

    # Shared animation & indicator primitives in styles.css (with reduced-motion coverage)
    for token in (
        "@keyframes spin-ring",
        "@keyframes typing-wave",
        "@keyframes progress-glide",
        "@keyframes eq-bar",
        ".btn.is-loading::before",
        ".typing-dots",
        ".pturn.is-thinking",
        ".practice-coach-card.is-thinking",
        ".debrief-pending-card",
        ".speaking-bars",
        ".practice-speaking-btn.is-preparing",
        ".practice-mic-btn.is-transcribing",
        "prefers-reduced-motion: reduce",
    ):
        assert token in css, f"Missing CSS primitive: {token}"

    # Practice Studio staged latency indicators & clean persistence across debrief polling
    for hook in (
        "speakingState",
        'setSpeakingUI("preparing"',
        'setSpeakingUI("speaking"',
        "showPendingTurnCard",
        "showPendingDebriefCard",
        "setCoachRailThinking",
        "clearPendingOpUI",
        "debriefStartedAt",
        "is-transcribing",
    ):
        assert hook in practice_js, f"Missing practice_studio.js hook: {hook}"

    # Live Copilot probes, calibration, and LLM test indicators
    assert "typing-dots" in app_js
    assert 'calibBtn.classList.toggle("is-loading", calibRunning)' in app_js
    assert 'btn.classList.add("is-loading")' in app_js


def test_ux92_interactive_ergonomics_contract():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    practice_js = (STATIC_DIR / "practice_studio.js").read_text(encoding="utf-8")
    profile_js = (STATIC_DIR / "profile_addons.js").read_text(encoding="utf-8")

    # Split studio: resizable divider (outside the studio slot, so plugin pointer capture is untouched)
    assert 'id="practiceSplitHandle"' in html
    assert 'role="separator"' in html
    assert html.index('id="practiceStudioSlot"') < html.index('id="practiceSplitHandle"') < html.index(
        'id="practiceActiveView"'
    )
    assert 'id="practiceConvToggleBtn"' in html
    for token in (
        ".practice-split-handle",
        "cursor: col-resize",
        "clamp(260px, var(--split-pct, max(300px, 30%)), 560px)",
        "clamp(260px, var(--split-pct, max(320px, 34%)), 560px)",
        "[data-conv-collapsed]",
    ):
        assert token in css, f"Missing split-studio CSS: {token}"
    for hook in (
        "mockery_split_pct_",
        "bindSplitHandle",
        "setPointerCapture",
        '"dblclick"',
        "setConversationCollapsed",
    ):
        assert hook in practice_js, f"Missing split-studio hook: {hook}"

    # Pending debrief: the whole composer footer steps aside so the staged progress
    # card is the single status on screen (no competing footer line), restored by phase
    assert ".practice-composer.is-debriefing {" in css
    assert 'classList.toggle("is-debriefing", isDebriefing)' in practice_js
    assert "Finalizing your coaching debrief…" not in practice_js

    # Home ↔ active target JD alignment (role + level), hand edits always win
    assert "autoAlignHomeToJd" in profile_js
    assert "roleEditedByHand" in profile_js
    assert "From your active target job" in profile_js
    assert "setTargetLevel" in practice_js
    assert "levelTouched" in practice_js
    assert "onLevelApply" in app_js
    assert "isPracticeHome" in app_js
    # Stock defaults stay intact for an empty profile
    assert 'value="Senior Software Engineer"' in html
    assert 'class="seg-opt active" data-practice-level="L5"' in html
