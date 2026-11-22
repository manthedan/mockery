import { createProfileAddonsController } from "./profile_addons.js";
import { createPracticeStudioController } from "./practice_studio.js";
import { createSystemDesignStudioPlugin } from "./design_studio.js";
import { createCodingStudioPlugin } from "./coding_studio.js";
import {
  createMachine,
  createMachineRegistry,
  createModalTokenController,
  createPackOpTracker,
  deriveAppAllowedActions,
  deriveAppRoute,
  formatApiDetail,
} from "./state_machines.js";

const CHANNEL_INTERVIEWER = 0x01;
const CHANNEL_CANDIDATE = 0x02;
const CHANNEL_ROOM_MIC = 0x03;

const TYPE_SHORT_LABELS = {
  behavioral: "Behavioral",
  system_design: "System Design",
  coding: "Coding",
  general: "General",
};

const SIGNAL_LABELS = {
  strong_hire: "Strong Hire",
  lean_hire: "Lean Hire",
  lean_no_hire: "Lean No",
  strong_no_hire: "Strong No",
};

const HUMANIZED_ENUMS = {
  strong_hire: "Strong Hire",
  lean_hire: "Lean Hire",
  lean_no_hire: "Lean No",
  strong_no_hire: "Strong No",
  system_design: "System Design",
  behavioral: "Behavioral",
  coding: "Coding",
  general: "General",
};

const PHASE_LABELS = {
  ready: "Not started",
  capturing: "Recording",
  awaiting_self_rating: "Waiting for your call",
  rated: "Call locked",
  calibrating: "Auditing…",
  calibrated: "Calibrated",
};

const MODE_NOTES = {
  remote_meet: "Your mic and the Meet tab are captured separately. Wear headphones.",
  in_person: "For in-person or solo practice. Speakers are told apart automatically.",
};

const START_HINTS = {
  remote_meet: "Chrome will ask for your Meet tab, then your microphone.",
  in_person: "Chrome will ask for your microphone.",
};

/* ---------------- Explicit Client Finite State Machines ---------------- */

const WS_TRANSITIONS = {
  connecting: new Set(["connected", "reconnecting"]),
  connected: new Set(["reconnecting"]),
  reconnecting: new Set(["connecting", "connected"]),
};

const CAPTURE_TRANSITIONS = {
  idle: new Set(["requesting_permissions"]),
  requesting_permissions: new Set(["streaming", "idle"]),
  streaming: new Set(["stopping", "idle"]),
  stopping: new Set(["idle"]),
};

const clientMachines = createMachineRegistry({ maxTransitions: 40 });
const packOps = createPackOpTracker(clientMachines);

const state = {
  wsState: "connecting",
  captureState: "idle",
  modalState: "none",
  route: "launchpad",
  rightTab: "probes",
  wideCopilot: false,
  editingSelfRating: false,
  searchQuery: "",
  simulatorOpen: false,
  forceWorkspace: false,

  ws: null,
  wsReconnectTimer: null,
  outbox: [],
  session: null,
  optimisticBackup: null,
  selectedQaId: null,
  asr: { state: "loading", generation: 0, last_inference_ms: 0 },
  diarizer: null,
  llm: null,
  manualSpeaker: "interviewer",
  askedProbes: new Set(),

  baseElapsedMs: 0,
  localSegmentStartedAt: null,
  timerHandle: null,
  stopFallbackTimer: null,
  toastTimer: null,
  navLockUntil: 0,
  pendingNewSession: false,
  pendingQARerender: false,
  savingSettings: false,
  loadingSavedSession: false,

  sharedAudioCtx: null,
  audioCleanupFns: [],
  audioFlushFns: [],
  lastSyncedSessionId: null,
  activeCitationUttId: null,
  homeVisible: false,
  streamingCalibrationMd: "",
  calibrationStreamRaf: null,
  discovery: null,
  onboardingChecked: false,
  onboardProvider: "gemini",
  onboardStep: "ai", // "ai" | "packs"
  addons: null,
  tts: null,
  profileSummary: null,
  practiceActive: false,
  clientMachines,
  packOps,
};

let profileAddons = null;
let practiceStudio = null;

const _mockeryStateDescriptor = {
  get() {
    const regSnap = clientMachines.snapshot();
    const practiceSnap = practiceStudio?.getStateSnapshot?.() || null;
    const serverMachines = state.session?.state_machines || null;
    const serverAllowedEvents = serverMachines?.session?.allowed_events || [];
    return {
      wsState: state.wsState,
      captureState: state.captureState,
      modalState: state.modalState,
      onboardStep: state.onboardStep,
      rightTab: state.rightTab,
      wideCopilot: state.wideCopilot,
      selectedQaId: state.selectedQaId,
      themePref: document.documentElement.getAttribute("data-theme-pref"),
      theme: document.documentElement.getAttribute("data-theme"),
      asr: state.asr,
      diarizer: state.diarizer,
      addons: state.addons,
      tts: state.tts,
      profileSummary: state.profileSummary,
      practiceActive: state.practiceActive,
      serverMachines,
      route: state.route,
      practice: practiceSnap,
      profileAddons: profileAddons?.getStateSnapshot?.() || null,
      activeOps: regSnap.activeOps,
      recentTransitions: regSnap.recentTransitions,
      clientMachines: regSnap.machines,
      allowedActions: deriveAppAllowedActions({
        route: state.route,
        modalState: state.modalState,
        captureState: state.captureState,
        wsState: state.wsState,
        serverAllowedEvents,
        practice: practiceSnap,
      }),
    };
  },
};
Object.defineProperty(window, "__MOCKERY_STATE__", _mockeryStateDescriptor);
Object.defineProperty(window, "__MOCK_HELPER_STATE__", _mockeryStateDescriptor);

const el = (id) => document.getElementById(id);

/* ---------------- Theme Management (Auto / Light / Dark) ---------------- */

const colorSchemeMql = window.matchMedia
  ? window.matchMedia("(prefers-color-scheme: light)")
  : null;

function applyTheme(pref) {
  const validPref = pref === "light" || pref === "dark" || pref === "auto" ? pref : "auto";
  localStorage.setItem("mockery_theme", validPref);
  const resolved =
    validPref === "auto"
      ? colorSchemeMql && colorSchemeMql.matches
        ? "light"
        : "dark"
      : validPref;
  document.documentElement.setAttribute("data-theme-pref", validPref);
  document.documentElement.setAttribute("data-theme", resolved);

  for (const btn of document.querySelectorAll("[data-theme-set]")) {
    const active = btn.getAttribute("data-theme-set") === validPref;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  }
}

function initTheme() {
  const initial = localStorage.getItem("mockery_theme") || localStorage.getItem("mock_helper_theme") || "auto";
  applyTheme(initial);
  if (colorSchemeMql && colorSchemeMql.addEventListener) {
    colorSchemeMql.addEventListener("change", () => {
      const saved = localStorage.getItem("mockery_theme") || localStorage.getItem("mock_helper_theme") || "auto";
      if (saved === "auto") {
        applyTheme("auto");
      }
    });
  }
  for (const btn of document.querySelectorAll("[data-theme-set]")) {
    btn.addEventListener("click", () => applyTheme(btn.getAttribute("data-theme-set")));
  }
}

/* ---------------- Core State Helpers ---------------- */

let _wsHadConnected = false;
let _wsLostConnection = false;

const wsMachine = createMachine({
  name: "ws_transport",
  initial: "connecting",
  transitions: WS_TRANSITIONS,
  registry: clientMachines,
  strict: false,
  onTransition: (next, prev) => {
    state.wsState = next;
    renderConnectionIndicator();
    if (next === "reconnecting" && prev === "connected") {
      _wsLostConnection = true;
      showToast("Lost the connection. Reconnecting…", 3000);
    } else if (next === "connected") {
      if (_wsHadConnected && _wsLostConnection) {
        showToast("Reconnected.", 2000);
      }
      _wsHadConnected = true;
      _wsLostConnection = false;
    }
  },
});

const captureMachine = createMachine({
  name: "copilot_capture",
  initial: "idle",
  transitions: CAPTURE_TRANSITIONS,
  registry: clientMachines,
  strict: true,
  onTransition: (next) => {
    state.captureState = next;
    renderCaptureControls();
    renderWorkspaceMode();
  },
});

let _lastModalTrigger = null;

function _modalRootFor(modalName) {
  if (modalName === "settings") return el("settingsModal");
  if (modalName === "history") return el("historyModal");
  if (modalName === "onboarding") return el("onboardingModal");
  if (modalName === "profile") return el("profileModal");
  return null;
}

function _focusableIn(root) {
  if (!root) return [];
  return [...root.querySelectorAll(
    'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
  )].filter((node) => node.offsetParent !== null && !node.closest(".hidden"));
}

const modalCtrl = createModalTokenController({
  initial: "none",
  registry: clientMachines,
  onChange: (next, prev) => {
    if (prev === "none" && next !== "none") {
      _lastModalTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    }
    state.modalState = next;
    el("settingsModal")?.classList.toggle("hidden", next !== "settings");
    el("historyModal")?.classList.toggle("hidden", next !== "history");
    el("onboardingModal")?.classList.toggle("hidden", next !== "onboarding");
    el("profileModal")?.classList.toggle("hidden", next !== "profile");
    if (next !== "none") {
      const root = _modalRootFor(next);
      const focusables = _focusableIn(root);
      if (focusables.length > 0 && root && !root.contains(document.activeElement)) {
        focusables[0].focus({ preventScroll: true });
      }
    } else if (prev !== "none" && _lastModalTrigger?.isConnected) {
      try {
        _lastModalTrigger.focus({ preventScroll: true });
      } catch {
        /* ignore */
      }
      _lastModalTrigger = null;
    }
    if (prev === "onboarding" && next !== "onboarding") {
      if (state.onboardStep === "packs") {
        try {
          window.localStorage?.setItem("mockery_onboard_packs_seen", "1");
        } catch {
          /* ignore */
        }
      }
      if (!practiceStudio?.isPracticeActive?.()) {
        practiceStudio?.stopSpeaking?.();
      }
    }
  },
});

function setWsState(next, event = null) {
  wsMachine.transition(next, { event });
}

function setCaptureState(next, event = null) {
  captureMachine.transition(next, { event });
}

function setModalState(next, opts = {}) {
  return modalCtrl.set(next, opts);
}

function closeAllPopovers() {
  const setupPop = el("sessionSetupPopover");
  const menuPop = el("transcriptMenuPopover");
  if (setupPop) setupPop.classList.add("hidden");
  if (menuPop) menuPop.classList.add("hidden");
  el("sessionContextBtn")?.setAttribute("aria-expanded", "false");
  el("transcriptMenuBtn")?.setAttribute("aria-expanded", "false");
}

function showToast(message, durationMs = 4000) {
  const banner = el("toastBanner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  if (state.toastTimer) clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => {
    banner.classList.add("hidden");
  }, durationMs);
}

function fmtClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

function fmtMs(ms) {
  return fmtClock(Math.floor((ms || 0) / 1000));
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function allowedEventsSet() {
  return new Set(state.session?.state_machines?.session?.allowed_events || []);
}

/* ---------------- WebSocket Transport State Machine ---------------- */

function saveOptimisticBackup() {
  const sm = state.session?.state_machines;
  if (!sm || state.optimisticBackup) return;
  state.optimisticBackup = {
    sessionId: state.session.id,
    phase: sm.session?.phase,
    allowed_events: Array.isArray(sm.session?.allowed_events) ? [...sm.session.allowed_events] : [],
    jobs: sm.jobs ? { ...sm.jobs } : {},
  };
}

function rollbackOptimisticState() {
  const b = state.optimisticBackup;
  state.optimisticBackup = null;
  if (!b || !state.session || state.session.id !== b.sessionId || !state.session.state_machines) {
    return;
  }
  const sm = state.session.state_machines;
  if (sm.session) {
    sm.session.phase = b.phase;
    sm.session.allowed_events = b.allowed_events;
  }
  if (sm.jobs) {
    sm.jobs = { ...sm.jobs, ...b.jobs };
  }
  renderPhaseAndControls();
  renderQAList();
  renderCalibrationPanel();
}

function connectWS() {
  if (state.wsReconnectTimer) {
    clearTimeout(state.wsReconnectTimer);
    state.wsReconnectTimer = null;
  }
  if (state.wsState === "reconnecting") {
    setWsState("connecting", "ws_reconnect");
  }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/session`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = () => {
    if (state.ws !== ws) return;
    if (state.wsReconnectTimer) {
      clearTimeout(state.wsReconnectTimer);
      state.wsReconnectTimer = null;
    }
    setWsState("connected", "ws_open");
    let flushedStart = false;
    while (state.outbox.length > 0 && ws.readyState === WebSocket.OPEN) {
      const raw = state.outbox.shift();
      try {
        const parsed = JSON.parse(raw);
        if (parsed?.type === "start_capture") flushedStart = true;
      } catch {
        /* ignore */
      }
      ws.send(raw);
    }
    if (state.captureState === "streaming" && !flushedStart) {
      sendCmd({ type: "start_capture" });
    }
  };

  ws.onmessage = (ev) => {
    if (state.ws !== ws) return;
    const msg = JSON.parse(ev.data);
    handleServerMessage(msg);
  };

  ws.onclose = () => {
    if (state.ws !== ws) return;
    state.pendingNewSession = false;
    rollbackOptimisticState();
    setWsState("reconnecting", "ws_close");
    if (state.wsReconnectTimer) clearTimeout(state.wsReconnectTimer);
    state.wsReconnectTimer = setTimeout(() => {
      state.wsReconnectTimer = null;
      connectWS();
    }, 1400);
  };
}

function sendCmd(payload) {
  const serialized = JSON.stringify(payload);
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(serialized);
  } else if (state.outbox.length < 32) {
    state.outbox.push(serialized);
  }
}

function sendBinaryPcm(channelId, pcmArrayBuffer) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const pcmBytes = new Uint8Array(pcmArrayBuffer);
  const packet = new Uint8Array(1 + pcmBytes.byteLength);
  packet[0] = channelId;
  packet.set(pcmBytes, 1);
  state.ws.send(packet.buffer);
}

/* ---------------- Server Event Handling ---------------- */

function handleServerMessage(msg) {
  if (msg.asr) {
    state.asr = msg.asr;
    renderAsrBadge();
  }
  if (msg.diarizer) {
    state.diarizer = msg.diarizer;
  }

  if (msg.type === "transition_rejected") {
    state.pendingNewSession = false;
    rollbackOptimisticState();
    showToast(`${msg.remediation || msg.message || "That isn't available right now."}`);
    return;
  }

  if (msg.type === "capture_error") {
    state.pendingNewSession = false;
    rollbackOptimisticState();
    if (state.captureState === "streaming" || state.captureState === "starting") {
      stopCaptureSession({ flush: false });
    }
    if ((state.session?.utterances || []).length === 0) {
      state.forceWorkspace = false;
      renderWorkspaceMode();
    }
    showToast(msg.message || "Capture failed to start.", 4500);
    return;
  }

  if (msg.type === "error") {
    state.pendingNewSession = false;
    rollbackOptimisticState();
    showToast(msg.message || "Something went wrong.");
    return;
  }

  if (msg.type === "manual_speaker") {
    state.manualSpeaker = msg.manual_speaker;
    renderSpeakerToggle();
    return;
  }

  if (msg.type === "dive_deeper_loading") {
    const qa = (state.session?.qa_groups_resolved || []).find((q) => q.id === msg.qa_id);
    if (qa) {
      qa.probe_state = "loading";
      renderQAList();
      renderProbesPanel();
    }
    return;
  }

  if (msg.type === "dive_deeper_result") {
    if (!state.selectedQaId) {
      state.selectedQaId = msg.result.qa_id;
    }
    if (state.selectedQaId === msg.result.qa_id) {
      activateRightTab("probes");
    }
    return;
  }

  if (msg.type === "regroup_loading") {
    if (state.session?.state_machines?.jobs) {
      saveOptimisticBackup();
      state.session.state_machines.jobs.regroup = "running";
      state.session.state_machines.session.allowed_events =
        state.session.state_machines.session.allowed_events.filter(
          (e) => !["force_new_card", "split_card", "merge_card", "regroup_cards"].includes(e)
        );
      renderPhaseAndControls();
      renderQAList();
    }
    return;
  }

  if (msg.type === "calibration_loading") {
    state.streamingCalibrationMd = "";
    if (state.calibrationStreamRaf) {
      cancelAnimationFrame(state.calibrationStreamRaf);
      state.calibrationStreamRaf = null;
    }
    if (state.session?.state_machines) {
      saveOptimisticBackup();
      state.session.state_machines.jobs.calibration = "running";
      state.session.state_machines.session.phase = "calibrating";
      state.editingSelfRating = false;
      renderPhaseAndControls();
      renderCalibrationPanel();
    }
    activateRightTab("calibration");
    return;
  }

  if (msg.type === "calibration_delta") {
    if (!state.session || (msg.session_id && msg.session_id !== state.session.id)) {
      return;
    }
    const delta = typeof msg.delta === "string" ? msg.delta : "";
    if (!delta) return;
    state.streamingCalibrationMd += delta;
    if (!state.calibrationStreamRaf) {
      state.calibrationStreamRaf = requestAnimationFrame(() => {
        state.calibrationStreamRaf = null;
        renderCalibrationPanel();
      });
    }
    return;
  }

  if (msg.type === "calibration_ready") {
    state.optimisticBackup = null;
    state.streamingCalibrationMd = "";
    if (state.calibrationStreamRaf) {
      cancelAnimationFrame(state.calibrationStreamRaf);
      state.calibrationStreamRaf = null;
    }
    if (state.session) {
      state.session.calibration_markdown = msg.markdown;
    }
    state.editingSelfRating = false;
    renderCalibrationPanel();
    return;
  }

  if (msg.type === "utterance_draft") {
    applyUtteranceDraftPatch(msg);
    return;
  }

  if (msg.type === "session_state") {
    state.optimisticBackup = null;
    const prevSelected = state.selectedQaId;
    const prevSessionId = state.session?.id;
    state.session = msg.session;
    if (prevSessionId && msg.session.id !== prevSessionId) {
      state.streamingCalibrationMd = "";
      if (state.calibrationStreamRaf) {
        cancelAnimationFrame(state.calibrationStreamRaf);
        state.calibrationStreamRaf = null;
      }
    }
    if (msg.manual_speaker) {
      state.manualSpeaker = msg.manual_speaker;
    }

    if (msg.session_ended && state.captureState === "stopping") {
      if (state.stopFallbackTimer) {
        clearTimeout(state.stopFallbackTimer);
        state.stopFallbackTimer = null;
      }
      setCaptureState("idle");
    }

    if (state.pendingNewSession && prevSessionId && msg.session.id !== prevSessionId) {
      state.pendingNewSession = false;
      state.askedProbes.clear();
      state.editingSelfRating = false;
      state.forceWorkspace = false;
    }

    if (state.captureState !== "streaming" && typeof msg.session.live_elapsed_ms === "number") {
      state.baseElapsedMs = msg.session.live_elapsed_ms;
      renderTimerPill(state.baseElapsedMs / 1000);
    }

    const qas = state.session.qa_groups_resolved || [];
    if (msg.opened_new_qa && msg.latest_qa_id) {
      state.selectedQaId = msg.latest_qa_id;
    } else if (!prevSelected || !qas.some((q) => q.id === prevSelected)) {
      state.selectedQaId = qas.length ? qas[qas.length - 1].id : null;
    }

    if (msg.session_ended && !state.pendingNewSession && qas.length > 0) {
      activateRightTab("calibration");
    }
    if (msg.regrouped) {
      showToast("Cards regrouped.", 2600);
    }
    if (msg.roles_auto_swapped) {
      const why = msg.roles_auto_swapped_rationale ? ` (${msg.roles_auto_swapped_rationale})` : "";
      showToast(`Speakers were the wrong way round, so they were swapped${why}.`, 5500);
    } else if (msg.roles_swapped) {
      showToast("Speakers swapped.", 2600);
    }

    syncConfigInputsFromSession();
    renderWorkspaceMode();
    renderPhaseAndControls();
    renderQAList();
    renderProbesPanel();
    renderCalibrationPanel();
    renderMetricsBar();
    renderSpeakerToggle();
  }
}

function applyUtteranceDraftPatch(msg) {
  const session = state.session;
  const utt = msg.utterance;
  if (!session || !utt || (msg.session_id && session.id !== msg.session_id)) {
    return;
  }

  let prevSpeaker = null;
  const allUtts = session.utterances || [];
  const uIdx = allUtts.findIndex((u) => u.id === utt.id);
  if (uIdx >= 0) {
    if (utt.rev < allUtts[uIdx].rev) return;
    prevSpeaker = allUtts[uIdx].speaker;
    allUtts[uIdx] = utt;
  } else {
    allUtts.push(utt);
  }

  let targetQa = null;
  let probeStateChanged = false;
  if (msg.qa_id) {
    targetQa = (session.qa_groups_resolved || []).find((q) => q.id === msg.qa_id) || null;
    if (targetQa) {
      const qUtts = targetQa.utterances || (targetQa.utterances = []);
      const qIdx = qUtts.findIndex((u) => u.id === utt.id);
      if (qIdx >= 0) {
        qUtts[qIdx] = utt;
      } else {
        qUtts.push(utt);
      }
      if (msg.probe_state && targetQa.probe_state !== msg.probe_state) {
        targetQa.probe_state = msg.probe_state;
        probeStateChanged = true;
      }
      if (typeof msg.has_speculative_cache === "boolean") {
        targetQa.has_speculative_cache = msg.has_speculative_cache;
      }
    }
  } else {
    const warm = session.warmup_resolved || (session.warmup_resolved = []);
    const wIdx = warm.findIndex((u) => u.id === utt.id);
    if (wIdx >= 0) {
      warm[wIdx] = utt;
    } else {
      warm.push(utt);
    }
  }

  const container = el("qaList");
  const speakerChanged = prevSpeaker !== null && prevSpeaker !== utt.speaker;
  const searchActive = Boolean(state.searchQuery.trim());
  const span = !speakerChanged && !searchActive
    ? container?.querySelector(`[data-utt-row="${CSS.escape(utt.id)}"]`)
    : null;

  if (msg.metrics) {
    session.metrics = msg.metrics;
    renderMetricsBar();
  }

  if (!span) {
    renderQAList();
    if (probeStateChanged) renderProbesPanel();
    return;
  }

  // O(1) DOM patch: update only the live draft text span without rebuilding any card
  const wasNearBottom =
    container.scrollHeight - container.scrollTop - container.clientHeight < 64;
  span.innerHTML = dimFillersHtml(String(utt.text || "").trim());

  // Invalidate only this card's cached render signature so the next full reconciliation stays exact
  const cardEl = span.closest(".qa-card");
  if (cardEl) {
    cardEl._bodySig = "";
    cardEl._uttSig = "";
    if (probeStateChanged && targetQa) {
      const pillHost = cardEl.querySelector(".qa-probe-pill-host");
      if (pillHost) {
        pillHost.innerHTML = renderCardProbePill(targetQa.probe_state || "idle");
      }
    }
  }

  if (wasNearBottom) {
    container.scrollTop = container.scrollHeight;
  }
  updateJumpToBottomVisibility();
  if (probeStateChanged) {
    renderProbesPanel();
  }
}

/* ---------------- Shared-Clock Web Audio Capture ---------------- */

async function ensureSharedAudioContext() {
  if (!state.sharedAudioCtx || state.sharedAudioCtx.state === "closed") {
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule("/static/pcm-worklet.js?v=perf4");
    state.sharedAudioCtx = ctx;
  }
  if (state.sharedAudioCtx.state === "suspended") {
    await state.sharedAudioCtx.resume();
  }
  return state.sharedAudioCtx;
}

async function startCapture() {
  if (state.captureState !== "idle") return;
  if (state.practiceActive || practiceStudio?.isPracticeStartingOrActive?.()) {
    showToast("Exit Practice Mode before starting live recording.", 3000);
    return;
  }

  const mode = el("modeSelect").value;
  pushSessionConfig();
  closeAllPopovers();
  const prevForceWorkspace = state.forceWorkspace;
  state.forceWorkspace = true;
  setCaptureState("requesting_permissions", "start_capture");

  try {
    let streamsToAttach;
    if (mode === "remote_meet") {
      streamsToAttach = await acquireRemoteMeetStreams();
    } else {
      streamsToAttach = await acquireSingleMicStreams();
    }

    sendCmd({ type: "start_capture" });
    state.localSegmentStartedAt = Date.now();
    setCaptureState("streaming", "permissions_granted");

    for (const item of streamsToAttach) {
      attachStreamToSharedContext(item.ctx, item.stream, item.channelId, item.meterId);
    }

    if (state.timerHandle) clearInterval(state.timerHandle);
    state.timerHandle = setInterval(updateTimer, 500);
  } catch (err) {
    stopAudioStreamsOnly();
    state.forceWorkspace = prevForceWorkspace;
    setCaptureState("idle", "permissions_failed");
    sendCmd({ type: "cancel_capture" });
    showToast(`Could not start recording: ${err.message || err}`, 5500);
  }
}

async function acquireRemoteMeetStreams() {
  let tabStream;
  try {
    tabStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: {
        suppressLocalAudioPlayback: false,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    const detail =
      err && err.name === "NotAllowedError"
        ? "no tab was chosen."
        : err?.message || String(err);
    throw new Error(`the Meet tab was not shared — ${detail}`);
  }

  tabStream.getVideoTracks().forEach((t) => t.stop());

  const tabAudioTracks = tabStream.getAudioTracks();
  if (tabAudioTracks.length === 0) {
    tabStream.getTracks().forEach((t) => t.stop());
    throw new Error(
      "the tab was shared without sound. Pick your Meet tab again and turn on “Also share tab audio”."
    );
  }

  let micStream;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
  } catch (err) {
    tabStream.getTracks().forEach((t) => t.stop());
    throw err;
  }

  let sharedCtx;
  try {
    sharedCtx = await ensureSharedAudioContext();
  } catch (err) {
    tabStream.getTracks().forEach((t) => t.stop());
    micStream.getTracks().forEach((t) => t.stop());
    throw err;
  }

  tabAudioTracks[0].onended = () => {
    if (state.captureState === "streaming") {
      stopCaptureSession();
    }
  };
  for (const micTrack of micStream.getAudioTracks()) {
    micTrack.onended = () => {
      if (state.captureState === "streaming") {
        showToast("Microphone disconnected — saving captured audio.", 3500);
        stopCaptureSession();
      }
    };
  }

  return [
    { ctx: sharedCtx, stream: micStream, channelId: CHANNEL_INTERVIEWER, meterId: "meterYou" },
    { ctx: sharedCtx, stream: tabStream, channelId: CHANNEL_CANDIDATE, meterId: "meterCand" },
  ];
}

async function acquireSingleMicStreams() {
  const micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
  try {
    const sharedCtx = await ensureSharedAudioContext();
    for (const micTrack of micStream.getAudioTracks()) {
      micTrack.onended = () => {
        if (state.captureState === "streaming") {
          showToast("Microphone disconnected — saving captured audio.", 3500);
          stopCaptureSession();
        }
      };
    }
    return [{ ctx: sharedCtx, stream: micStream, channelId: CHANNEL_ROOM_MIC, meterId: "meterYou" }];
  } catch (err) {
    micStream.getTracks().forEach((t) => t.stop());
    throw err;
  }
}

function attachStreamToSharedContext(ctx, mediaStream, channelId, meterElementId) {
  const source = ctx.createMediaStreamSource(mediaStream);
  const worklet = new AudioWorkletNode(ctx, "pcm16-downsampler", {
    processorOptions: { targetSampleRate: 16000, chunkSamples: 1600 },
  });

  const silentGain = ctx.createGain();
  silentGain.gain.value = 0;
  source.connect(worklet);
  worklet.connect(silentGain);
  silentGain.connect(ctx.destination);

  let flushResolve = null;
  let detached = false;

  worklet.port.onmessage = (ev) => {
    const data = ev.data || {};
    if (data.pcmBuffer && !detached) {
      sendBinaryPcm(channelId, data.pcmBuffer);
      const pct = Math.min(100, Math.round(Math.pow((data.rms || 0) * 6.5, 0.65) * 100));
      const meterEl = el(meterElementId);
      if (meterEl) meterEl.style.width = `${pct}%`;
    }
    if (data.flushed && flushResolve) {
      const done = flushResolve;
      flushResolve = null;
      done();
    }
  };

  state.audioFlushFns.push(
    () =>
      new Promise((resolve) => {
        if (detached) {
          resolve();
          return;
        }
        const timer = setTimeout(() => {
          flushResolve = null;
          resolve();
        }, 150);
        flushResolve = () => {
          clearTimeout(timer);
          resolve();
        };
        try {
          worklet.port.postMessage({ type: "flush" });
        } catch (_) {
          clearTimeout(timer);
          flushResolve = null;
          resolve();
        }
      })
  );

  state.audioCleanupFns.push(() => {
    detached = true;
    worklet.port.onmessage = null;
    if (flushResolve) {
      const done = flushResolve;
      flushResolve = null;
      done();
    }
    worklet.disconnect();
    source.disconnect();
    silentGain.disconnect();
    mediaStream.getTracks().forEach((t) => t.stop());
  });
}

function stopAudioStreamsOnly() {
  state.audioFlushFns.length = 0;
  while (state.audioCleanupFns.length) {
    const fn = state.audioCleanupFns.pop();
    try {
      fn();
    } catch (_) {}
  }
  if (state.sharedAudioCtx && state.sharedAudioCtx.state !== "closed") {
    state.sharedAudioCtx.close().catch(() => {});
    state.sharedAudioCtx = null;
  }
  el("meterYou").style.width = "0%";
  el("meterCand").style.width = "0%";
}

async function stopCaptureSession({ flush = true } = {}) {
  if (state.captureState !== "streaming") return;
  setCaptureState("stopping");

  if (state.localSegmentStartedAt) {
    state.baseElapsedMs += Math.max(0, Date.now() - state.localSegmentStartedAt);
    state.localSegmentStartedAt = null;
  }
  if (state.timerHandle) {
    clearInterval(state.timerHandle);
    state.timerHandle = null;
  }

  if (flush && state.audioFlushFns.length) {
    const flushers = state.audioFlushFns.splice(0, state.audioFlushFns.length);
    await Promise.all(flushers.map((fn) => fn().catch(() => {})));
  }

  stopAudioStreamsOnly();
  if (flush) {
    sendCmd({ type: "flush_and_end" });
    if (state.stopFallbackTimer) clearTimeout(state.stopFallbackTimer);
    state.stopFallbackTimer = setTimeout(() => {
      state.stopFallbackTimer = null;
      if (state.captureState === "stopping") {
        setCaptureState("idle");
      }
    }, 3500);
  } else {
    setCaptureState("idle");
  }
}

function updateTimer() {
  let totalMs = state.baseElapsedMs;
  if (state.captureState === "streaming" && state.localSegmentStartedAt) {
    totalMs += Math.max(0, Date.now() - state.localSegmentStartedAt);
  }
  renderTimerPill(totalMs / 1000);
}

function isSessionCompleted() {
  return Boolean(state.session?.calibration_markdown) && state.captureState === "idle";
}

const INTERVIEW_LIMIT_SEC = 45 * 60;

function renderTimerPill(elapsedSec) {
  const pill = el("timerPill");
  el("timerText").textContent = fmtClock(elapsedSec);
  const pct = Math.min(100, Math.max(0, (elapsedSec / INTERVIEW_LIMIT_SEC) * 100));
  el("timerProgress").style.width = `${pct.toFixed(1)}%`;
  const completed = isSessionCompleted();
  // Calm until it matters: amber for the last five minutes, muted red for the last one and beyond.
  const remaining = INTERVIEW_LIMIT_SEC - Math.floor(elapsedSec);
  const over = !completed && remaining <= 60;
  const warn = !completed && !over && remaining <= 5 * 60;
  pill.classList.toggle("completed", completed);
  pill.classList.toggle("warn", warn);
  pill.classList.toggle("over", over);
  pill.title =
    over && remaining <= 0
      ? "Past the 45-minute mark"
      : over
      ? "Less than a minute left of 45"
      : warn
      ? "Less than five minutes left of 45"
      : "Elapsed of a 45-minute interview";
}

/* ---------------- Workspace Mode (Launchpad vs Active Split Studio) ---------------- */

function renderWorkspaceMode() {
  const qas = state.session?.qa_groups_resolved || [];
  const warmup = state.session?.warmup_resolved || [];
  const hasContent = qas.length > 0 || warmup.length > 0;
  const nextRoute = deriveAppRoute({
    practiceActive: state.practiceActive,
    captureState: state.captureState,
    forceWorkspace: state.forceWorkspace,
    hasCopilotContent: hasContent,
  });
  if (nextRoute !== state.route) {
    clientMachines.recordTransition({
      machine: "app_route",
      from: state.route,
      to: nextRoute,
      event: "render_workspace_mode",
      reason: null,
    });
    state.route = nextRoute;
  }

  if (nextRoute === "practice") {
    el("appShell").classList.remove("setup-mode");
    el("setupLaunchpad").classList.add("hidden");
    el("workspaceGrid").classList.add("hidden");
    state.homeVisible = false;
    return;
  }

  const showLaunchpad = nextRoute === "launchpad";

  el("appShell").classList.toggle("setup-mode", showLaunchpad);
  el("setupLaunchpad").classList.toggle("hidden", !showLaunchpad);
  el("workspaceGrid").classList.toggle("hidden", showLaunchpad);

  // Start Live Capture is live from first paint; it only goes busy while a capture request is in flight
  const setupStartBtn = el("setupStartBtn");
  if (setupStartBtn) {
    setupStartBtn.disabled = state.captureState !== "idle";
  }
  const setupHint = el("setupStartHint");
  if (setupHint) {
    setupHint.textContent = START_HINTS[el("modeSelect").value] || START_HINTS.remote_meet;
  }
  if (showLaunchpad && !state.homeVisible) {
    refreshHomeSessions();
    practiceStudio?.refreshPracticeHistory();
  }
  state.homeVisible = showLaunchpad;
  syncTopbarTelemetryVisibility();
}

function syncTopbarTelemetryVisibility() {
  const m = state.session?.metrics;
  const hasSpeech = Boolean(m && (m.interviewer_words > 0 || m.candidate_words > 0));
  const isStreaming = state.captureState === "streaming";
  el("liveMetersGroup")?.classList.toggle("hidden", !isStreaming);
  el("liveTelemetryCluster")?.classList.toggle("hidden", !(isStreaming || hasSpeech));
}

function renderConnectionIndicator() {
  const dot = el("brandDot");
  dot.title = `WebSocket: ${state.wsState}`;
  dot.classList.toggle("disconnected", state.wsState !== "connected");
}

function renderCaptureControls() {
  const startBtn = el("startCaptureBtn");
  const stopBtn = el("stopCaptureBtn");
  const startLabel = el("startCaptureLabel");
  const stopLabel = el("stopCaptureLabel");
  const dot = el("brandDot");
  const statusEl = el("captureStatusPill");
  const statusText = el("captureStatusText");
  const cs = state.captureState;
  const phase = state.session?.state_machines?.session?.phase || "ready";
  const hasCalib = Boolean(state.session?.calibration_markdown);
  const completed = hasCalib && cs === "idle";
  // Once a recording has been stopped it reads "Stopped", and the interview can still be picked back up.
  const isResumable =
    phase === "awaiting_self_rating" || phase === "rated" || phase === "calibrated";

  dot.classList.toggle("recording", cs === "streaming");
  el("modeSelect").disabled = cs !== "idle";

  // Live-only transcript controls disappear once the session is complete
  el("newCardBtn")?.classList.toggle("hidden", completed);

  if (statusEl && statusText) {
    let tone = "idle";
    let label = "Not recording";
    if (cs === "streaming") {
      tone = "rec";
      label = "Recording";
    } else if (cs === "stopping") {
      tone = "busy";
      label = "Saving…";
    } else if (cs === "requesting_permissions") {
      tone = "busy";
      label = "Waiting for permission…";
    } else if (phase === "calibrating") {
      tone = "busy";
      label = "Stopped";
    } else if (isResumable) {
      tone = "ended";
      label = "Stopped";
    }
    statusEl.className = `capture-state ${tone}`;
    statusText.textContent = label;
  }

  renderTimerPill(state.baseElapsedMs / 1000);

  if (cs === "streaming") {
    startBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = false;
    stopLabel.textContent = "End interview";
  } else if (cs === "requesting_permissions") {
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    startBtn.disabled = true;
    startBtn.classList.add("btn-primary");
    startBtn.classList.remove("btn-outline");
    startLabel.textContent = "Starting…";
  } else if (cs === "stopping") {
    startBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = true;
    stopLabel.textContent = "Saving…";
  } else {
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    startBtn.disabled = phase === "calibrating";
    startLabel.textContent = isResumable ? "Resume interview" : "Start recording";
    // Starting is the one solid-ink action; resuming an ended interview is a quiet outline so the
    // transcript and audit lead.
    startBtn.classList.toggle("btn-primary", !isResumable);
    startBtn.classList.toggle("btn-outline", isResumable);
  }
}

function renderPhaseAndControls() {
  const sm = state.session?.state_machines;
  if (!sm) return;

  const phase = sm.session?.phase || "ready";
  const allowed = allowedEventsSet();
  const blocked = sm.session?.blocked_reasons || {};
  const regroupRunning = sm.jobs?.regroup === "running";

  const phaseBadge = el("phaseBadge");
  phaseBadge.textContent = PHASE_LABELS[phase] || phase;

  const regroupBtn = el("regroupBtn");
  regroupBtn.disabled = !allowed.has("regroup_cards");
  regroupBtn.querySelector("span").textContent = regroupRunning ? "Regrouping…" : "Regroup cards";
  regroupBtn.title = blocked.regroup_cards || "Re-draw card boundaries from the whole transcript";

  const newCardBtn = el("newCardBtn");
  newCardBtn.disabled = !allowed.has("force_new_card");

  const hasCalib = Boolean(state.session?.calibration_markdown);
  el("calibReadyDot").classList.toggle("hidden", !hasCalib);

  renderCaptureControls();
}

function renderAsrBadge() {
  // The speech model only earns screen space when it is not ready.
  const badge = el("asrBadge");
  const textEl = el("asrBadgeText");
  if (!badge || !textEl) return;
  const st = state.asr.state;
  const failed = st === "error" || st === "failed";
  badge.classList.toggle("hidden", !(st === "loading" || st === "switching" || failed));
  badge.classList.toggle("error", failed);
  badge.removeAttribute("title");
  if (st === "loading") {
    textEl.textContent = "Preparing offline speech engine (downloads once on first launch)…";
  } else if (st === "switching") {
    textEl.textContent = "Switching speech model…";
  } else if (failed) {
    textEl.textContent = "The speech model failed to load.";
    badge.title = state.asr.error || "";
  }
}

function diarizerEnrolledCount() {
  const raw = state.diarizer?.enrolled_speakers;
  if (Array.isArray(raw)) return raw.length;
  return Number(raw || 0);
}

function isAutoDiarizerActive() {
  const autoDiarize = Boolean(state.session?.config?.auto_diarize ?? true);
  const diarizerReady = Boolean(state.diarizer?.ready ?? state.diarizer?.embedder_ready);
  return autoDiarize && diarizerReady && !state.diarizer?.manual_override;
}

function renderSpeakerToggle() {
  const mode = el("modeSelect").value;
  const btn = el("speakerToggleBtn");
  const swapBtn = el("swapAllSpeakersBtn");
  const candMeter = el("candMeterWrap");
  const tabKbd = el("speakerTabKbd");
  const hasUtts = (state.session?.utterances || []).length > 0;
  const completed = isSessionCompleted();

  if (mode === "in_person" && !completed) {
    btn.classList.remove("hidden");
    candMeter.classList.add("hidden");
    swapBtn.classList.toggle("hidden", !hasUtts);

    const dot = el("diarizerModeDot");
    const label = el("manualSpeakerLabel");
    if (isAutoDiarizerActive()) {
      const enrolled = diarizerEnrolledCount();
      label.textContent = enrolled >= 2 ? "Auto (2)" : "Auto";
      dot.classList.remove("you");
      tabKbd.classList.add("hidden");
    } else {
      const isYou = state.manualSpeaker === "interviewer";
      label.textContent = isYou ? "You" : "Cand";
      dot.classList.toggle("you", isYou);
      tabKbd.classList.remove("hidden");
    }
  } else {
    btn.classList.add("hidden");
    candMeter.classList.toggle("hidden", mode === "in_person");
    swapBtn.classList.add("hidden");
  }
}

function updateContextSummaryChip() {
  const role = el("roleInput").value.trim() || "Untitled interview";
  const typeVal = el("typeSelect").value || "behavioral";
  el("contextRoleSummary").textContent = role;
  el("contextTypeSummary").textContent = TYPE_SHORT_LABELS[typeVal] || typeVal;

  if (!practiceStudio?.isHomePracticeMode?.()) {
    for (const btn of document.querySelectorAll("#setupTypePills [data-setup-type]")) {
      btn.classList.toggle("active", btn.getAttribute("data-setup-type") === typeVal);
    }
  }
  const modeVal = el("modeSelect").value || "remote_meet";
  for (const btn of document.querySelectorAll("#setupModePills [data-setup-mode]")) {
    btn.classList.toggle("active", btn.getAttribute("data-setup-mode") === modeVal);
  }
  const modeNote = el("setupModeDesc");
  if (modeNote) modeNote.textContent = MODE_NOTES[modeVal] || "";
}

function syncConfigInputsFromSession() {
  if (!state.session) return;
  const cfg = state.session.config;
  if (document.activeElement !== el("roleInput")) {
    el("roleInput").value = cfg.role_title;
  }
  if (!practiceStudio?.isHomePracticeMode?.() && document.activeElement !== el("setupRoleInput")) {
    el("setupRoleInput").value = cfg.role_title;
  }
  if (document.activeElement !== el("typeSelect")) {
    el("typeSelect").value = cfg.interview_type;
  }
  if (document.activeElement !== el("rubricInput")) {
    el("rubricInput").value = cfg.rubric_notes || "";
  }
  if (document.activeElement !== el("setupRubricInput")) {
    el("setupRubricInput").value = cfg.rubric_notes || "";
  }

  const wavVal = Boolean(cfg.save_raw_audio);
  el("saveAudioCheckbox").checked = wavVal;
  el("setupSaveAudioProxy").checked = wavVal;

  if (state.captureState === "idle" && document.activeElement !== el("modeSelect")) {
    el("modeSelect").value = cfg.mode;
  }

  updateContextSummaryChip();

  const isPhase = cfg.interview_type === "system_design" || cfg.interview_type === "coding";
  el("leftPaneTitle").textContent = isPhase ? "Phases" : "Q&A";

  const sessionChanged = state.lastSyncedSessionId !== state.session.id;
  state.lastSyncedSessionId = state.session.id;

  if (state.session.self_rating) {
    const sr = state.session.self_rating;
    if (document.activeElement !== el("selfHireSignal")) {
      setHireSignalValue(sr.hire_signal || "lean_hire");
    }
    if (document.activeElement !== el("selfLevelInput")) {
      el("selfLevelInput").value = sr.perceived_level || "";
    }
    if (document.activeElement !== el("selfRationaleInput")) {
      el("selfRationaleInput").value = sr.gut_rationale || "";
    }
    if (document.activeElement !== el("selfConcernInput")) {
      el("selfConcernInput").value = sr.biggest_concern || "";
    }
  } else if (sessionChanged) {
    setHireSignalValue("lean_hire");
    el("selfLevelInput").value = "";
    el("selfRationaleInput").value = "";
    el("selfConcernInput").value = "";
  }
}

function setHireSignalValue(val) {
  el("selfHireSignal").value = val;
  for (const btn of document.querySelectorAll("#hireSignalPills [data-signal]")) {
    btn.classList.toggle("active", btn.getAttribute("data-signal") === val);
  }
}

function renderMetricsBar() {
  const m = state.session?.metrics;
  const youEl = el("ratioYouText");
  const candEl = el("ratioCandText");
  if (!m || (m.interviewer_words === 0 && m.candidate_words === 0)) {
    if (youEl) youEl.textContent = "You —";
    if (candEl) candEl.textContent = "Candidate —";
    syncTopbarTelemetryVisibility();
    return;
  }
  const youPct = Math.round(m.interviewer_time_pct);
  const candPct = Math.round(m.candidate_time_pct);
  if (youEl) youEl.textContent = `You ${youPct}%`;
  if (candEl) candEl.textContent = `Candidate ${candPct}%`;
  syncTopbarTelemetryVisibility();
}

function renderCardProbePill(probeState) {
  if (probeState === "ready_cached") return "follow-ups ready";
  if (probeState === "warming" || probeState === "loading") return "thinking…";
  if (probeState === "stale_cached" || probeState === "revealed_stale") return "new answers";
  return "";
}

function renderPendingReviewsHtml(uttId) {
  const reviews = (state.session?.pending_reviews || []).filter((r) => r.utterance_id === uttId);
  if (!reviews.length) return "";
  return reviews
    .map((r) => {
      let label = "Suggestion";
      if (r.kind === "flip_speaker") {
        const target = r.suggested_speaker === "interviewer" ? "You" : "Candidate";
        label = `This sounds like ${target === "You" ? "you" : "the candidate"}`;
      } else if (r.kind === "split_mixed_turn") {
        const segCount = (r.suggested_segments || []).length || 2;
        label = `${segCount} speakers in one turn — split it`;
      } else if (r.kind === "split_card") {
        label = r.suggested_title ? `Start a new card here: “${r.suggested_title}”` : "Start a new card here";
      }
      const rationale = r.rationale ? ` — ${r.rationale}` : "";
      return `
        <div class="review-chip" data-review-chip="${escapeHtml(r.id)}">
          <span class="review-chip-text">
            <strong>${escapeHtml(label)}</strong>${escapeHtml(rationale)}
          </span>
          <span class="review-chip-actions">
            <button type="button" class="link-quiet is-strong" data-action="accept-review" data-review="${escapeHtml(r.id)}">Accept</button>
            <button type="button" class="link-quiet" data-action="dismiss-review" data-review="${escapeHtml(r.id)}">Dismiss</button>
          </span>
        </div>
      `;
    })
    .join("");
}

function highlightSearchHtml(rawText, query) {
  const safe = escapeHtml(rawText);
  if (!query) return safe;
  const re = new RegExp(`(${escapeRegExp(escapeHtml(query))})`, "gi");
  return safe.replace(re, "<mark>$1</mark>");
}

const BACKCHANNEL_RE = /^(um+|uh+|hmm+|mhm+|yeah|yep|ok(ay)?|right|sure|got it)[.,…]*$/i;

function isBackchannelFiller(text) {
  const t = String(text || "").trim();
  return t.length <= 10 && BACKCHANNEL_RE.test(t);
}

// Hesitation sounds get one treatment wherever they fall — opening a turn, mid-sentence, or as a
// turn of their own — so a transcript never shows one "um" dimmed and the next one not.
// Real words that double as backchannels ("yeah", "right") are only dimmed as a whole turn.
const FILLER_WORD_RE =
  /(^|[^\p{L}\p{N}'’-])((?:u+h+m*|u+m+|h+m+|mhm+|erm+|uh-huh|mm-hmm)(?![\p{L}\p{N}'’-])[,.…]*)/giu;

function dimFillersHtml(rawText) {
  const text = String(rawText || "");
  let out = "";
  let last = 0;
  for (const m of text.matchAll(FILLER_WORD_RE)) {
    const start = m.index + m[1].length;
    out += `${escapeHtml(text.slice(last, start))}<span class="utt-filler">${escapeHtml(m[2])}</span>`;
    last = start + m[2].length;
  }
  return out + escapeHtml(text.slice(last));
}

function renderUtteranceRowsHtml(utts, { qaId = null, canSplitAny = false } = {}) {
  const q = state.searchQuery.trim();
  const filtered = q
    ? utts.filter((u) => (u.text || "").toLowerCase().includes(q.toLowerCase()))
    : utts;

  if (!filtered.length) {
    return q
      ? `<div class="utt-empty-note">Nothing here matches.</div>`
      : `<div class="utt-empty-note">Listening…</div>`;
  }

  // Group consecutive turns by the same speaker into cohesive Granola/Metaview prose blocks
  // while keeping every utterance individually addressable via [data-utt-row] for citations & edits.
  const blocks = [];
  for (let i = 0; i < filtered.length; i++) {
    const u = filtered[i];
    const hasReviews = (state.session?.pending_reviews || []).some((r) => r.utterance_id === u.id);
    const prevBlock = blocks[blocks.length - 1];
    if (
      prevBlock &&
      prevBlock.speaker === u.speaker &&
      !q &&
      !hasReviews &&
      !prevBlock.hasReviews &&
      u.final &&
      prevBlock.allFinal
    ) {
      prevBlock.items.push({ u, idx: i });
    } else {
      blocks.push({
        speaker: u.speaker,
        hasReviews,
        allFinal: Boolean(u.final),
        items: [{ u, idx: i }],
      });
    }
  }

  return blocks
    .map((block) => {
      const firstItem = block.items[0];
      const firstUtt = firstItem.u;
      const isYou = block.speaker === "interviewer";
      const canSplit = qaId && firstItem.idx > 0 && canSplitAny && !q;

      // Break long monologues into breathable ~65-word paragraphs or at >= 10s pauses
      const paragraphs = [];
      let curPara = [];
      let curWords = 0;
      let prevEndMs = null;

      for (const { u } of block.items) {
        const text = String(u.text || "").trim();
        const isFiller = isBackchannelFiller(text);
        const words = text ? text.split(/\s+/).length : 0;
        const pauseMs = prevEndMs != null && u.start_ms != null ? u.start_ms - prevEndMs : 0;
        if (curPara.length > 0 && !isFiller && (pauseMs >= 10000 || curWords >= 65)) {
          paragraphs.push(curPara);
          curPara = [];
          curWords = 0;
        }
        curPara.push({ u, text, isFiller });
        curWords += words;
        prevEndMs = u.end_ms || u.start_ms;
      }
      if (curPara.length > 0) {
        paragraphs.push(curPara);
      }

      const parasHtml = paragraphs
        .map((para, pIdx) => {
          const paraStartTs = fmtMs(para[0].u.start_ms);
          const spansHtml = para
            .map(({ u, text, isFiller }, sIdx) => {
              const isSelectedCite = state.activeCitationUttId === u.id;
              const fillerCls = isFiller && para.length > 1 ? " utt-filler" : "";
              const citeCls = isSelectedCite ? " cite-selected" : "";
              const draftCls = u.final ? "" : " draft-utt";
              // While searching, the marker highlight is the only inline treatment.
              const body = q ? highlightSearchHtml(text, q) : dimFillersHtml(text);
              const space = sIdx < para.length - 1 ? " " : "";
              return `<span class="utt-seg${fillerCls}${citeCls}${draftCls}" data-utt-row="${escapeHtml(u.id)}" title="${fmtMs(u.start_ms)}">${body}</span>${space}`;
            })
            .join("");

          // Continuation turns carry their timestamp in the fixed left gutter (never inline mid-paragraph);
          // the first paragraph's time already sits in the same gutter column on the block header row.
          const gutterHtml =
            pIdx > 0
              ? `<span class="utt-ts" title="Continued at ${paraStartTs}">${paraStartTs}</span>`
              : `<span class="utt-ts" aria-hidden="true"></span>`;
          return `<div class="utt-row${pIdx > 0 ? " utt-row-cont" : ""}">${gutterHtml}<p class="utt-para">${spansHtml}</p></div>`;
        })
        .join("");

      const reviewsHtml = block.items.map(({ u }) => renderPendingReviewsHtml(u.id)).join("");

      return `
        <div class="utterance-block ${block.speaker}">
          <div class="utt-row utt-row-head">
            <span class="utt-ts utt-ts-head">${fmtMs(firstUtt.start_ms)}</span>
            <div class="utt-meta">
              <div class="utt-meta-left">
                <button
                  type="button"
                  class="role-flip-btn ${isYou ? "role-you" : "role-cand"}"
                  data-action="flip-speaker"
                  data-utt="${escapeHtml(firstUtt.id)}"
                  title="Wrong speaker? Click to switch."
                >
                  ${isYou ? "You" : "Candidate"}
                </button>
                ${firstUtt.final ? "" : `<span class="draft-pill">live</span>`}
              </div>
              <div class="utt-actions">
                ${
                  canSplit
                    ? `<button type="button" class="link-quiet" data-action="split" data-qa="${escapeHtml(qaId)}" data-utt="${escapeHtml(firstUtt.id)}" title="Start a new card from this turn">New card from here</button>`
                    : ""
                }
              </div>
            </div>
          </div>
          <div class="utt-prose">${parasHtml}</div>
          ${reviewsHtml ? `<div class="utt-reviews">${reviewsHtml}</div>` : ""}
        </div>
      `;
    })
    .join("");
}

const PHASE_STOPWORDS = new Set([
  "the", "and", "for", "of", "with", "to", "in", "a", "an", "on", "at", "by",
  "phase", "question", "topic", "discussion", "overview", "system", "design",
  "interview", "deep", "dive", "follow", "up", "final", "initial", "general",
]);

function shortPhaseKeyword(title) {
  const clean = String(title || "")
    .replace(/^(?:P|Q)\d+\s*[:—–-]\s*/i, "")
    .replace(/[&/]+/g, " ")
    .trim();
  if (!clean) return "";
  const words = clean
    .split(/\s+/)
    .map((w) => w.replace(/[^A-Za-z0-9-]/g, ""))
    .filter(Boolean);
  if (!words.length) return "";
  const meaningful = words.filter((w) => !PHASE_STOPWORDS.has(w.toLowerCase()) && w.length >= 3);
  let pick = meaningful.length >= 2 ? meaningful[meaningful.length - 1] : meaningful[0] || words[0];
  if (clean.toLowerCase().includes("wrap")) pick = "Wrap-up";
  else if (clean.toLowerCase().includes("intro") && clean.toLowerCase().includes("scop")) pick = "Scope";
  else if (clean.toLowerCase().includes("fan-out") || clean.toLowerCase().includes("messaging")) pick = "Messaging";
  else if (clean.toLowerCase().includes("schema")) pick = "Schema";
  else if (clean.toLowerCase().includes("scale") || clean.toLowerCase().includes("scaling")) pick = "Scale";
  else if (clean.toLowerCase().includes("sync")) pick = "Sync";
  else if (clean.toLowerCase().includes("api")) pick = "API";
  if (pick.length > 10) pick = `${pick.slice(0, 9)}…`;
  return pick;
}

// The calibration phase table mirrors whichever phase is open in the transcript.
function syncPhaseTableActive() {
  const table = el("phaseSignalsStrip");
  if (!table) return;
  const qas = state.session?.qa_groups_resolved || [];
  const openBadge = String(qas.find((q) => q.id === state.selectedQaId)?.badge || "").toUpperCase();
  for (const row of table.querySelectorAll("[data-jump-badge]")) {
    const isOpen = Boolean(openBadge) && row.getAttribute("data-jump-badge") === openBadge;
    row.classList.toggle("active", isOpen);
    if (isOpen) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
  }
}

function renderPhaseNavStrip(qas) {
  const strip = el("phaseNavStrip");
  syncPhaseTableActive();
  if (!qas.length) {
    if (strip._navSig !== "") {
      strip.innerHTML = "";
      strip._navSig = "";
    }
    return;
  }
  const navSig = qas
    .map((qa) => `${qa.id}|${qa.badge}|${qa.title}|${qa.id === state.selectedQaId ? 1 : 0}`)
    .join("||");
  if (strip._navSig === navSig) {
    return;
  }
  strip._navSig = navSig;
  strip.innerHTML = qas
    .map((qa) => {
      const kw = shortPhaseKeyword(qa.title);
      return `
      <button
        type="button"
        class="phase-tab ${qa.id === state.selectedQaId ? "active" : ""}"
        data-jump-qa="${escapeHtml(qa.id)}"
        title="${escapeHtml(qa.badge)}: ${escapeHtml(qa.title)}"
      ><span class="phase-tab-code">${escapeHtml(qa.badge)}</span>${kw ? `<span>${escapeHtml(kw)}</span>` : ""}</button>
    `;
    })
    .join("");
}

const _cardHtmlTemplate = document.createElement("template");

function _buildCardElement(html) {
  _cardHtmlTemplate.innerHTML = html.trim();
  return _cardHtmlTemplate.content.firstElementChild;
}

function _computeUtterancesSig(utts, { q, activeCite, canSplitAny, reviewsByUttId }) {
  let parts = `${q}|${canSplitAny ? 1 : 0}|${utts.length}`;
  for (let i = 0; i < utts.length; i++) {
    const u = utts[i];
    const citeMark = activeCite && activeCite === u.id ? "!" : "";
    const revList = reviewsByUttId ? reviewsByUttId.get(u.id) : null;
    const revMark = revList ? `#${revList.join(",")}` : "";
    parts += `\n${u.id}:${u.rev}:${u.final ? 1 : 0}:${u.speaker[0]}:${u.start_ms}:${citeMark}${revMark}:${u.text || ""}`;
  }
  return parts;
}

function renderQAList() {
  const container = el("qaList");
  const session = state.session;
  if (!session) return;

  const qas = session.qa_groups_resolved || [];
  const warmup = session.warmup_resolved || [];
  const allowed = allowedEventsSet();
  const canSplitAny = allowed.has("split_card");
  const canMergeAny = allowed.has("merge_card");
  const completed = isSessionCompleted();
  const q = state.searchQuery.trim();
  const activeCite = state.activeCitationUttId || "";

  let reviewsByUttId = null;
  const pendingReviews = session.pending_reviews;
  if (pendingReviews && pendingReviews.length > 0) {
    reviewsByUttId = new Map();
    for (const r of pendingReviews) {
      let arr = reviewsByUttId.get(r.utterance_id);
      if (!arr) {
        arr = [];
        reviewsByUttId.set(r.utterance_id, arr);
      }
      arr.push(r.id);
    }
  }

  renderPhaseNavStrip(qas);

  const activeEl = document.activeElement;
  if (
    activeEl &&
    container.contains(activeEl) &&
    (activeEl.classList.contains("qa-title-input") || activeEl.classList.contains("qa-note-input"))
  ) {
    state.pendingQARerender = true;
    const warmCard = container.querySelector('[data-warmup-card="true"]');
    if (warmCard && warmup.length > 0) {
      const wSig = _computeUtterancesSig(warmup, { q, activeCite, canSplitAny: false, reviewsByUttId });
      if (warmCard._uttSig !== wSig) {
        const warmList = warmCard.querySelector('[data-warmup-list="true"]');
        if (warmList) warmList.innerHTML = renderUtteranceRowsHtml(warmup);
        warmCard._uttSig = wSig;
        warmCard._bodySig = "";
      }
    }
    for (const qa of qas) {
      const cardEl = container.querySelector(`[data-qa-card="${CSS.escape(qa.id)}"]`);
      if (!cardEl) continue;
      const uSig = _computeUtterancesSig(qa.utterances || [], {
        q,
        activeCite,
        canSplitAny,
        reviewsByUttId,
      });
      if (cardEl._uttSig !== uSig) {
        const uttListEl = cardEl.querySelector(".utterance-list");
        if (uttListEl) {
          uttListEl.innerHTML = renderUtteranceRowsHtml(qa.utterances || [], {
            qaId: qa.id,
            canSplitAny,
          });
        }
        cardEl._uttSig = uSig;
        cardEl._bodySig = "";
      }
      const pState = qa.probe_state || "idle";
      if (cardEl._probeState !== pState) {
        const pillHost = cardEl.querySelector(".qa-probe-pill-host");
        if (pillHost) pillHost.innerHTML = renderCardProbePill(pState);
        cardEl._probeState = pState;
        cardEl._bodySig = "";
      }
    }
    return;
  }

  state.pendingQARerender = false;
  const prevScrollTop = container.scrollTop;
  const hadContent = container.querySelector(".qa-card") !== null;
  const wasNearBottom =
    !hadContent || container.scrollHeight - container.scrollTop - container.clientHeight < 64;

  if (qas.length === 0 && warmup.length === 0) {
    const emptySig = `${state.captureState === "streaming" ? 1 : 0}|${state.simulatorOpen ? 1 : 0}`;
    if (container._emptySig !== emptySig || hadContent) {
      container.innerHTML = `
        <div class="quiet-state">
          <p class="quiet-title">${state.captureState === "streaming" ? "Listening…" : "Nothing said yet."}</p>
          <p>${
            state.simulatorOpen
              ? "Type a turn below and the transcript will build itself into cards."
              : "The transcript appears here as you talk, grouped into cards as the interview moves along."
          }</p>
        </div>
      `;
      container._emptySig = emptySig;
    }
    updateJumpToBottomVisibility();
    return;
  }

  container._emptySig = "";

  // Index existing card DOM nodes so untouched cards survive with zero rebuilds
  let existingWarmup = null;
  const existingById = new Map();
  for (const child of Array.from(container.children)) {
    if (child.hasAttribute("data-warmup-card")) {
      existingWarmup = child;
    } else if (child.hasAttribute("data-qa-card")) {
      existingById.set(child.getAttribute("data-qa-card"), child);
    } else {
      child.remove();
    }
  }

  const desiredNodes = [];
  let domMutated = false;

  if (warmup.length > 0) {
    const warmUttSig = _computeUtterancesSig(warmup, {
      q,
      activeCite,
      canSplitAny: false,
      reviewsByUttId,
    });
    const warmBodySig = `warm|${warmup.length}|${warmUttSig}`;
    let warmEl = existingWarmup;
    if (!warmEl || warmEl._bodySig !== warmBodySig) {
      const nextWarm = _buildCardElement(`
        <article class="qa-card warmup-card" data-warmup-card="true">
          <div class="qa-card-header">
            <span class="qa-badge"></span>
            <span class="qa-title-static">Before the first question</span>
            <div class="qa-card-actions"><span>${warmup.length} turn${warmup.length === 1 ? "" : "s"}</span></div>
          </div>
          <div class="utterance-list" data-warmup-list="true">${renderUtteranceRowsHtml(warmup)}</div>
        </article>
      `);
      nextWarm._bodySig = warmBodySig;
      nextWarm._uttSig = warmUttSig;
      if (warmEl) {
        warmEl.replaceWith(nextWarm);
      }
      warmEl = nextWarm;
      domMutated = true;
    }
    desiredNodes.push(warmEl);
  }

  for (let i = 0; i < qas.length; i++) {
    const qa = qas[i];
    const isSelected = qa.id === state.selectedQaId;
    const pState = qa.probe_state || "idle";
    const utts = qa.utterances || [];
    const canMerge = i > 0 && canMergeAny && !completed;
    const uttSig = _computeUtterancesSig(utts, {
      q,
      activeCite,
      canSplitAny,
      reviewsByUttId,
    });
    const bodySig = `${qa.badge}|${qa.title}|${qa.notes || ""}|${pState}|${canMerge ? 1 : 0}|${uttSig}`;

    let cardEl = existingById.get(qa.id);
    if (cardEl && cardEl._bodySig === bodySig) {
      cardEl.classList.toggle("selected", isSelected);
    } else {
      const nextCard = _buildCardElement(`
        <article class="qa-card ${isSelected ? "selected" : ""}" data-qa-card="${escapeHtml(qa.id)}">
          <div class="qa-card-header">
            <span class="qa-badge">${escapeHtml(qa.badge)}</span>
            <input
              class="qa-title-input"
              data-qa-title="${escapeHtml(qa.id)}"
              value="${escapeHtml(qa.title)}"
              title="Click to rename"
              spellcheck="false"
            />
            <div class="qa-card-actions">
              ${
                canMerge
                  ? `<button type="button" class="link-quiet qa-merge-btn" data-action="merge" data-qa="${escapeHtml(qa.id)}" title="Merge this card into the one above">Merge up</button>`
                  : ""
              }
              <span class="qa-probe-pill-host qa-probe-state">${renderCardProbePill(pState)}</span>
              <span>${utts.length} turn${utts.length === 1 ? "" : "s"}</span>
            </div>
          </div>
          <div class="utterance-list">
            ${renderUtteranceRowsHtml(utts, { qaId: qa.id, canSplitAny })}
          </div>
          <div class="qa-card-footer">
            <input
              class="qa-note-input"
              data-qa-note="${escapeHtml(qa.id)}"
              placeholder="Add a private note…"
              value="${escapeHtml(qa.notes || "")}"
            />
          </div>
        </article>
      `);
      nextCard._bodySig = bodySig;
      nextCard._uttSig = uttSig;
      nextCard._probeState = pState;
      if (cardEl) {
        cardEl.replaceWith(nextCard);
      }
      cardEl = nextCard;
      domMutated = true;
    }
    desiredNodes.push(cardEl);
  }

  // Remove stale cards no longer in session and reconcile order without detaching unchanged nodes
  const desiredSet = new Set(desiredNodes);
  for (const child of Array.from(container.children)) {
    if (!desiredSet.has(child)) {
      child.remove();
      domMutated = true;
    }
  }
  for (let i = 0; i < desiredNodes.length; i++) {
    const node = desiredNodes[i];
    const current = container.children[i];
    if (current !== node) {
      container.insertBefore(node, current || null);
      domMutated = true;
    }
  }

  if (domMutated) {
    if (wasNearBottom) {
      container.scrollTop = container.scrollHeight;
    } else {
      container.scrollTop = prevScrollTop;
    }
  }
  updateJumpToBottomVisibility();
}

function updateJumpToBottomVisibility() {
  const container = el("qaList");
  const btn = el("jumpToBottomBtn");
  if (!container || !btn) return;
  if (isSessionCompleted()) {
    btn.classList.add("hidden");
    return;
  }
  const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
  const hasCards = container.querySelector(".qa-card") !== null;
  btn.classList.toggle("hidden", !hasCards || distFromBottom < 140);
}

/* ---------------- Follow-ups (the teleprompter) ---------------- */

const ICON_COPY = `<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><rect x="5.5" y="5.5" width="7.75" height="7.75" rx="1.75" stroke="currentColor" stroke-width="1.4"/><path d="M10.5 5.5V4.25A1.5 1.5 0 009 2.75H4.25a1.5 1.5 0 00-1.5 1.5V9a1.5 1.5 0 001.5 1.5H5.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>`;
const ICON_CHECK = `<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3.5 8.4l3 3 6-6.6" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

function sentenceCase(str) {
  const t = String(str || "").trim();
  if (!t) return "";
  // Keep acronyms (API, SQL) intact; lower-case ordinary Title Case words after the first.
  return t
    .split(/\s+/)
    .map((w, i) => (i === 0 || /^[A-Z0-9&/+-]{2,}$/.test(w) ? w : w.toLowerCase()))
    .join(" ");
}

// A follow-up keeps its number for life: 1–3 always mean the same question, wherever it sits on
// screen, so the key that marked it asked is also the key that brings it back.
function probeKeyForNumber(num) {
  return (
    el("probesContent")
      .querySelector(`[data-probe-num="${Number(num)}"]`)
      ?.getAttribute("data-probe-asked") || null
  );
}

function toggleProbeAsked(key) {
  if (!key) return;
  const hadFocus = document.activeElement?.getAttribute?.("data-probe-asked") === key;
  if (state.askedProbes.has(key)) {
    state.askedProbes.delete(key);
  } else {
    state.askedProbes.add(key);
  }
  renderProbesPanel();
  if (hadFocus) {
    el("probesContent").querySelector(`[data-probe-asked="${CSS.escape(key)}"]`)?.focus();
  }
}

function renderProbesPanel() {
  const qas = state.session?.qa_groups_resolved || [];
  const qa = qas.find((q) => q.id === state.selectedQaId) || qas[qas.length - 1] || null;

  const diveBtn = el("diveDeeperBtn");
  const label = el("selectedQaLabel");
  const stateEl = el("probeReadinessBadge");
  const content = el("probesContent");

  if (qa) {
    state.selectedQaId = qa.id;
  }

  let probesSig;
  if (!qa) {
    probesSig = "none";
  } else {
    const pState = qa.probe_state || "idle";
    const history = qa.dive_deeper_history || [];
    const latest = history.length > 0 ? history[history.length - 1] : null;
    let askedBits = "";
    if (latest && latest.probes) {
      for (let idx = 0; idx < latest.probes.length; idx++) {
        askedBits += state.askedProbes.has(`${qa.id}:${history.length}:${idx}`) ? "1" : "0";
      }
    }
    probesSig = `${qa.id}|${qa.badge}|${qa.title}|${pState}|${history.length}|${askedBits}|${latest?.quick_read || ""}|${latest?.model_used || ""}`;
  }

  if (content._probesSig === probesSig) {
    return;
  }
  content._probesSig = probesSig;

  const setDiveBtn = (text, { disabled = false, hidden = false } = {}) => {
    diveBtn.disabled = disabled;
    diveBtn.classList.toggle("hidden", hidden);
    diveBtn.innerHTML = `<span id="diveDeeperLabel">${escapeHtml(text)}</span> <kbd>D</kbd>`;
  };
  const setState = (text, ready = false) => {
    stateEl.textContent = text;
    stateEl.className = `probes-context-state${text ? "" : " hidden"}${ready ? " is-ready" : ""}`;
  };

  if (!qa) {
    label.textContent = "";
    label.removeAttribute("title");
    setState("");
    setDiveBtn("Ask now", { disabled: true, hidden: true });
    content.innerHTML = `
      <div class="quiet-state">
        <p class="quiet-title">Follow-ups will appear here.</p>
        <p>When the candidate gives a real answer, three follow-up questions are written in the background. Press <kbd>D</kbd> to see them — even mid-sentence.</p>
      </div>
    `;
    return;
  }

  const pState = qa.probe_state || "idle";
  const history = qa.dive_deeper_history || [];
  const hasProbes = history.length > 0;

  // One muted line names what these follow-ups are about; the transcript header carries the rest.
  label.textContent = `${qa.badge} · ${qa.title}`;
  label.title = qa.title;

  if (pState === "loading") {
    setState("");
    setDiveBtn("Thinking…", { disabled: true });
    content.innerHTML = `<p class="probe-thinking"><span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span><span>Thinking of follow-ups…</span></p>`;
    return;
  }

  if (!hasProbes) {
    let title = "No follow-ups yet.";
    let body = `Press <kbd>D</kbd> to get three, grounded in what the candidate has said in ${escapeHtml(qa.badge)}.`;
    let action = "";
    if (pState === "ready_cached") {
      title = "Three follow-ups are ready.";
      body = `Press <kbd>D</kbd> to see them.`;
      action = `<button type="button" class="btn btn-primary" data-trigger-dive="true">Show follow-ups</button>`;
    } else if (pState === "warming") {
      title = "Writing follow-ups…";
      body = `They are prepared in the background while the candidate talks.`;
    } else if (pState === "stale_cached") {
      title = "The candidate has said more.";
      body = `Press <kbd>D</kbd> for follow-ups on the latest answer.`;
    } else if (pState === "empty") {
      title = "Nothing to follow up on yet.";
      body = `Follow-ups appear once the candidate has answered.`;
    }
    setState("");
    setDiveBtn("Ask now", { disabled: pState === "empty", hidden: pState === "empty" });
    content.innerHTML = `
      <div class="quiet-state">
        <p class="quiet-title">${title}</p>
        <p>${body}</p>
        ${action}
      </div>
    `;
    return;
  }

  const latest = history[history.length - 1];
  const isStale = pState === "revealed_stale" || pState === "stale_cached";
  setState(pState === "ready_cached" ? "new ones ready" : pState === "warming" ? "writing new ones…" : "", pState === "ready_cached");
  setDiveBtn(pState === "ready_cached" ? "Show new" : "New angles");

  // Unasked questions keep their order at the top of the pane; asked ones sink below them and fold
  // down to a single checked line, so the next thing to ask always sits on the eye-line.
  const probes = (latest.probes || []).map((p, idx) => {
    const key = `${qa.id}:${history.length}:${idx}`;
    return { p, num: idx + 1, key, isAsked: state.askedProbes.has(key) };
  });
  const ordered = [...probes.filter((r) => !r.isAsked), ...probes.filter((r) => r.isAsked)];

  const rowsHtml = ordered
    .map(({ p, num, key, isAsked }) => {
      const hotkey = num <= 9 ? ` or press ${num}` : "";
      if (isAsked) {
        return `
        <li
          class="probe asked"
          data-probe-asked="${escapeHtml(key)}"
          data-probe-num="${num}"
          role="button"
          tabindex="0"
          aria-pressed="true"
          title="Asked — click${hotkey} to bring it back"
        >
          <span class="probe-num">${ICON_CHECK}</span>
          <div class="probe-body">
            <p class="probe-question">${escapeHtml(p.question)}</p>
          </div>
        </li>
      `;
      }
      const angle = sentenceCase(p.angle);
      return `
        <li
          class="probe"
          data-probe-asked="${escapeHtml(key)}"
          data-probe-num="${num}"
          role="button"
          tabindex="0"
          aria-pressed="false"
          title="Click${hotkey} once you've asked it"
        >
          <span class="probe-num">${num}</span>
          <div class="probe-body">
            ${angle ? `<p class="probe-angle">${escapeHtml(angle)}</p>` : ""}
            <p class="probe-question">${escapeHtml(p.question)}</p>
            ${p.why ? `<p class="probe-why">${escapeHtml(p.why)}</p>` : ""}
          </div>
          <button type="button" class="probe-copy" data-probe-copy="${escapeHtml(p.question)}" title="Copy question" aria-label="Copy question">${ICON_COPY}</button>
        </li>
      `;
    })
    .join("");

  // The header button already offers new angles; this line only teaches the number keys.
  const keysHtml = probes
    .slice(0, 9)
    .map((r) => `<kbd>${r.num}</kbd>`)
    .join(" ");
  const allAsked = probes.length > 0 && probes.every((r) => r.isAsked);
  const hintHtml = allAsked
    ? `All asked. ${keysHtml} bring one back.`
    : `${keysHtml} mark as asked`;

  content.innerHTML = `
    <p class="probe-read">${escapeHtml(latest.quick_read)}</p>
    ${isStale ? `<p class="probe-stale">The candidate has said more since these were written.</p>` : ""}
    <ol class="probe-list" title="Written by ${escapeHtml(latest.model_used || "the default model")}">${rowsHtml}</ol>
    ${probes.length ? `<p class="probe-hint">${hintHtml}</p>` : ""}
  `;
}

/* ---------------- Metaview-Grade Calibration Parser, Scorecard & Markdown ---------------- */

const CALIB_SCALE = [
  { key: "strong_no_hire", short: "Strong No" },
  { key: "lean_no_hire", short: "Lean No" },
  { key: "lean_hire", short: "Lean Hire" },
  { key: "strong_hire", short: "Strong Hire" },
];

// "evidence supports **No Hire (or Lean No Hire) for Senior**" -> captures the bolded lead recommendation
const VERDICT_LEAD_RE =
  /(?:supports?|recommends?|recommendation(?:\s+is)?|verdict\s*(?:is|:)?|points?\s+to|indicates?|suggests?|warrants?|lands?\s+(?:at|on)|reads?\s+as)\s*:?\s*(?:(?:an?|the|your)\s+)?(?:(?:clear|firm|solid|borderline)\s+)?\*\*([^*]+)\*\*/i;

const VERDICT_PHRASE_RE =
  /\b(strong(?:ly)?[\s_-]+no(?:[\s_-]+hire)?|lean[\s_-]+no(?:[\s_-]+hire)?|no[\s_-]+hire|strong[\s_-]+hire|lean[\s_-]+hire|hire)\b/gi;

function classifyVerdictPhrase(phrase) {
  const p = String(phrase).toLowerCase().replace(/[\s_-]+/g, " ").trim();
  if (p.startsWith("strong") && p.includes(" no")) return { key: "strong_no_hire", label: "Strong No" };
  if (p.startsWith("lean no")) return { key: "lean_no_hire", label: "Lean No" };
  // An unqualified "No Hire" is a firm no: it sits below "Lean No" on the 4-step scale and keeps its own label.
  if (p === "no hire") return { key: "strong_no_hire", label: "No Hire" };
  if (p === "strong hire") return { key: "strong_hire", label: "Strong Hire" };
  return { key: "lean_hire", label: p === "hire" ? "Hire" : "Lean Hire" };
}

// Returns every verdict phrase in reading order (first = primary lead recommendation, rest = alternatives).
function findVerdictPhrases(source, { allowBareHire = true } = {}) {
  const found = [];
  for (const m of String(source || "").matchAll(VERDICT_PHRASE_RE)) {
    if (!allowBareHire && m[1].toLowerCase() === "hire") continue;
    found.push({ ...classifyVerdictPhrase(m[1]), index: m.index });
  }
  return found;
}

function sectionOneText(text) {
  const firstHeading = text.search(/^#{1,4}\s+/m);
  const from = firstHeading >= 0 ? firstHeading + 1 : 0;
  const nextHeading = text.slice(from).search(/\n#{1,4}\s+/);
  return nextHeading >= 0 ? text.slice(0, from + nextHeading) : text.slice(0, 2400);
}

function parsePhaseSignalLine(line) {
  const m = String(line).trim().match(/^(?:[-*]\s+\*\*|#{2,5}\s+\*{0,2})((?:P|Q)\d+)\s*:\s*(.+)$/);
  if (!m) return null;
  // Drop bold markers and any trailing stats parenthetical, then split on the LAST dash so hyphenated
  // titles ("Wrap-Up", "Fan-out") stay intact.
  const body = m[2].replace(/\*\*/g, "").replace(/\s*\(`.*$/, "").trim();
  let cut = Math.max(body.lastIndexOf("—"), body.lastIndexOf("–"));
  let sepLen = 1;
  if (cut < 0) {
    cut = body.lastIndexOf(" - ");
    sepLen = 3;
  }
  if (cut <= 0) return null;
  const title = body.slice(0, cut).trim();
  const rating = body
    .slice(cut + sepLen)
    .replace(/\s*signal\s*$/i, "")
    .split("/")[0]
    .replace(/[:.]+$/, "")
    .trim();
  if (!title || !rating || rating.length > 24) return null;
  const rl = rating.toLowerCase();
  const tone =
    rl.includes("strong") || rl.includes("positive") || rl.includes("solid")
      ? "strong"
      : rl.includes("weak") || /\bno\b/.test(rl) || rl.includes("red") || rl.includes("concern")
      ? "weak"
      : "mixed";
  return { badge: m[1].toUpperCase(), title, rating, tone };
}

function parseCalibrationInsights(md) {
  const text = String(md || "");
  const lines = text.replace(/\r\n/g, "\n").split("\n");

  // 1. Extract section headings for quick jump nav
  const allHeadings = [];
  for (const line of lines) {
    const m = line.trim().match(/^(#{1,4})\s+(.+)$/);
    if (m) {
      const rawTitle = m[2].replace(/\*\*/g, "").trim();
      const shortMatch = rawTitle.match(/^(\d+\.)\s*([^:–—(]+)/);
      let shortLabel = shortMatch ? shortMatch[2].trim() : rawTitle;
      shortLabel = shortLabel
        .replace(/\bCalibration Check\b/i, "Calibration")
        .replace(/\bInterviewer Probing\b/i, "Probing")
        .replace(/\bConstructive Takeaways\b/i, "Takeaways");
      if (shortLabel.length > 20) {
        // Keep the first two words, but never end a tab on a dangling connector ("Probing &").
        const words = shortLabel.split(/\s+/).slice(0, 2);
        if (words.length > 1 && /^(?:&|and|for|of|the|vs\.?|to|in|on)$/i.test(words[1])) words.pop();
        shortLabel = words.join(" ");
      }
      allHeadings.push({
        id: `audit-sec-${allHeadings.length + 1}`,
        level: m[1].length,
        title: rawTitle,
        short: shortLabel,
      });
    }
  }
  // Pinned tabs list only the top-level audit sections (per-phase sub-headings stay in the document flow)
  const topLevel = allHeadings.reduce((min, h) => Math.min(min, h.level), 9);
  const sections = allHeadings.filter((h) => h.level === topLevel);

  // 2. Extract the AI Evidence Verdict from Section 1 (first 950 chars).
  //    The chip must mirror the *primary* (lead) recommendation of the first sentence, so the earliest
  //    verdict phrase wins and softer parenthetical alternatives ("No Hire (or Lean No Hire)") are kept
  //    only as a secondary note. Scanning for "lean no" anywhere used to downgrade that lead to Lean No.
  const intro = text.slice(0, 950);
  let aiKey = null;
  let aiLabel = "See Audit";
  let aiDetail = "";
  let aiAlt = "";
  let aiLead = "";

  const verdictMatch = intro.match(VERDICT_LEAD_RE);
  let verdicts = [];
  if (verdictMatch) {
    verdicts = findVerdictPhrases(verdictMatch[1], { allowBareHire: true });
    aiLead = cleanAuditLine(verdictMatch[0]).replace(/[*`]/g, "").replace(/\s+/g, " ").trim();
  }
  if (!verdicts.length) {
    // No bolded lead verdict: drop every echo of the interviewer's own blind call (raw `lean_hire` enum
    // tokens and "Your blind rating of **…**" clauses) so it is never mistaken for the AI read, then take
    // the first explicit verdict phrase that remains.
    const withoutUserCall = intro
      .replace(/`(?:strong|lean)_(?:no_)?hire`/gi, " ")
      .replace(
        /\b(?:your|the\s+interviewer'?s?)\s+(?:blind\s+|gut\s+|initial\s+|pre-calibration\s+)*(?:rating|call|signal|read|verdict)\b[^.\n]*?\*\*[^*]+\*\*/gi,
        " "
      );
    verdicts = findVerdictPhrases(withoutUserCall, { allowBareHire: false });
    if (!verdicts.length) {
      // Only when the lead paragraph endorses the blind call (and never disputes it) does the AI read
      // equal the interviewer's own rating; otherwise stay neutral rather than invent an agreement.
      const leadPara = intro.split("\n").map((l) => l.trim()).find((l) => l && !l.startsWith("#")) || "";
      const disputes =
        /\btoo\s+(?:generous|harsh|strict|lenient|high|low)\b|unsupported|not\s+supported|cannot\s+support|inflated|over-?rated|under-?rated/i.test(
          leadPara
        );
      const endorses =
        /\b(?:well[-\s]calibrated|accurate|spot[-\s]on|aligned|aligns|matches|agrees?|captures|consistent\s+with|defensible)\b/i.test(
          intro
        );
      if (endorses && !disputes) {
        verdicts = findVerdictPhrases(intro, { allowBareHire: false }).slice(0, 1);
      }
    }
  }
  if (verdicts.length) {
    aiKey = verdicts[0].key;
    aiLabel = verdicts[0].label;
    const alt = verdicts.find((v) => v.label !== aiLabel);
    if (verdictMatch && alt) aiAlt = alt.label;
  } else if (/insufficient\s+signal|\bno[-\s]read\b|inconclusive/i.test(sectionOneText(text))) {
    aiLabel = "No Read";
    aiDetail = "Insufficient signal to rate";
  }

  const downlevelMatch = intro.match(
    /down-?level(?:ing|ed)?\s+(?:the\s+candidate\s+)?to\s+(?:an?\s+)?([A-Za-z][\w/-]*(?:\s+(?:[A-Z][\w/-]*|\d+))?)/
  );
  if (downlevelMatch && aiKey) {
    const lvl = downlevelMatch[1].trim();
    aiDetail = `Down-level → ${lvl.charAt(0).toUpperCase()}${lvl.slice(1)}`;
  } else if (aiKey && verdictMatch && /for\s+(Senior|Staff|Mid|Junior|L\d)/i.test(verdictMatch[1])) {
    const barM = verdictMatch[1].match(/for\s+(Senior|Staff|Mid|Junior|L\d)/i);
    aiDetail = `For ${barM[1]} bar`;
  }

  // 3. Extract Per-Phase Signals from Section 2 (`* **P1: Intro — Mixed Signal**` or `#### P1: Intro — **Mixed**`)
  const phaseSignals = [];
  const seenBadges = new Set();
  for (const line of lines) {
    const sig = parsePhaseSignalLine(line);
    if (sig && !seenBadges.has(sig.badge)) {
      seenBadges.add(sig.badge);
      phaseSignals.push(sig);
    }
  }

  return { sections, aiKey, aiLabel, aiDetail, aiAlt, aiLead, phaseSignals };
}

function cleanAuditLine(raw) {
  // Gracefully strip raw empty-metadata parentheticals like "(with `Perceived Level: Unspecified`, `Initial Rationale: None`, and `Biggest Concern: None`)"
  return String(raw || "").replace(
    /\s*\(with\s+`?Perceived Level:\s*Unspecified`?,\s*`?Initial Rationale:\s*None`?,?\s*and\s+`?Biggest Concern:\s*None`?\)/gi,
    ""
  );
}

function formatInlineMarkdown(raw) {
  let s = escapeHtml(cleanAuditLine(raw));
  const tokens = [];
  const stash = (html) => {
    const idx = tokens.length;
    tokens.push(html);
    return `\u0000TOK${idx}\u0000`;
  };

  // 1. Handle backticked spans cleanly:
  //    - `[MM:SS, utt_id]` -> clickable citation chip
  //    - `"quoted speech..."` -> pure italic editorial quote (ZERO jagged multi-line box!)
  //    - raw enum codes (`lean_hire`, `system_design`) -> clean readable text (not noisy chips)
  //    - short technical identifiers (`Redis`, `channels`) -> subtle inline <code>
  s = s.replace(/`([^`]+)`/g, (_, inner) => {
    const trimmed = inner.trim();
    const citeMatch = trimmed.match(/^\[(\d{1,2}:\d{2}),\s*(utt_[A-Za-z0-9_-]+)\]$/);
    if (citeMatch) {
      const isAct = state.activeCitationUttId === citeMatch[2] ? " active" : "";
      return stash(
        `<button type="button" class="cite-link${isAct}" data-cite-utt="${citeMatch[2]}" title="Show this moment in the transcript">${citeMatch[1]}</button>`
      );
    }

    const quoteMatch = trimmed.match(/^&quot;([\s\S]+)&quot;$/);
    if (quoteMatch) {
      return stash(`<q class="evidence-quote">${quoteMatch[1]}</q>`);
    }

    const enumHuman = HUMANIZED_ENUMS[trimmed];
    if (enumHuman) {
      return stash(`<strong>${enumHuman}</strong>`);
    }

    if (/^(Perceived Level|Initial Rationale|Biggest Concern):/i.test(trimmed)) {
      return stash(trimmed);
    }

    return stash(`<code>${inner}</code>`);
  });

  // 2. Also handle bare (un-backticked) [MM:SS, utt_id] citations
  s = s.replace(/\[(\d{1,2}:\d{2}),\s*(utt_[A-Za-z0-9_-]+)\]/g, (_, ts, uttId) => {
    const isAct = state.activeCitationUttId === uttId ? " active" : "";
    return stash(
      `<button type="button" class="cite-link${isAct}" data-cite-utt="${uttId}" title="Show this moment in the transcript">${ts}</button>`
    );
  });

  // Unwrap redundant outer parentheses around stashed citation chips
  s = s.replace(/\(\s*(\u0000TOK\d+\u0000(?:\s*\u0000TOK\d+\u0000)*)\s*\)/g, "$1");

  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  s = s.replace(/\u0000TOK(\d+)\u0000/g, (_, idx) => tokens[Number(idx)] || "");
  return s;
}

function parseTableRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

function isTableSeparator(line) {
  const cells = parseTableRow(line);
  return cells.length >= 2 && cells.every((c) => /^:?-{2,}:?$/.test(c));
}

function renderMarkdownSafe(md) {
  const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let inUl = false;
  let inOl = false;
  let para = [];
  let secIdx = 0;

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${para.map(formatInlineMarkdown).join("<br>")}</p>`);
      para = [];
    }
  };
  const closeLists = () => {
    if (inUl) {
      out.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      out.push("</ol>");
      inOl = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushPara();
      closeLists();
      continue;
    }

    if (trimmed.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1].trim())) {
      flushPara();
      closeLists();
      const headers = parseTableRow(trimmed);
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().includes("|") && lines[i].trim() !== "") {
        bodyRows.push(parseTableRow(lines[i].trim()));
        i++;
      }
      i--;
      const thead = `<thead><tr>${headers.map((h) => `<th>${formatInlineMarkdown(h)}</th>`).join("")}</tr></thead>`;
      const tbody = `<tbody>${bodyRows
        .map((r) => `<tr>${r.map((c) => `<td>${formatInlineMarkdown(c)}</td>`).join("")}</tr>`)
        .join("")}</tbody>`;
      out.push(`<div class="md-table-wrap"><table class="md-table">${thead}${tbody}</table></div>`);
      continue;
    }

    const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      flushPara();
      closeLists();
      secIdx += 1;
      const level = Math.min(4, Math.max(2, hMatch[1].length));
      out.push(`<h${level} id="audit-sec-${secIdx}">${formatInlineMarkdown(hMatch[2])}</h${level}>`);
      continue;
    }
    if (/^[-*_]{3,}$/.test(trimmed)) {
      flushPara();
      closeLists();
      out.push(`<hr />`);
      continue;
    }
    const bqMatch = trimmed.match(/^>\s?(.*)$/);
    if (bqMatch) {
      flushPara();
      closeLists();
      out.push(`<blockquote>${formatInlineMarkdown(bqMatch[1])}</blockquote>`);
      continue;
    }

    // Top-level semantic category header bullets (e.g. `* **Where your gut was spot-on:**`) -> crisp visual subhead!
    const topSubheadMatch = rawLine.match(/^[-*]\s+\*\*([^*]+:?)\*\*\s*$/);
    if (topSubheadMatch) {
      flushPara();
      closeLists();
      const label = topSubheadMatch[1].replace(/:$/, "").trim();
      const phaseSig = parsePhaseSignalLine(rawLine);
      out.push(
        phaseSig
          ? `<div class="audit-subhead">${escapeHtml(phaseSig.badge)} · ${escapeHtml(phaseSig.title)}<span class="audit-subhead-rating ${phaseSig.tone}">${escapeHtml(phaseSig.rating)}</span></div>`
          : `<div class="audit-subhead">${escapeHtml(label)}</div>`
      );
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (ulMatch) {
      flushPara();
      if (inOl) {
        out.push("</ol>");
        inOl = false;
      }
      if (!inUl) {
        out.push("<ul>");
        inUl = true;
      }
      out.push(`<li>${formatInlineMarkdown(ulMatch[1])}</li>`);
      continue;
    }
    const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (olMatch) {
      flushPara();
      if (inUl) {
        out.push("</ul>");
        inUl = false;
      }
      if (!inOl) {
        out.push("<ol>");
        inOl = true;
      }
      out.push(`<li>${formatInlineMarkdown(olMatch[1])}</li>`);
      continue;
    }
    closeLists();
    para.push(trimmed);
  }
  flushPara();
  closeLists();
  return out.join("\n");
}

function jumpToUtteranceCitation(uttId, triggerBtn = null) {
  if (!uttId) return;
  const container = el("qaList");
  const row = container.querySelector(`[data-utt-row="${CSS.escape(uttId)}"]`);
  if (!row) {
    if (state.searchQuery) {
      state.searchQuery = "";
      el("transcriptSearchInput").value = "";
      renderQAList();
      return jumpToUtteranceCitation(uttId, triggerBtn);
    }
    showToast("That moment is no longer in the transcript.");
    return;
  }

  state.activeCitationUttId = uttId;

  // Keep active state on clicked citation chip
  for (const chip of el("calibrationOutput").querySelectorAll(".cite-link.active")) {
    chip.classList.remove("active");
  }
  if (triggerBtn) {
    triggerBtn.classList.add("active");
  } else {
    el("calibrationOutput")
      .querySelector(`[data-cite-utt="${CSS.escape(uttId)}"]`)
      ?.classList.add("active");
  }

  // The cited words get a single highlighter stroke — the timestamp already lives in the citation
  // that was clicked, so it is never printed a second time inside the transcript.
  for (const prevSeg of container.querySelectorAll("[data-utt-row].cite-selected")) {
    prevSeg.classList.remove("cite-selected");
  }
  row.classList.add("cite-selected");

  const card = row.closest("[data-qa-card]");
  if (card) {
    const qaId = card.getAttribute("data-qa-card");
    if (qaId && state.selectedQaId !== qaId) {
      state.selectedQaId = qaId;
      for (const c of container.querySelectorAll(".qa-card.selected")) {
        c.classList.remove("selected");
      }
      card.classList.add("selected");
      renderPhaseNavStrip(state.session?.qa_groups_resolved || []);
      renderProbesPanel();
    }
  }

  const containerRect = container.getBoundingClientRect();
  const rowRect = row.getBoundingClientRect();
  const rowTopInContainer = rowRect.top - containerRect.top + container.scrollTop;
  let targetTop = rowTopInContainer - Math.round((container.clientHeight - rowRect.height) * 0.42);

  if (card) {
    const header = card.querySelector(".qa-card-header");
    const headerH = header ? header.offsetHeight : 44;
    const cutoff = targetTop + headerH;
    const cardBlocks = card.querySelectorAll(".utterance-block");
    for (const blk of cardBlocks) {
      const br = blk.getBoundingClientRect();
      const bTop = br.top - containerRect.top + container.scrollTop;
      const bBot = bTop + br.height;
      if (bTop < cutoff + 6 && bBot > cutoff) {
        // Snap cleanly so the top visible speaker block starts with its full speaker/timestamp header intact
        targetTop = bTop - headerH - 8;
        break;
      }
    }
  }
  const finalTop = Math.max(0, targetTop);
  const jumpDist = Math.abs(finalTop - container.scrollTop);
  container.scrollTo({ top: finalTop, behavior: jumpDist > 2200 ? "auto" : "smooth" });
  row.classList.remove("cite-highlight");
  void row.offsetWidth;
  row.classList.add("cite-highlight");
  setTimeout(() => row.classList.remove("cite-highlight"), 1600);
}

/* ---------------- Final Calibration Panel ---------------- */

const NOTCH_WORDS = ["", "One notch", "Two notches", "Three notches"];

// The verdict is one sentence, said once: your call, then what the evidence says.
function renderVerdictCompareCard(insights) {
  const sr = state.session?.self_rating || {};
  const userKey = sr.hire_signal || "lean_hire";
  const userLabel = SIGNAL_LABELS[userKey] || userKey;
  const userIdx = CALIB_SCALE.findIndex((s) => s.key === userKey);
  const aiIdx = CALIB_SCALE.findIndex((s) => s.key === insights.aiKey);
  const aiTone = insights.aiKey ? (insights.aiKey.includes("no_hire") ? "tone-no" : "tone-hire") : "";

  const headline = el("verdictHeadline");
  const you = `You said ${escapeHtml(userLabel)}.`;
  let sub = "";
  if (userIdx >= 0 && aiIdx >= 0) {
    const diff = Math.abs(userIdx - aiIdx);
    const crossedLine = userIdx >= 2 !== aiIdx >= 2;
    if (diff === 0) {
      headline.innerHTML = `${you} <span class="verdict-ai ${aiTone}">The evidence agrees.</span>`;
      sub = "Your call matches what the transcript supports.";
    } else {
      headline.innerHTML = `${you} The evidence says <span class="verdict-ai ${aiTone}">${escapeHtml(insights.aiLabel)}</span>.`;
      const notches = NOTCH_WORDS[diff] || `${diff} notches`;
      sub = crossedLine
        ? `${notches} apart, on opposite sides of the hire line.`
        : `${notches} ${userIdx > aiIdx ? "more generous" : "stricter"} than the evidence.`;
    }
  } else if (insights.aiLabel === "No Read") {
    headline.innerHTML = `${you} The evidence is too thin to say.`;
    sub = "There was not enough signal in the transcript to compare against.";
  } else {
    headline.innerHTML = you;
    sub = "The audit below weighs that against the transcript.";
  }
  headline.title = insights.aiLead ? `From the audit: “${insights.aiLead}”` : "";

  const downLevel = (insights.aiDetail || "").match(/^Down-level → (.+)$/);
  if (downLevel) sub += ` The audit suggests ${downLevel[1]} instead.`;
  const yourNote = [sr.perceived_level, sr.gut_rationale].filter(Boolean).join(" — ");
  if (yourNote) sub += ` Your note: ${yourNote}.`;
  el("verdictSub").textContent = sub;

  renderPhaseTable(insights);
}

// One calm table under the verdict: every phase, how it read, and how long it ran.
function renderPhaseTable(insights) {
  const perCard = state.session?.metrics?.per_card || [];
  const signals = new Map((insights.phaseSignals || []).map((ps) => [ps.badge, ps]));
  const table = el("phaseSignalsStrip");

  const rows = perCard.length
    ? perCard.map((c) => ({ badge: String(c.badge || "").toUpperCase(), title: c.title, stats: c }))
    : [...signals.values()].map((ps) => ({ badge: ps.badge, title: ps.title, stats: null }));

  table.innerHTML = rows
    .map(({ badge, title, stats }) => {
      const sig = signals.get(badge);
      const detail = stats
        ? `${fmtClock(stats.span_seconds)} · candidate ${stats.candidate_words} words · you ${stats.interviewer_words} words`
        : "";
      return `
        <button type="button" class="phase-row" data-jump-badge="${escapeHtml(badge)}" title="${escapeHtml(detail ? `Show ${badge} in the transcript — ${detail}` : `Show ${badge} in the transcript`)}">
          <span class="phase-row-code">${escapeHtml(badge)}</span>
          <span class="phase-row-title">${escapeHtml(title)}</span>
          <span class="phase-row-rating ${sig ? sig.tone : ""}">${sig ? escapeHtml(sig.rating) : ""}</span>
          <span class="phase-row-time">${stats ? fmtClock(stats.span_seconds) : ""}</span>
        </button>
      `;
    })
    .join("");
  syncPhaseTableActive();
}

// Keep the pinned section tabs in sync with whichever audit section is currently being read
function syncReportNavToScroll() {
  if (Date.now() < (state.navLockUntil || 0)) return;
  const view = el("calibrationView");
  const pills = [...el("reportSectionNav").querySelectorAll("[data-scroll-sec]")];
  if (!pills.length || view.classList.contains("hidden")) return;
  const threshold = view.getBoundingClientRect().top + el("reportToolbar").offsetHeight + 24;
  let activePill = pills[0];
  for (const pill of pills) {
    const heading = document.getElementById(pill.getAttribute("data-scroll-sec"));
    if (heading && heading.getBoundingClientRect().top <= threshold) {
      activePill = pill;
    }
  }
  for (const pill of pills) {
    const isAct = pill === activePill;
    pill.classList.toggle("active", isAct);
    pill.setAttribute("aria-selected", String(isAct));
  }
}

function renderCalibrationPanel() {
  const sm = state.session?.state_machines;
  const calibRunning = sm?.jobs?.calibration === "running";
  const allowed = allowedEventsSet();
  const reportMd = state.session?.calibration_markdown || "";
  const hasReport = Boolean(reportMd);

  const calibBtn = el("runCalibrationBtn");
  calibBtn.disabled = !allowed.has("generate_calibration") || calibRunning;
  calibBtn.classList.toggle("is-loading", calibRunning);
  calibBtn.setAttribute("aria-busy", calibRunning ? "true" : "false");
  calibBtn.textContent = calibRunning ? "Reading the transcript…" : "Reveal the audit";
  const blindNote = el("blindCallNote");
  if (blindNote) {
    const blockedWhy = sm?.session?.blocked_reasons?.generate_calibration || "";
    blindNote.textContent =
      !calibRunning && !allowed.has("generate_calibration")
        ? state.captureState === "streaming"
          ? "Available when the interview ends."
          : blockedWhy
        : "";
  }

  const showVerdictCard = hasReport && !state.editingSelfRating && !calibRunning;
  el("verdictCompareCard").classList.toggle("hidden", !showVerdictCard);
  el("selfRatingCard").classList.toggle("hidden", showVerdictCard);
  el("collapseSelfRatingBtn").classList.toggle("hidden", !hasReport);

  const outputEl = el("calibrationOutput");
  const toolbarEl = el("reportToolbar");
  const secNavEl = el("reportSectionNav");

  const sr = state.session?.self_rating || null;
  const perCard = state.session?.metrics?.per_card || null;
  const srSig = sr
    ? `${sr.hire_signal || ""}|${sr.perceived_level || ""}|${sr.gut_rationale || ""}|${sr.biggest_concern || ""}`
    : "";
  const perCardSig = perCard
    ? perCard
        .map((c) => `${c.badge}|${c.title}|${c.span_seconds}|${c.candidate_words}|${c.interviewer_words}`)
        .join(";")
    : "";
  const streamingMd = calibRunning ? state.streamingCalibrationMd : "";
  const reportSig = `${calibRunning ? 1 : 0}|${hasReport ? 1 : 0}|${showVerdictCard ? 1 : 0}|${srSig}|${perCardSig}|${streamingMd.length}|${reportMd}`;

  if (outputEl._reportSig === reportSig) {
    if (showVerdictCard) {
      syncPhaseTableActive();
    }
    return;
  }
  outputEl._reportSig = reportSig;

  if (calibRunning) {
    toolbarEl.classList.add("hidden");
    if (streamingMd) {
      outputEl.classList.remove("empty-report");
      outputEl.innerHTML = renderMarkdownSafe(streamingMd);
    } else {
      outputEl.classList.add("empty-report");
      outputEl.innerHTML = `<p class="probe-thinking" role="status" aria-live="polite"><span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span><span>Reading the whole transcript against your call…</span></p>`;
    }
  } else if (hasReport) {
    const insights = parseCalibrationInsights(reportMd);
    if (showVerdictCard) {
      renderVerdictCompareCard(insights);
    }

    toolbarEl.classList.remove("hidden");
    secNavEl.innerHTML = insights.sections
      .map(
        (s, idx) =>
          `<button type="button" class="report-nav-link${idx === 0 ? " active" : ""}" role="tab" aria-selected="${idx === 0}" data-scroll-sec="${escapeHtml(s.id)}" title="${escapeHtml(s.title)}">${escapeHtml(s.short)}</button>`
      )
      .join("");

    outputEl.classList.remove("empty-report");
    outputEl.innerHTML = renderMarkdownSafe(reportMd);
    syncReportNavToScroll();
  } else {
    toolbarEl.classList.add("hidden");
    outputEl.classList.remove("empty-report");
    outputEl.textContent = "";
  }
}

function activateRightTab(which) {
  state.rightTab = which;
  const isProbes = which === "probes";
  el("tabProbes").classList.toggle("active", isProbes);
  el("tabProbes").setAttribute("aria-selected", String(isProbes));
  el("tabCalibration").classList.toggle("active", !isProbes);
  el("tabCalibration").setAttribute("aria-selected", String(!isProbes));
  el("probesView").classList.toggle("hidden", !isProbes);
  el("calibrationView").classList.toggle("hidden", isProbes);
  renderCaptureControls();
  renderMetricsBar();
}

/* ---------------- Settings & History Modals ---------------- */

// Model choosers. With named presets on offer (the agy tool), the chips are the single source of
// truth and the raw model ID only appears under "Custom". Providers without presets just show the ID.
// `redetect` re-derives preset-vs-custom from the current value (on open and on provider change);
// otherwise Custom stays put while the ID is being typed, even if it happens to spell a preset.
function syncModelChoosers({ redetect = false } = {}) {
  for (const group of document.querySelectorAll(".quick-chips[data-target-input]")) {
    const input = el(group.getAttribute("data-target-input"));
    if (!input) continue;
    const hasPresets = !group.classList.contains("hidden");
    const curVal = input.value.trim();
    const presetChips = [...group.querySelectorAll("[data-chip-val]")];
    const matchesPreset = presetChips.some((c) => c.getAttribute("data-chip-val") === curVal);
    if (redetect) {
      group.dataset.custom = String(!matchesPreset);
    }
    const isCustom = group.dataset.custom === "true";
    for (const chip of presetChips) {
      const active = !isCustom && chip.getAttribute("data-chip-val") === curVal;
      chip.classList.toggle("active", active);
      chip.setAttribute("aria-pressed", String(active));
    }
    const customChip = group.querySelector("[data-chip-custom]");
    if (customChip) {
      customChip.classList.toggle("active", isCustom);
      customChip.setAttribute("aria-pressed", String(isCustom));
    }
    input.classList.toggle("hidden", hasPresets && !isCustom);
  }
}

async function fetchDiscovery() {
  try {
    const res = await fetch("/api/providers/discover");
    if (res.ok) {
      state.discovery = await res.json();
    }
  } catch {
    /* ignore */
  }
  return state.discovery || { detected_clis: [], local_servers: {}, env_keys_present: [], system_memory_gb: null };
}

function syncHomeAiPrompt(llm) {
  const btn = el("homeSetupAiBtn");
  if (!btn) return;
  const unconfigured = !llm || !llm.is_configured;
  btn.classList.toggle("hidden", !unconfigured);
}

function showInlineTestResult(targetEl, resObj) {
  if (!targetEl) return;
  targetEl.classList.remove("hidden", "is-ok", "is-err");
  if (resObj.ok) {
    targetEl.classList.add("is-ok");
    targetEl.textContent = `✓ ${resObj.message || "Connected."}`;
  } else {
    targetEl.classList.add("is-err");
    const detail = resObj.hint ? `${resObj.hint}` : (resObj.error || "Connection failed.");
    targetEl.textContent = `✕ ${detail}`;
  }
}

async function openSettingsModal() {
  closeAllPopovers();
  const reqToken = modalCtrl.beginRequest();
  let statusRes;
  let disc;
  try {
    [statusRes, disc] = await Promise.all([fetch("/api/status"), fetchDiscovery()]);
  } catch (err) {
    if (modalCtrl.isCurrent(reqToken)) {
      showToast(`Could not load settings: ${err.message || "network error"}`);
    }
    return;
  }
  const data = await statusRes.json();
  if (!modalCtrl.isCurrent(reqToken)) return;
  state.llm = data.llm;
  syncHomeAiPrompt(data.llm);
  profileAddons?.applyStatusSnapshot(data);
  profileAddons?.renderAddonsSection();

  const testResultEl = el("testLlmResult");
  if (testResultEl) {
    testResultEl.classList.add("hidden");
    testResultEl.textContent = "";
  }

  const dataDirEl = el("dataDirPath");
  if (dataDirEl && data.sessions_dir) {
    dataDirEl.textContent = data.sessions_dir;
  }

  const ckptSelect = el("parakeetModelSelect");
  const ckpts = data.asr.checkpoints || {};
  ckptSelect.innerHTML = Object.entries(ckpts)
    .map(([id, label]) => `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`)
    .join("");
  if (data.asr.model_id) {
    ckptSelect.value = data.asr.model_id;
  }

  const presetSelect = el("llmPresetSelect");
  const presets = data.llm.presets || {};
  const detectedCliIds = new Set((disc.detected_clis || []).map((c) => c.provider));
  const localServers = disc.local_servers || {};

  // Build categorized <optgroup>s: Detected CLIs (only those installed or currently active), Cloud APIs, Local Offline, Transcript-Only
  const cliOpts = Object.entries(presets)
    .filter(([key, val]) => val.category === "cli" && (detectedCliIds.has(key) || data.llm.provider === key))
    .map(([key, val]) => {
      const suffix = detectedCliIds.has(key) ? " · Detected" : " · Not installed";
      return `<option value="${escapeHtml(key)}">${escapeHtml(val.label || key)}${suffix}</option>`;
    });

  const cloudOpts = Object.entries(presets)
    .filter(([, val]) => val.category === "cloud")
    .map(([key, val]) => {
      const badge = val.badge ? ` · ${val.badge}` : "";
      return `<option value="${escapeHtml(key)}">${escapeHtml(val.label || key)}${badge}</option>`;
    });

  const localOpts = Object.entries(presets)
    .filter(([, val]) => val.category === "local")
    .map(([key, val]) => {
      const srv = localServers[key];
      let badge = " · Offline";
      if (srv?.running) {
        badge = srv.auth_required
          ? " · Running (API key required)"
          : ` · Running (${(srv.models || []).length} models)`;
      }
      return `<option value="${escapeHtml(key)}">${escapeHtml(val.label || key)}${badge}</option>`;
    });

  let html = "";
  if (cliOpts.length) {
    html += `<optgroup label="Detected on this Mac (No API Key)">${cliOpts.join("")}</optgroup>`;
  }
  html += `<optgroup label="Cloud APIs">${cloudOpts.join("")}</optgroup>`;
  html += `<optgroup label="Local Models (100% Offline)">${localOpts.join("")}</optgroup>`;
  html += `<optgroup label="No AI"><option value="none">Transcript-Only (Offline speech only)</option></optgroup>`;
  presetSelect.innerHTML = html;

  if (data.llm.provider === "none" || presets[data.llm.provider]) {
    presetSelect.value = data.llm.provider;
  }

  el("llmBaseUrlInput").value = data.llm.base_url || "";
  el("llmFastModelInput").value = data.llm.fast_model || "";
  el("llmCalibModelInput").value = data.llm.calibration_model || "";
  el("llmApiKeyInput").value = "";

  const localRuntimeName = (providerKey) =>
    providerKey === "omlx" ? "oMLX" : providerKey === "ollama" ? "Ollama" : "LM Studio";

  const populateLocalModelPicker = (providerKey) => {
    const localRow = el("localModelRow");
    const localSelect = el("localModelSelect");
    const ramNote = el("localRamNote");
    const srv = localServers[providerKey];
    const isLocal = presets[providerKey]?.category === "local";
    localRow.classList.toggle("hidden", !isLocal);
    if (!isLocal) return;

    const models = srv?.models || [];
    if (srv?.running && models.length) {
      localSelect.disabled = false;
      localSelect.innerHTML = models
        .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`)
        .join("");
      const cur = el("llmFastModelInput").value.trim();
      if (models.includes(cur)) {
        localSelect.value = cur;
      } else {
        localSelect.value = models[0];
        el("llmFastModelInput").value = models[0];
        el("llmCalibModelInput").value = models[0];
      }
      localSelect.onchange = () => {
        el("llmFastModelInput").value = localSelect.value;
        el("llmCalibModelInput").value = localSelect.value;
      };
    } else {
      localSelect.disabled = true;
      let emptyLabel = `${localRuntimeName(providerKey)} is not running`;
      if (srv?.running && srv.auth_required) {
        emptyLabel = `${localRuntimeName(providerKey)} requires an API key — enter it above`;
      } else if (srv?.running) {
        emptyLabel = "No models loaded yet";
      }
      localSelect.innerHTML = `<option value="">${escapeHtml(emptyLabel)}</option>`;
    }
    const memGb = disc.system_memory_gb;
    if (ramNote) {
      ramNote.textContent = memGb
        ? `Your Mac has ${memGb} GB unified memory${memGb >= 16 ? " (handles 7B–14B+ local models smoothly)." : " (8B or smaller models recommended, or use a Cloud API)."}`
        : "16 GB+ unified memory recommended for local LLMs.";
    }
  };

  const updateProviderFields = (providerKey) => {
    const preset = presets[providerKey] || {};
    const srv = localServers[providerKey];
    const isNone = providerKey === "none";
    const isCli = preset.category === "cli";
    const isLocal = preset.category === "local";
    const needsKey = Boolean(preset.requires_key);
    const showOptionalKey =
      Boolean(preset.optional_key) &&
      (Boolean(srv?.auth_required) || (providerKey === data.llm.provider && Boolean(data.llm.has_api_key)));
    const showKeyRow = needsKey || showOptionalKey;

    el("apiKeyRow").classList.toggle("hidden", !showKeyRow);
    el("advancedLlmDetails").classList.toggle("hidden", isNone);
    el("baseUrlRow").classList.toggle("hidden", isCli);
    el("testLlmBtn").classList.toggle("hidden", isNone);

    for (const group of document.querySelectorAll(".quick-chips")) {
      group.classList.toggle("hidden", providerKey !== "agy");
    }

    const hintEl = el("providerHintText");
    const signupEl = el("providerSignupLink");
    if (isNone) {
      hintEl.textContent = "Offline transcription, diarization, cards, and Markdown export work with zero AI setup.";
      signupEl.classList.add("hidden");
    } else if (isCli) {
      hintEl.textContent = preset.hint || "Uses your signed-in local CLI. No API key needed.";
      signupEl.classList.add("hidden");
    } else {
      hintEl.textContent = preset.hint || "";
      if (preset.signup_url) {
        signupEl.href = preset.signup_url;
        signupEl.textContent = isLocal ? `Get ${localRuntimeName(providerKey)} ↗` : "Get API key ↗";
        signupEl.classList.remove("hidden");
      } else {
        signupEl.classList.add("hidden");
      }
    }

    if (showKeyRow) {
      const sameProvider = providerKey === data.llm.provider;
      if (showOptionalKey && !needsKey) {
        el("maskedKeyHint").textContent =
          sameProvider && data.llm.has_api_key
            ? `Saved key: ${data.llm.masked_api_key}`
            : srv?.auth_required
              ? `${localRuntimeName(providerKey)} has API key auth enabled.`
              : "Optional — leave blank if local auth is disabled.";
        el("llmApiKeyInput").placeholder =
          sameProvider && data.llm.has_api_key
            ? "Leave blank to keep saved key"
            : `Paste your ${localRuntimeName(providerKey)} API key`;
      } else {
        el("maskedKeyHint").textContent =
          sameProvider && data.llm.has_api_key
            ? `Saved key: ${data.llm.masked_api_key}`
            : "No key saved yet.";
        el("llmApiKeyInput").placeholder =
          sameProvider && data.llm.has_api_key
            ? "Leave blank to keep saved key"
            : `Paste your ${preset.label || providerKey} API key`;
      }
    }

    populateLocalModelPicker(providerKey);
  };

  updateProviderFields(presetSelect.value);
  syncModelChoosers({ redetect: true });

  presetSelect.onchange = () => {
    const provKey = presetSelect.value;
    const p = presets[provKey];
    const srv = localServers[provKey];
    if (p) {
      el("llmBaseUrlInput").value = (srv?.base_url || p.base_url || "");
      el("llmFastModelInput").value = p.fast_model || "";
      el("llmCalibModelInput").value = p.calibration_model || "";
    } else if (provKey === "none") {
      el("llmBaseUrlInput").value = "";
      el("llmFastModelInput").value = "";
      el("llmCalibModelInput").value = "";
    }
    if (testResultEl) {
      testResultEl.classList.add("hidden");
    }
    updateProviderFields(provKey);
    syncModelChoosers({ redetect: true });
  };

  setModalState("settings", { reqToken, event: "open_settings" });
}

async function testSettingsConnection() {
  const btn = el("testLlmBtn");
  const out = el("testLlmResult");
  btn.disabled = true;
  btn.classList.add("is-loading");
  btn.setAttribute("aria-busy", "true");
  const origLabel = btn.textContent;
  btn.textContent = "Testing…";
  out.classList.remove("hidden", "is-ok", "is-err");
  out.textContent = "Sending a quick test ping…";

  try {
    const res = await fetch("/api/settings/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: el("llmPresetSelect").value,
        base_url: el("llmBaseUrlInput").value,
        fast_model: el("llmFastModelInput").value,
        calibration_model: el("llmCalibModelInput").value,
        api_key: el("llmApiKeyInput").value || null,
      }),
    });
    const data = await res.json();
    showInlineTestResult(out, data);
  } catch (err) {
    showInlineTestResult(out, { ok: false, error: err.message || "Network error" });
  } finally {
    btn.disabled = false;
    btn.classList.remove("is-loading");
    btn.setAttribute("aria-busy", "false");
    btn.textContent = origLabel;
  }
}

async function openDataDirectory() {
  try {
    const res = await fetch("/api/open-data-dir", { method: "POST" });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      showToast(formatApiDetail(data.detail, `Sessions folder: ${data.path || "local"}`), 3500);
    }
  } catch (err) {
    showToast(`Could not open folder: ${err.message || "network error"}`);
  }
}

async function saveSettings() {
  if (state.savingSettings) return;
  state.savingSettings = true;
  const settingsOpenToken = modalCtrl.currentToken;
  clientMachines.beginOp("save_settings");
  const saveBtn = el("saveSettingsBtn");
  const origLabel = saveBtn?.textContent || "Save";
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving…";
  }
  const payload = {
    provider: el("llmPresetSelect").value,
    base_url: el("llmBaseUrlInput").value,
    fast_model: el("llmFastModelInput").value,
    calibration_model: el("llmCalibModelInput").value,
    api_key: el("llmApiKeyInput").value || null,
    parakeet_model: el("parakeetModelSelect").value,
  };
  try {
    const res = await fetch("/api/settings/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast(formatApiDetail(err.detail, `Could not save settings (${res.status}).`));
      return;
    }
    const data = await res.json();
    if (data.llm) {
      state.llm = data.llm;
      syncHomeAiPrompt(data.llm);
    }
    if (data.asr) {
      state.asr = data.asr;
      renderAsrBadge();
    }
    showToast("Settings saved.", 2200);
    modalCtrl.closeIf("settings", settingsOpenToken);
  } catch (err) {
    showToast(`Failed to save settings: ${err.message || "network error"}`);
  } finally {
    state.savingSettings = false;
    clientMachines.endOp("save_settings");
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = origLabel;
    }
  }
}

function shouldPromptVoiceStep() {
  if (state.tts) {
    const anyInstalled = Boolean(
      state.tts.neural_installed || state.tts.kokoro_installed || state.tts.phonon_installed
    );
    const anyAvailable = Boolean(state.tts.kokoro_available || state.tts.phonon_available);
    return anyAvailable && !anyInstalled;
  }
  const packs = state.addons?.packs || [];
  const voicePacks = packs.filter((p) => p.id === "phonon_tts" || p.id === "kokoro_tts");
  const anyInstalled = voicePacks.some((p) => p.installed);
  const anyAvailable = voicePacks.some((p) => p.available);
  return anyAvailable && !anyInstalled;
}

function isCloudLlmActive() {
  const prov = state.llm?.provider || "";
  return ["gemini", "anthropic", "openai", "deepseek", "openrouter"].includes(prov);
}

function renderOnboardingPacksStep() {
  const packs = state.addons?.packs || [];
  const ttsEngines = state.tts?.engines || {};

  const renderVoiceCard = ({
    packId,
    cardId,
    badgeId,
    previewBtnId,
    installBtnId,
    errId,
    shortName,
    defaultSize,
    ttsInstalled,
    ttsAvailable,
    ttsError,
  }) => {
    const pack = packs.find((p) => p.id === packId) || null;
    const op = packOps.get(packId);
    const installed = Boolean(ttsInstalled || pack?.installed);
    const available =
      ttsAvailable !== undefined ? Boolean(ttsAvailable) : pack?.available !== false;
    const size = pack?.size_mb || defaultSize;

    const cardEl = el(cardId);
    const badgeEl = el(badgeId);
    const previewBtn = el(previewBtnId);
    const installBtn = el(installBtnId);
    const errEl = el(errId);

    if (cardEl) {
      cardEl.classList.toggle("is-installed", installed);
    }
    if (badgeEl) badgeEl.classList.toggle("hidden", !installed);
    if (previewBtn) {
      previewBtn.classList.toggle("hidden", !installed);
    }
    if (installBtn) {
      installBtn.classList.toggle("hidden", installed || !available);
      const busy = op === "installing" || op === "uninstalling";
      installBtn.disabled = busy;
      // Size lives once, in the card's description line — never repeated on the button.
      installBtn.textContent =
        op === "installing"
          ? "Downloading…"
          : op === "uninstalling"
            ? "Removing…"
            : pack?.state === "error"
              ? `Retry ${shortName}`
              : `Install ${shortName}`;
      installBtn.classList.toggle("is-loading", op === "installing");
      installBtn.title = `${shortName} · ${size} MB download, then fully offline`;
    }
    if (errEl) {
      const errMsg = pack?.error || ttsError || "";
      errEl.textContent = errMsg;
      errEl.classList.toggle("hidden", !errMsg);
    }
    return installed;
  };

  const phononInstalled = renderVoiceCard({
    packId: "phonon_tts",
    cardId: "onboardPhononCard",
    badgeId: "onboardPhononBadge",
    previewBtnId: "onboardPhononPreviewBtn",
    installBtnId: "onboardPhononInstallBtn",
    errId: "onboardPhononError",
    shortName: "Phonon",
    defaultSize: 62,
    ttsInstalled: state.tts?.phonon_installed ?? ttsEngines.phonon?.installed,
    ttsAvailable: state.tts?.phonon_available ?? ttsEngines.phonon?.available,
    ttsError: ttsEngines.phonon?.error,
  });

  const kokoroInstalled = renderVoiceCard({
    packId: "kokoro_tts",
    cardId: "onboardKokoroCard",
    badgeId: "onboardKokoroBadge",
    previewBtnId: "onboardKokoroPreviewBtn",
    installBtnId: "onboardKokoroInstallBtn",
    errId: "onboardKokoroError",
    shortName: "Kokoro",
    defaultSize: 183,
    ttsInstalled: state.tts?.kokoro_installed ?? ttsEngines.kokoro?.installed,
    ttsAvailable: state.tts?.kokoro_available ?? ttsEngines.kokoro?.available,
    ttsError: ttsEngines.kokoro?.error,
  });

  const anyVoiceInstalled = phononInstalled || kokoroInstalled;
  const doneBtn = el("onboardDoneBtn");
  if (doneBtn) {
    doneBtn.textContent = anyVoiceInstalled ? "Start using Mockery" : "Continue with macOS voice";
    // One filled CTA per step: until a neural voice is in, that is "Install Phonon".
    doneBtn.classList.toggle("btn-primary", anyVoiceInstalled);
    doneBtn.classList.toggle("btn-outline", !anyVoiceInstalled);
  }

  const optList = el("onboardOptionalPacksList");
  if (optList) {
    const cloudActive = isCloudLlmActive();
    const optionalIds = ["rampart_pii", "pyodide_python"];
    const optionalPacks = optionalIds
      .map((id) => packs.find((p) => p.id === id))
      .filter(Boolean);

    // Pack names come from the catalog, so every surface says the same thing;
    // onboarding only shortens the description and states the size once.
    const shortDesc = {
      rampart_pii: "Hides names and contact details before anything reaches a cloud AI",
      pyodide_python: "Runs your Python tests on this Mac",
    };

    optList.innerHTML = optionalPacks
      .map((pk) => {
        const desc = [shortDesc[pk.id] || pk.description, pk.size_mb ? `${pk.size_mb} MB` : ""]
          .filter(Boolean)
          .join(" · ");
        const op = packOps.get(pk.id);
        const isInstalling = op === "installing" || pk.state === "installing";
        const isUninstalling = op === "uninstalling" || pk.state === "uninstalling";
        const recTag =
          pk.id === "rampart_pii" && cloudActive
            ? `<span class="addon-size-pill is-rec">Recommended with cloud AI</span>`
            : "";
        let actionHtml = "";
        if (isInstalling) {
          actionHtml = `<span class="addon-state-badge is-busy">Downloading…</span>`;
        } else if (isUninstalling) {
          actionHtml = `<span class="addon-state-badge is-busy">Removing…</span>`;
        } else if (pk.installed) {
          actionHtml = `<span class="addon-state-badge is-ok">Installed</span>`;
        } else {
          actionHtml = `<button type="button" class="btn btn-outline btn-sm" data-onboard-install="${escapeHtml(
            pk.id
          )}">Install</button>`;
        }
        return `
          <div class="onboard-pack-row">
            <div class="onboard-pack-main">
              <div class="onboard-pack-title">
                ${escapeHtml(pk.name)}
                ${recTag}
              </div>
              <div class="onboard-pack-desc">${escapeHtml(desc)}</div>
              ${pk.error ? `<div class="addon-row-err">${escapeHtml(pk.error)}</div>` : ""}
            </div>
            <div class="onboard-pack-actions">${actionHtml}</div>
          </div>
        `;
      })
      .join("");
  }
}

function setOnboardStep(step) {
  const nextStep = step === "packs" ? "packs" : "ai";
  state.onboardStep = nextStep;
  el("onboardStepAi")?.classList.toggle("hidden", nextStep !== "ai");
  el("onboardStepPacks")?.classList.toggle("hidden", nextStep !== "packs");

  for (const pill of document.querySelectorAll("[data-onboard-step]")) {
    const active = pill.getAttribute("data-onboard-step") === nextStep;
    pill.classList.toggle("active", active);
    pill.classList.toggle(
      "is-done",
      nextStep === "packs" && pill.getAttribute("data-onboard-step") === "ai"
    );
    pill.setAttribute("aria-selected", active ? "true" : "false");
    if (active) pill.setAttribute("aria-current", "step");
    else pill.removeAttribute("aria-current");
  }

  el("onboardSkipBtn")?.classList.toggle("hidden", nextStep !== "ai");
  el("onboardBackBtn")?.classList.toggle("hidden", nextStep !== "packs");
  el("onboardNextStepBtn")?.classList.toggle("hidden", nextStep !== "ai");
  el("onboardDoneBtn")?.classList.toggle("hidden", nextStep !== "packs");

  if (nextStep === "packs") {
    renderOnboardingPacksStep();
  }
}

async function openOnboardingModal(statusData = null, { initialStep = null } = {}) {
  const reqToken = modalCtrl.beginRequest();
  let data;
  let disc;
  try {
    [data, disc] = await Promise.all([
      statusData ? Promise.resolve(statusData) : fetch("/api/status").then((r) => r.json()),
      fetchDiscovery(),
    ]);
  } catch (err) {
    if (modalCtrl.isCurrent(reqToken)) {
      showToast(`Could not load AI setup: ${err.message || "network error"}`);
    }
    return;
  }
  if (!modalCtrl.isCurrent(reqToken)) return;
  state.llm = data.llm;
  if (data.addons) {
    state.addons = data.addons;
    profileAddons?.applyStatusSnapshot({ addons: data.addons });
  }
  if (data.tts) {
    state.tts = data.tts;
    practiceStudio?.applyStatusSnapshot({ tts: data.tts });
  }
  const presets = data.llm.presets || {};

  // 1. Detected local CLIs & running local LLM servers (1-click activation!)
  const detectedItems = [];
  for (const cli of disc.detected_clis || []) {
    detectedItems.push({
      provider: cli.provider,
      label: cli.label,
      sub: cli.hint || `Uses local ${cli.cli_bin} CLI`,
      badge: "Use now →",
      base_url: cli.base_url,
      fast_model: cli.fast_model,
      calibration_model: cli.calibration_model,
    });
  }
  const localSrv = disc.local_servers || {};
  for (const localKey of ["omlx", "ollama", "lmstudio"]) {
    const srv = localSrv[localKey];
    if (srv?.running && !srv.auth_required && (srv.models || []).length > 0) {
      const m = srv.models[0];
      const p = presets[localKey] || {};
      detectedItems.push({
        provider: localKey,
        label: `${p.label || localKey} (${m})`,
        sub: `${srv.models.length} local model${srv.models.length === 1 ? "" : "s"} ready offline on your Mac`,
        badge: "Use offline →",
        base_url: srv.base_url || p.base_url || "",
        fast_model: m,
        calibration_model: m,
      });
    }
  }

  const detSection = el("onboardDetectedSection");
  const detList = el("onboardDetectedList");
  if (detectedItems.length > 0) {
    detSection.classList.remove("hidden");
    detList.innerHTML = detectedItems
      .map(
        (item, idx) => `
          <button type="button" class="onboard-detected-btn" data-onboard-detected="${idx}">
            <div>
              <div class="onboard-detected-title">${escapeHtml(item.label)}</div>
              <div class="onboard-detected-sub">${escapeHtml(item.sub)}</div>
            </div>
            <span class="onboard-detected-badge">${escapeHtml(item.badge)}</span>
          </button>
        `
      )
      .join("");
    detList.onclick = async (ev) => {
      const btn = ev.target.closest("[data-onboard-detected]");
      if (!btn) return;
      const chosen = detectedItems[Number(btn.getAttribute("data-onboard-detected"))];
      if (!chosen) return;
      btn.disabled = true;
      const res = await fetch("/api/settings/llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: chosen.provider,
          base_url: chosen.base_url,
          fast_model: chosen.fast_model,
          calibration_model: chosen.calibration_model,
        }),
      });
      if (res.ok) {
        const saved = await res.json();
        state.llm = saved.llm;
        syncHomeAiPrompt(saved.llm);
        showToast(`Connected ${chosen.label}.`, 2600);
        if (shouldPromptVoiceStep()) {
          btn.disabled = false;
          setOnboardStep("packs");
        } else {
          setModalState("none");
        }
      } else {
        btn.disabled = false;
        showToast("Could not activate provider.");
      }
    };
  } else {
    detSection.classList.add("hidden");
    detList.innerHTML = "";
  }

  // 2. Cloud API quick selector
  const selectOnboardCloud = (provKey) => {
    state.onboardProvider = provKey;
    const p = presets[provKey] || {};
    for (const pill of document.querySelectorAll("[data-onboard-provider]")) {
      pill.classList.toggle("active", pill.getAttribute("data-onboard-provider") === provKey);
    }
    el("onboardApiKeyInput").placeholder = `Paste your ${p.label || provKey} API key`;
    el("onboardProviderHint").textContent = p.hint || "";
    const link = el("onboardSignupLink");
    if (p.signup_url) {
      link.href = p.signup_url;
      link.textContent = provKey === "gemini" ? "Get a free key ↗" : "Get an API key ↗";
      link.classList.remove("hidden");
    } else {
      link.classList.add("hidden");
    }
    el("onboardStatus").classList.add("hidden");
  };
  selectOnboardCloud(state.onboardProvider || "gemini");
  el("onboardCloudPills").onclick = (ev) => {
    const pill = ev.target.closest("[data-onboard-provider]");
    if (pill) selectOnboardCloud(pill.getAttribute("data-onboard-provider"));
  };

  el("onboardApiKeyInput").value = "";
  el("onboardStatus").classList.add("hidden");
  setOnboardStep(initialStep || "ai");
  setModalState("onboarding", { reqToken, event: "open_onboarding" });
}

async function testAndSaveOnboardingKey() {
  const onboardToken = modalCtrl.currentToken;
  const provKey = state.onboardProvider || "gemini";
  const presets = state.llm?.presets || {};
  const p = presets[provKey] || {};
  const apiKey = el("onboardApiKeyInput").value.trim();
  const statusEl = el("onboardStatus");
  if (!apiKey) {
    showInlineTestResult(statusEl, { ok: false, hint: `Paste your ${p.label || provKey} API key first.` });
    return;
  }
  const btn = el("onboardTestSaveBtn");
  btn.disabled = true;
  btn.classList.add("is-loading");
  btn.setAttribute("aria-busy", "true");
  btn.textContent = "Testing…";
  statusEl.classList.remove("hidden", "is-ok", "is-err");
  statusEl.textContent = `Testing ${p.label || provKey}…`;
  try {
    const testRes = await fetch("/api/settings/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: provKey,
        base_url: p.base_url || "",
        fast_model: p.fast_model || "",
        calibration_model: p.calibration_model || "",
        api_key: apiKey,
      }),
    });
    const result = await testRes.json();
    showInlineTestResult(statusEl, result);
    if (!result.ok) return;

    const saveRes = await fetch("/api/settings/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: provKey,
        base_url: p.base_url || "",
        fast_model: p.fast_model || "",
        calibration_model: p.calibration_model || "",
        api_key: apiKey,
      }),
    });
    if (saveRes.ok) {
      const saved = await saveRes.json();
      state.llm = saved.llm;
      syncHomeAiPrompt(saved.llm);
      setTimeout(() => {
        if (!modalCtrl.isCurrent(onboardToken) || state.modalState !== "onboarding") return;
        showToast(`${p.label || provKey} connected and ready.`, 2600);
        if (shouldPromptVoiceStep()) {
          setOnboardStep("packs");
        } else {
          modalCtrl.closeIf("onboarding", onboardToken);
        }
      }, 450);
    }
  } catch (err) {
    showInlineTestResult(statusEl, { ok: false, error: err.message || "Network error" });
  } finally {
    btn.disabled = false;
    btn.classList.remove("is-loading");
    btn.setAttribute("aria-busy", "false");
    btn.textContent = "Test & Save";
  }
}

async function skipOnboardingTranscriptOnly() {
  try {
    const res = await fetch("/api/settings/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: "none",
        base_url: "",
        fast_model: "",
        calibration_model: "",
      }),
    });
    if (res.ok) {
      const saved = await res.json();
      state.llm = saved.llm;
      syncHomeAiPrompt(saved.llm);
      setModalState("none");
      showToast("Transcript-only mode enabled. You can connect an AI anytime in Settings.", 3200);
      return;
    }
    console.warn("[onboarding] Failed to persist transcript-only mode:", res.status);
  } catch (err) {
    console.warn("[onboarding] Failed to persist transcript-only mode:", err);
  }
  setModalState("none");
  showToast("Couldn't save transcript-only preference — you can update it anytime in Settings.", 3600);
}

async function checkInitialOnboarding() {
  if (state.onboardingChecked) return;
  state.onboardingChecked = true;
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    state.llm = data.llm;
    state.addons = data.addons || state.addons;
    state.tts = data.tts || state.tts;
    if (data.diarizer) {
      state.diarizer = data.diarizer;
      renderSpeakerToggle();
    }
    syncHomeAiPrompt(data.llm);
    profileAddons?.applyStatusSnapshot(data);
    practiceStudio?.applyStatusSnapshot(data);
    practiceStudio?.refreshPracticeHistory();
    if (data.llm?.needs_onboarding && state.modalState === "none") {
      await openOnboardingModal(data, { initialStep: "ai" });
    } else if (
      (data.tts?.kokoro_available || data.tts?.phonon_available) &&
      !(data.tts?.neural_installed || data.tts?.kokoro_installed || data.tts?.phonon_installed) &&
      state.modalState === "none"
    ) {
      let seenPacks = false;
      try {
        seenPacks = window.localStorage?.getItem("mockery_onboard_packs_seen") === "1";
      } catch {
        seenPacks = false;
      }
      if (!seenPacks) {
        await openOnboardingModal(data, { initialStep: "packs" });
      }
    }
  } catch {
    /* ignore */
  }
}

function fmtDayLabel(isoStr) {
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return "Earlier";
  const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((startOf(new Date()) - startOf(d)) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  return d.toLocaleDateString(undefined, { month: "long", day: "numeric" });
}

function fmtDurationShort(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  if (total < 60) return "<1m";
  const mins = Math.round(total / 60);
  if (mins < 60) return `${mins}m`;
  const rem = mins % 60;
  return rem ? `${Math.floor(mins / 60)}h ${rem}m` : `${Math.floor(mins / 60)}h`;
}

// What tells two interviews for the same role apart: the format, how far it got, how long it ran —
// and, in its own column, when it happened. (`/api/sessions` carries no card titles to quote.)
function sessionMetaParts(s, { withDay = false } = {}) {
  const d = new Date(s.created_at_iso);
  const clock = Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const cards = Number(s.qa_count) || 0;
  const main = [
    TYPE_SHORT_LABELS[s.interview_type] || s.interview_type,
    cards > 0 ? `${cards} card${cards === 1 ? "" : "s"}` : "No cards",
    fmtDurationShort(s.duration_seconds),
  ]
    .filter(Boolean)
    .join(" · ");
  const time = withDay && clock ? `${fmtDayLabel(s.created_at_iso)}, ${clock}` : clock;
  return { main, time };
}

function sessionDotHtml(s) {
  return s.has_calibration
    ? `<span class="session-dot" role="img" aria-label="Calibrated" title="Calibrated — the verdict and audit are saved"></span>`
    : "";
}

async function fetchSavedSessions() {
  const res = await fetch("/api/sessions");
  const data = await res.json();
  return data.sessions || [];
}

// Home sidebar: past interviews, grouped by day, one click to reopen.
async function refreshHomeSessions() {
  const list = el("homeSessionList");
  if (!list) return;
  let sessions = [];
  try {
    sessions = await fetchSavedSessions();
  } catch (_) {
    return;
  }
  if (!sessions.length) {
    list.innerHTML = `<div class="home-empty">Interviews you run are kept here.</div>`;
    return;
  }
  let lastDay = null;
  list.innerHTML = sessions
    .map((s) => {
      const day = fmtDayLabel(s.created_at_iso);
      const head = day !== lastDay ? `<div class="home-group">${escapeHtml(day)}</div>` : "";
      lastDay = day;
      const meta = sessionMetaParts(s);
      return `${head}
        <button type="button" class="home-item" data-load-session="${escapeHtml(s.id)}">
          <span class="home-item-top">
            <span class="home-item-title">${escapeHtml(s.role_title || "Untitled interview")}</span>
            ${sessionDotHtml(s)}
          </span>
          <span class="home-item-meta">
            <span class="session-meta-main">${escapeHtml(meta.main)}</span>
            <span class="session-time">${escapeHtml(meta.time)}</span>
          </span>
        </button>`;
    })
    .join("");
}

async function openHistoryModal() {
  closeAllPopovers();
  const reqToken = modalCtrl.beginRequest();
  let sessions;
  try {
    sessions = await fetchSavedSessions();
  } catch (err) {
    if (modalCtrl.isCurrent(reqToken)) {
      showToast(`Could not load interview history: ${err.message || "network error"}`);
    }
    return;
  }
  if (!modalCtrl.isCurrent(reqToken)) return;
  const list = el("historyList");
  if (!sessions.length) {
    list.innerHTML = `<div class="history-empty">No interviews saved yet.</div>`;
  } else {
    list.innerHTML = sessions
      .map((s) => {
        const meta = sessionMetaParts(s, { withDay: true });
        return `
        <div class="history-item">
          <button type="button" class="history-open" data-load-session="${escapeHtml(s.id)}">
            <span class="history-top">
              <span class="history-title">${escapeHtml(s.role_title || "Untitled interview")}</span>
              ${sessionDotHtml(s)}
            </span>
            <span class="history-meta">${escapeHtml([meta.time, meta.main].filter(Boolean).join(" · "))}</span>
          </button>
          <span class="history-actions">
            <a class="link-quiet" href="/api/sessions/${encodeURIComponent(s.id)}/markdown?audience=interviewer" title="Download the full debrief as Markdown">Debrief</a>
            <a class="link-quiet" href="/api/sessions/${encodeURIComponent(s.id)}/markdown?audience=candidate" title="Download a clean transcript for the candidate">Candidate copy</a>
          </span>
        </div>
      `;
      })
      .join("");
  }
  setModalState("history", { reqToken, event: "open_history" });
}

async function loadSavedSession(id) {
  if (!id || state.loadingSavedSession) return;
  if (!allowedEventsSet().has("load_historical_session") || state.captureState !== "idle") {
    showToast("End the recording before opening another interview.");
    return;
  }
  state.loadingSavedSession = true;
  clientMachines.beginOp("load_saved_session", { sessionId: id });
  try {
    if (state.practiceActive && practiceStudio?.exitPracticeToHome) {
      await practiceStudio.exitPracticeToHome();
    }
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/load`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast(formatApiDetail(err.detail, "Could not open that interview."));
      return;
    }
    const data = await res.json();
    state.session = data.session;
    state.optimisticBackup = null;
    state.activeCitationUttId = null;
    state.editingSelfRating = false;
    state.baseElapsedMs = data.session.live_elapsed_ms || 0;
    renderTimerPill(state.baseElapsedMs / 1000);
    const qas = state.session.qa_groups_resolved || [];
    state.selectedQaId = qas.length ? qas[0].id : null;
    syncConfigInputsFromSession();
    renderWorkspaceMode();
    renderSpeakerToggle();
    renderPhaseAndControls();
    renderQAList();
    // A reopened interview starts on its first phase, so show the top of the transcript to match
    // the highlighted phase tab (and the phase table in Calibration).
    el("qaList").scrollTop = 0;
    updateJumpToBottomVisibility();
    renderProbesPanel();
    renderCalibrationPanel();
    renderMetricsBar();
    activateRightTab(state.session.calibration_markdown ? "calibration" : "probes");
    setModalState("none", { event: "loaded_saved_session" });
  } catch (err) {
    showToast(`Could not open that interview: ${err.message || "network error"}`);
  } finally {
    state.loadingSavedSession = false;
    clientMachines.endOp("load_saved_session");
  }
}

/* ---------------- Event Listeners & Hotkeys ---------------- */

function currentConfigPayload() {
  return {
    role_title: el("roleInput").value,
    interview_type: el("typeSelect").value,
    mode: el("modeSelect").value,
    rubric_notes: el("rubricInput").value,
    save_raw_audio: el("saveAudioCheckbox").checked,
  };
}

function pushSessionConfig() {
  updateContextSummaryChip();
  renderWorkspaceMode();
  sendCmd({ type: "configure_session", config: currentConfigPayload() });
}

function selectAndJumpToQa(qaId) {
  state.selectedQaId = qaId;
  renderPhaseNavStrip(state.session?.qa_groups_resolved || []);
  renderProbesPanel();
  const container = el("qaList");
  const card = container.querySelector(`[data-qa-card="${CSS.escape(qaId)}"]`);
  if (card) {
    for (const c of container.querySelectorAll(".qa-card.selected")) {
      c.classList.remove("selected");
    }
    card.classList.add("selected");
    const targetTop = Math.max(
      0,
      card.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop
    );
    const jumpDist = Math.abs(targetTop - container.scrollTop);
    container.scrollTo({ top: targetTop, behavior: jumpDist > 2200 ? "auto" : "smooth" });
  }
}

/* The Home role headline is a title that wraps instead of truncating. Enter commits
   rather than breaking the line, pasted line breaks collapse to spaces, and a very long
   title steps the type down so it still settles in two lines. Scripted writes go
   through the same path, so every caller keeps assigning `.value` as before. */
function bindRoleHeadline(field) {
  if (!field) return;
  const native = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
  const sizesItself = typeof CSS !== "undefined" && CSS.supports?.("field-sizing", "content");

  const fit = () => {
    const len = native.get.call(field).trim().length;
    const tier = len > 84 ? "sm" : len > 60 ? "md" : "";
    if (tier) field.dataset.fit = tier;
    else delete field.dataset.fit;
    if (sizesItself) return;
    // Browsers without field-sizing: measure the wrapped text (skipped while Home is hidden).
    field.style.height = "auto";
    field.style.height = field.scrollHeight ? `${field.scrollHeight}px` : "";
  };

  Object.defineProperty(field, "value", {
    configurable: true,
    get: () => native.get.call(field),
    set: (next) => {
      native.set.call(field, String(next ?? "").replace(/\s*[\r\n]+\s*/g, " "));
      fit();
    },
  });

  field.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" || ev.isComposing) return;
    ev.preventDefault();
    field.blur();
  });
  field.addEventListener("input", () => {
    const raw = native.get.call(field);
    if (/[\r\n]/.test(raw)) {
      const caret = field.selectionStart;
      field.value = raw;
      field.setSelectionRange(caret, caret);
    } else {
      fit();
    }
  });
  if (!sizesItself && typeof ResizeObserver === "function") {
    new ResizeObserver(fit).observe(field.parentElement || field);
  }
  fit();
}

function initEvents() {
  el("startCaptureBtn").onclick = startCapture;
  el("setupStartBtn").onclick = startCapture;
  el("stopCaptureBtn").onclick = () => stopCaptureSession({ flush: true });

  el("setupSimulateBtn").onclick = () => {
    state.forceWorkspace = true;
    state.simulatorOpen = true;
    el("manualForm").classList.remove("hidden");
    el("toggleSimulatorLabel").textContent = "Stop typing turns by hand";
    renderWorkspaceMode();
    el("manualTextInput").focus();
  };

  // Sync Setup Launchpad inputs <-> Topbar Popover inputs (only mutate Copilot config in Copilot mode)
  bindRoleHeadline(el("setupRoleInput"));
  el("setupRoleInput").oninput = () => {
    if (practiceStudio?.isHomePracticeMode?.()) return;
    el("roleInput").value = el("setupRoleInput").value;
    updateContextSummaryChip();
  };
  el("setupRoleInput").onchange = () => {
    if (practiceStudio?.isHomePracticeMode?.()) return;
    pushSessionConfig();
  };

  el("roleInput").oninput = () => {
    el("setupRoleInput").value = el("roleInput").value;
    updateContextSummaryChip();
  };
  el("roleInput").onchange = pushSessionConfig;

  el("setupRubricInput").onchange = () => {
    el("rubricInput").value = el("setupRubricInput").value;
    pushSessionConfig();
  };
  el("rubricInput").onchange = () => {
    el("setupRubricInput").value = el("rubricInput").value;
    pushSessionConfig();
  };

  el("setupTypePills").onclick = (ev) => {
    const btn = ev.target.closest("[data-setup-type]");
    if (!btn) return;
    if (practiceStudio?.isHomePracticeMode?.()) {
      for (const opt of document.querySelectorAll("#setupTypePills [data-setup-type]")) {
        opt.classList.toggle("active", opt === btn);
        opt.setAttribute("aria-pressed", opt === btn ? "true" : "false");
      }
      practiceStudio.refreshQuestionPicker?.();
      return;
    }
    el("typeSelect").value = btn.getAttribute("data-setup-type");
    pushSessionConfig();
  };

  el("setupModePills").onclick = (ev) => {
    const btn = ev.target.closest("[data-setup-mode]");
    if (!btn) return;
    el("modeSelect").value = btn.getAttribute("data-setup-mode");
    renderSpeakerToggle();
    pushSessionConfig();
  };

  el("saveAudioCheckbox").onchange = () => {
    el("setupSaveAudioProxy").checked = el("saveAudioCheckbox").checked;
    pushSessionConfig();
  };
  el("setupSaveAudioProxy").onchange = () => {
    el("saveAudioCheckbox").checked = el("setupSaveAudioProxy").checked;
    pushSessionConfig();
  };

  // Popover toggles
  el("sessionContextBtn").onclick = (ev) => {
    ev.stopPropagation();
    const pop = el("sessionSetupPopover");
    const willOpen = pop.classList.contains("hidden");
    closeAllPopovers();
    if (willOpen) {
      pop.classList.remove("hidden");
      el("sessionContextBtn").setAttribute("aria-expanded", "true");
    }
  };

  el("transcriptMenuBtn").onclick = (ev) => {
    ev.stopPropagation();
    const pop = el("transcriptMenuPopover");
    const willOpen = pop.classList.contains("hidden");
    closeAllPopovers();
    if (willOpen) {
      pop.classList.remove("hidden");
      el("transcriptMenuBtn").setAttribute("aria-expanded", "true");
    }
  };

  document.addEventListener("click", (ev) => {
    if (!ev.target.closest(".popover-anchor")) {
      closeAllPopovers();
    }
  });

  el("modeSelect").onchange = () => {
    renderSpeakerToggle();
    pushSessionConfig();
  };
  el("typeSelect").onchange = pushSessionConfig;

  el("speakerToggleBtn").onclick = () => {
    const next = state.manualSpeaker === "interviewer" ? "candidate" : "interviewer";
    sendCmd({ type: "set_manual_speaker", speaker: next });
  };

  const triggerSwapAll = () => {
    closeAllPopovers();
    sendCmd({ type: "swap_all_speakers" });
  };
  el("swapAllSpeakersBtn").onclick = triggerSwapAll;
  el("menuSwapSpeakersBtn").onclick = triggerSwapAll;

  el("newSessionBtn").onclick = async () => {
    if (state.startingNewSession) return;
    state.startingNewSession = true;
    try {
      closeAllPopovers();
      state.pendingNewSession = true;
      state.forceWorkspace = false;
      state.activeCitationUttId = null;
      if (state.captureState === "streaming") {
        await stopCaptureSession({ flush: true });
      }
      state.selectedQaId = null;
      state.baseElapsedMs = 0;
      state.localSegmentStartedAt = null;
      state.editingSelfRating = false;
      state.searchQuery = "";
      el("transcriptSearchInput").value = "";
      renderTimerPill(0);
      activateRightTab("probes");
      sendCmd({ type: "new_session", config: currentConfigPayload() });
    } finally {
      state.startingNewSession = false;
    }
  };

  el("newCardBtn").onclick = () => sendCmd({ type: "force_new_card" });
  el("regroupBtn").onclick = () => {
    closeAllPopovers();
    sendCmd({ type: "regroup_cards" });
  };

  el("toggleSearchBtn").onclick = () => {
    const wrap = el("searchWrap");
    const opening = wrap.classList.contains("hidden");
    wrap.classList.toggle("hidden", !opening);
    if (opening) {
      el("transcriptSearchInput").focus();
    } else {
      state.searchQuery = "";
      el("transcriptSearchInput").value = "";
      renderQAList();
    }
  };
  el("transcriptSearchInput").oninput = (ev) => {
    state.searchQuery = ev.target.value;
    renderQAList();
  };

  el("phaseNavStrip").onclick = (ev) => {
    const btn = ev.target.closest("[data-jump-qa]");
    if (!btn) return;
    selectAndJumpToQa(btn.getAttribute("data-jump-qa"));
  };

  el("qaList").addEventListener("scroll", updateJumpToBottomVisibility, { passive: true });
  el("jumpToBottomBtn").onclick = () => {
    const container = el("qaList");
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  };

  el("toggleSimulatorBtn").onclick = () => {
    closeAllPopovers();
    state.simulatorOpen = !state.simulatorOpen;
    el("manualForm").classList.toggle("hidden", !state.simulatorOpen);
    el("toggleSimulatorLabel").textContent = state.simulatorOpen
      ? "Stop typing turns by hand"
      : "Type turns by hand";
    renderQAList();
    if (state.simulatorOpen) {
      el("manualTextInput").focus();
    }
  };

  el("toggleWideCalibBtn").onclick = () => {
    state.wideCopilot = !state.wideCopilot;
    el("workspaceGrid").classList.toggle("wide-copilot", state.wideCopilot);
    const hint = state.wideCopilot ? "Narrow this pane" : "Widen this pane";
    el("toggleWideCalibBtn").title = hint;
    el("toggleWideCalibBtn").setAttribute("aria-label", hint);
  };

  el("diveDeeperBtn").onclick = () => {
    if (!state.selectedQaId) return;
    const qa = (state.session?.qa_groups_resolved || []).find((q) => q.id === state.selectedQaId);
    const hasProbes = Boolean(qa?.dive_deeper_history?.length);
    sendCmd({ type: "dive_deeper", qa_id: state.selectedQaId, force_refresh: hasProbes });
  };

  el("probesContent").onclick = async (ev) => {
    if (ev.target.closest("[data-trigger-dive]")) {
      if (state.selectedQaId) {
        sendCmd({ type: "dive_deeper", qa_id: state.selectedQaId, force_refresh: false });
      }
      return;
    }
    const copyBtn = ev.target.closest("[data-probe-copy]");
    if (copyBtn) {
      try {
        await navigator.clipboard.writeText(copyBtn.getAttribute("data-probe-copy") || "");
        copyBtn.classList.add("is-copied");
        copyBtn.innerHTML = ICON_CHECK;
        setTimeout(() => {
          copyBtn.classList.remove("is-copied");
          copyBtn.innerHTML = ICON_COPY;
        }, 1200);
      } catch (_) {
        showToast("Could not copy to the clipboard.");
      }
      return;
    }
    // The whole question is the control: click it once you've asked it.
    const probe = ev.target.closest("[data-probe-asked]");
    if (probe) {
      toggleProbeAsked(probe.getAttribute("data-probe-asked"));
    }
  };

  el("probesContent").addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" && ev.key !== " ") return;
    const probe = ev.target.closest?.("[data-probe-asked]");
    if (probe && ev.target === probe) {
      ev.preventDefault();
      toggleProbeAsked(probe.getAttribute("data-probe-asked"));
    }
  });

  el("hireSignalPills").onclick = (ev) => {
    const btn = ev.target.closest("[data-signal]");
    if (!btn) return;
    setHireSignalValue(btn.getAttribute("data-signal"));
  };

  el("editSelfRatingBtn").onclick = () => {
    state.editingSelfRating = true;
    renderCalibrationPanel();
  };
  el("collapseSelfRatingBtn").onclick = () => {
    state.editingSelfRating = false;
    renderCalibrationPanel();
  };

  el("runCalibrationBtn").onclick = () => {
    const rating = {
      hire_signal: el("selfHireSignal").value,
      perceived_level: el("selfLevelInput").value.trim(),
      gut_rationale: el("selfRationaleInput").value.trim(),
      biggest_concern: el("selfConcernInput").value.trim(),
    };
    state.editingSelfRating = false;
    sendCmd({
      type: "generate_calibration",
      rating,
      regroup_first: true,
    });
  };

  el("copyCalibrationBtn").onclick = async () => {
    const md = state.session?.calibration_markdown;
    if (!md) return;
    try {
      await navigator.clipboard.writeText(md);
      const btn = el("copyCalibrationBtn");
      btn.textContent = "Copied";
      setTimeout(() => {
        btn.textContent = "Copy as Markdown";
      }, 1400);
    } catch (_) {
      showToast("Could not copy to the clipboard.");
    }
  };

  el("calibrationOutput").onclick = (ev) => {
    const btn = ev.target.closest("[data-cite-utt]");
    if (!btn) return;
    jumpToUtteranceCitation(btn.getAttribute("data-cite-utt"), btn);
  };

  const sectionNav = el("reportSectionNav");
  if (sectionNav) {
    sectionNav.onclick = (ev) => {
      const btn = ev.target.closest("[data-scroll-sec]");
      if (!btn) return;
      const target = document.getElementById(btn.getAttribute("data-scroll-sec"));
      if (target) {
        // Hold the clicked tab active while the smooth scroll settles (scroll-spy resumes afterwards)
        state.navLockUntil = Date.now() + 900;
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      for (const pill of sectionNav.querySelectorAll(".report-nav-link")) {
        const isAct = pill === btn;
        pill.classList.toggle("active", isAct);
        pill.setAttribute("aria-selected", String(isAct));
      }
    };
  }

  el("calibrationView").addEventListener("scroll", syncReportNavToScroll, { passive: true });

  const phaseSignals = el("phaseSignalsStrip");
  if (phaseSignals) {
    phaseSignals.onclick = (ev) => {
      const chip = ev.target.closest("[data-jump-badge]");
      if (!chip) return;
      const badge = chip.getAttribute("data-jump-badge");
      const qas = state.session?.qa_groups_resolved || [];
      const match = qas.find((q) => (q.badge || "").toUpperCase() === (badge || "").toUpperCase());
      if (match) {
        selectAndJumpToQa(match.id);
      }
    };
  }

  const triggerFileDownload = (url) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  el("exportInterviewerMdBtn").onclick = () => {
    closeAllPopovers();
    if (state.session?.id) {
      triggerFileDownload(
        `/api/sessions/${encodeURIComponent(state.session.id)}/markdown?audience=interviewer`
      );
    }
  };
  el("exportCandidateMdBtn").onclick = () => {
    closeAllPopovers();
    if (state.session?.id) {
      triggerFileDownload(
        `/api/sessions/${encodeURIComponent(state.session.id)}/markdown?audience=candidate`
      );
    }
  };

  el("tabProbes").onclick = () => activateRightTab("probes");
  el("tabCalibration").onclick = () => activateRightTab("calibration");

  // Settings Modal
  el("settingsBtn").onclick = openSettingsModal;
  el("closeSettingsBtn").onclick = () => setModalState("none");
  el("cancelSettingsBtn").onclick = () => setModalState("none");
  el("saveSettingsBtn").onclick = saveSettings;
  el("testLlmBtn").onclick = testSettingsConnection;
  el("openDataDirBtn").onclick = openDataDirectory;
  el("llmFastModelInput").oninput = () => syncModelChoosers();
  el("llmCalibModelInput").oninput = () => syncModelChoosers();

  el("settingsModal").onclick = (ev) => {
    if (ev.target === el("settingsModal")) {
      setModalState("none");
      return;
    }
    const chip = ev.target.closest("[data-chip-val], [data-chip-custom]");
    if (chip) {
      const group = chip.closest("[data-target-input]");
      const input = el(group?.getAttribute("data-target-input"));
      if (!input) return;
      if (chip.hasAttribute("data-chip-custom")) {
        // Custom starts from whatever was selected, ready to be edited or replaced.
        group.dataset.custom = "true";
        syncModelChoosers();
        input.focus();
        input.select();
      } else {
        group.dataset.custom = "false";
        input.value = chip.getAttribute("data-chip-val");
        syncModelChoosers();
      }
    }
  };

  // First-Run Onboarding Modal (2-step wizard)
  el("homeSetupAiBtn")?.addEventListener("click", () => openOnboardingModal());
  el("closeOnboardingBtn")?.addEventListener("click", () => setModalState("none"));
  el("onboardTestSaveBtn")?.addEventListener("click", testAndSaveOnboardingKey);
  el("onboardSkipBtn")?.addEventListener("click", skipOnboardingTranscriptOnly);
  el("onboardNextStepBtn")?.addEventListener("click", () => setOnboardStep("packs"));
  el("onboardBackBtn")?.addEventListener("click", () => setOnboardStep("ai"));
  el("onboardDoneBtn")?.addEventListener("click", () => setModalState("none"));
  el("onboardPhononInstallBtn")?.addEventListener("click", () => {
    profileAddons?.installPack("phonon_tts");
  });
  el("onboardPhononPreviewBtn")?.addEventListener("click", () => {
    practiceStudio?.previewVoice?.({ engine: "phonon" });
  });
  el("onboardKokoroInstallBtn")?.addEventListener("click", () => {
    profileAddons?.installPack("kokoro_tts");
  });
  el("onboardKokoroPreviewBtn")?.addEventListener("click", () => {
    practiceStudio?.previewVoice?.({ engine: "kokoro" });
  });
  el("onboardOptionalPacksList")?.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-onboard-install]");
    if (btn) {
      profileAddons?.installPack(btn.getAttribute("data-onboard-install"));
    }
  });
  for (const stepBtn of document.querySelectorAll("[data-onboard-step]")) {
    stepBtn.addEventListener("click", () => {
      setOnboardStep(stepBtn.getAttribute("data-onboard-step"));
    });
  }
  el("onboardApiKeyInput")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      testAndSaveOnboardingKey();
    }
  });
  el("onboardingModal")?.addEventListener("click", (ev) => {
    if (ev.target === el("onboardingModal")) {
      setModalState("none");
    }
  });

  // History Modal
  el("historyBtn").onclick = openHistoryModal;
  el("closeHistoryBtn").onclick = () => setModalState("none");
  el("historyModal").onclick = (ev) => {
    if (ev.target === el("historyModal")) {
      setModalState("none");
    }
  };

  const onSessionListClick = (ev) => {
    const btn = ev.target.closest("[data-load-session]");
    if (btn) loadSavedSession(btn.getAttribute("data-load-session"));
  };
  el("historyList").onclick = onSessionListClick;
  el("homeSessionList").onclick = onSessionListClick;

  const qaList = el("qaList");
  qaList.onclick = (ev) => {
    const actionBtn = ev.target.closest("[data-action]");
    if (actionBtn) {
      ev.stopPropagation();
      const action = actionBtn.getAttribute("data-action");
      const qaId = actionBtn.getAttribute("data-qa");
      if (action === "merge") {
        sendCmd({ type: "merge_card", qa_id: qaId });
      } else if (action === "split") {
        sendCmd({
          type: "split_card",
          qa_id: qaId,
          utterance_id: actionBtn.getAttribute("data-utt"),
        });
      } else if (action === "flip-speaker") {
        sendCmd({
          type: "flip_utterance_speaker",
          utterance_id: actionBtn.getAttribute("data-utt"),
        });
      } else if (action === "accept-review") {
        sendCmd({
          type: "resolve_review",
          review_id: actionBtn.getAttribute("data-review"),
          accept: true,
        });
      } else if (action === "dismiss-review") {
        sendCmd({
          type: "resolve_review",
          review_id: actionBtn.getAttribute("data-review"),
          accept: false,
        });
      }
      return;
    }

    const card = ev.target.closest("[data-qa-card]");
    if (card) {
      const qaId = card.getAttribute("data-qa-card");
      if (state.selectedQaId !== qaId) {
        state.selectedQaId = qaId;
        renderQAList();
        renderProbesPanel();
      }
    }
  };

  qaList.onchange = (ev) => {
    const titleInput = ev.target.closest("[data-qa-title]");
    if (titleInput) {
      sendCmd({
        type: "update_qa_meta",
        qa_id: titleInput.getAttribute("data-qa-title"),
        title: titleInput.value,
      });
      return;
    }
    const noteInput = ev.target.closest("[data-qa-note]");
    if (noteInput) {
      sendCmd({
        type: "update_qa_meta",
        qa_id: noteInput.getAttribute("data-qa-note"),
        notes: noteInput.value,
      });
    }
  };

  qaList.addEventListener("focusout", () => {
    if (!state.pendingQARerender) return;
    setTimeout(() => {
      const activeEl = document.activeElement;
      const stillEditing =
        activeEl &&
        qaList.contains(activeEl) &&
        (activeEl.classList.contains("qa-title-input") ||
          activeEl.classList.contains("qa-note-input"));
      if (!stillEditing && state.pendingQARerender) {
        renderQAList();
      }
    }, 0);
  });

  el("manualForm").onsubmit = (ev) => {
    ev.preventDefault();
    const textInput = el("manualTextInput");
    const text = textInput.value.trim();
    if (!text) return;
    const speaker = el("manualRoleSelect").value;
    const utts = state.session?.utterances || [];
    const lastMs = utts.reduce((m, u) => Math.max(m, u.end_ms || 0), 0);
    sendCmd({
      type: "inject_utterance",
      speaker,
      text,
      start_ms: lastMs + 1000,
      end_ms: lastMs + 4500,
    });
    textInput.value = "";
    el("manualRoleSelect").value = speaker === "interviewer" ? "candidate" : "interviewer";
  };

  window.addEventListener("beforeunload", (ev) => {
    const unsavedPractice = Boolean(practiceStudio?.hasUnsavedWork?.());
    if (state.captureState === "streaming" || unsavedPractice) {
      ev.preventDefault();
      ev.returnValue = "";
    }
  });

  window.addEventListener("keydown", (ev) => {
    if (state.modalState !== "none" && ev.key === "Tab") {
      const root = _modalRootFor(state.modalState);
      const focusables = _focusableIn(root);
      if (focusables.length > 0) {
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (ev.shiftKey) {
          if (document.activeElement === first || !root.contains(document.activeElement)) {
            ev.preventDefault();
            last.focus();
          }
        } else if (document.activeElement === last || !root.contains(document.activeElement)) {
          ev.preventDefault();
          first.focus();
        }
      }
      return;
    }

    if (ev.key === "Escape") {
      if (state.modalState !== "none") {
        ev.preventDefault();
        setModalState("none");
        return;
      }
      if (state.practiceActive && practiceStudio?.handleEscape?.()) {
        ev.preventDefault();
        return;
      }
      closeAllPopovers();
      return;
    }

    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter" && state.modalState === "settings") {
      ev.preventDefault();
      saveSettings();
      return;
    }

    const tag = (ev.target?.tagName || "").toUpperCase();
    const isTyping =
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      Boolean(ev.target?.isContentEditable);
    if (isTyping || state.modalState !== "none") return;

    if (state.practiceActive) {
      if (
        (ev.key === "m" || ev.key === "M") &&
        !ev.repeat &&
        !ev.metaKey &&
        !ev.ctrlKey &&
        !ev.altKey &&
        tag !== "BUTTON" &&
        tag !== "A"
      ) {
        if (practiceStudio?.toggleVoiceShortcut?.()) {
          ev.preventDefault();
        }
      }
      return;
    }

    const copilotWorkspaceVisible =
      el("workspaceGrid") && !el("workspaceGrid").classList.contains("hidden");
    if (!copilotWorkspaceVisible) return;

    if ((ev.key === "d" || ev.key === "D") && !ev.repeat && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      ev.preventDefault();
      const qas = state.session?.qa_groups_resolved || [];
      const selectedQa = qas.find((q) => q.id === state.selectedQaId);
      if (
        selectedQa &&
        selectedQa.probe_state !== "empty" &&
        selectedQa.probe_state !== "loading" &&
        allowedEventsSet().has("dive_deeper")
      ) {
        sendCmd({ type: "dive_deeper", qa_id: state.selectedQaId, force_refresh: false });
      }
    } else if (/^[1-9]$/.test(ev.key) && !ev.metaKey && !ev.ctrlKey && !ev.altKey && state.rightTab === "probes") {
      const key = probeKeyForNumber(ev.key);
      if (key) {
        ev.preventDefault();
        toggleProbeAsked(key);
      }
    } else if ((ev.key === "n" || ev.key === "N") && !ev.repeat && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      ev.preventDefault();
      if (allowedEventsSet().has("force_new_card")) {
        sendCmd({ type: "force_new_card" });
      }
    } else if (ev.key === "Tab" && !ev.shiftKey && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      const manualMode = el("modeSelect").value === "in_person" && !isAutoDiarizerActive();
      if (state.captureState === "streaming" && manualMode) {
        ev.preventDefault();
        const next = state.manualSpeaker === "interviewer" ? "candidate" : "interviewer";
        sendCmd({ type: "set_manual_speaker", speaker: next });
      }
    }
  });
}

initTheme();
initEvents();
profileAddons = createProfileAddonsController({
  state,
  el,
  escapeHtml,
  showToast,
  setModalState,
  clientMachines,
  packOps,
  onPracticeQuestion: async (q) => {
    const ok = await practiceStudio?.selectBankQuestion(q);
    if (ok) setModalState("none");
    return ok;
  },
  onRoleApply: (jd) => {
    if (!jd?.title) return;
    const roleLabel = jd.company ? `${jd.title} (${jd.company})` : jd.title;
    el("setupRoleInput").value = roleLabel;
    el("roleInput").value = roleLabel;
    if (el("contextRoleSummary")) {
      el("contextRoleSummary").textContent = roleLabel;
    }
    pushSessionConfig();
    // The level pill follows the JD too; setTargetLevel refreshes the question picker itself.
    if (!practiceStudio?.setTargetLevel(jd.target_level)) {
      practiceStudio?.refreshQuestionPicker();
    }
  },
  onLevelApply: (level, opts) => practiceStudio?.setTargetLevel(level, opts),
  isPracticeHome: () => Boolean(practiceStudio?.isHomePracticeMode?.()),
  onAddonsChange: (addons, tts = null) => {
    state.addons = addons;
    if (tts) state.tts = tts;
    practiceStudio?.applyStatusSnapshot(tts ? { addons, tts } : { addons });
    if (state.modalState === "onboarding" && state.onboardStep === "packs") {
      renderOnboardingPacksStep();
    }
  },
});
practiceStudio = createPracticeStudioController({
  state,
  el,
  escapeHtml,
  showToast,
  clientMachines,
  packOps,
  onInstallPack: (packId) => profileAddons?.installPack(packId),
  onPracticeActiveChange: () => {
    renderWorkspaceMode();
  },
  onHomeModeChange: () => profileAddons?.syncHomeJdTarget(),
});
packOps.subscribe(() => {
  if (state.modalState === "onboarding" && state.onboardStep === "packs") {
    renderOnboardingPacksStep();
  }
});
practiceStudio.registerStudioPlugin(
  "system_design",
  createSystemDesignStudioPlugin({ escapeHtml, showToast, clientMachines })
);
practiceStudio.registerStudioPlugin(
  "coding",
  createCodingStudioPlugin({
    escapeHtml,
    showToast,
    clientMachines,
    packOps,
    getAddons: () => state.addons || profileAddons.getAddons(),
    onAddonsUpdated: (addons) => {
      state.addons = addons;
      profileAddons.applyStatusSnapshot({ addons });
      practiceStudio.applyStatusSnapshot({ addons });
    },
  })
);
renderSpeakerToggle();
renderCaptureControls();
connectWS();
checkInitialOnboarding();
