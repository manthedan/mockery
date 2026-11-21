const CHANNEL_INTERVIEWER = 0x01;
const CHANNEL_CANDIDATE = 0x02;
const CHANNEL_ROOM_MIC = 0x03;

const CONSENT_SCRIPT =
  "Quick heads-up before we jump in: I'm running a local mock-interview helper on my Mac to organize our Q&A transcript and generate coaching notes afterward. Is it okay with you if I transcribe our session? I'll share a clean copy of the transcript with you right after!";

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

const PHASE_LABELS = {
  setup: "Phase: Setup",
  ready: "Phase: Ready",
  capturing: "● Live Capture",
  awaiting_self_rating: "Phase: Self-Rate",
  rated: "Phase: Rated",
  calibrating: "Phase: Calibrating...",
  calibrated: "Phase: Calibrated",
};

const state = {
  wsState: "connecting", // connecting | connected | reconnecting
  captureState: "idle", // idle | requesting_permissions | streaming | stopping
  modalState: "none", // none | settings | history
  rightTab: "probes", // probes | calibration

  ws: null,
  outbox: [],
  session: null,
  selectedQaId: null,
  asr: { state: "loading", generation: 0, last_inference_ms: 0 },
  llm: null,
  manualSpeaker: "interviewer",

  baseElapsedMs: 0,
  localSegmentStartedAt: null,
  timerHandle: null,
  stopFallbackTimer: null,
  toastTimer: null,
  pendingNewSession: false,
  pendingQARerender: false,

  sharedAudioCtx: null,
  audioCleanupFns: [],
  lastSyncedSessionId: null,
};

// Expose read-only state snapshot for Agent Experience (AX) & DevTools inspection
Object.defineProperty(window, "__MOCK_HELPER_STATE__", {
  get() {
    return {
      wsState: state.wsState,
      captureState: state.captureState,
      modalState: state.modalState,
      rightTab: state.rightTab,
      selectedQaId: state.selectedQaId,
      asr: state.asr,
      serverMachines: state.session?.state_machines || null,
    };
  },
});

const el = (id) => document.getElementById(id);

function setWsState(next) {
  if (next === state.wsState) return;
  if (!WS_TRANSITIONS[state.wsState]?.has(next)) {
    console.warn(`Unexpected WS transition: ${state.wsState} -> ${next}`);
  }
  const prev = state.wsState;
  state.wsState = next;
  renderConnectionIndicator();
  if (next === "reconnecting" && prev === "connected") {
    showToast("Connection to local server lost — reconnecting...", 3000);
  } else if (next === "connected" && prev !== "connecting") {
    showToast("Reconnected to Mock Helper server.", 2000);
  }
}

function setCaptureState(next) {
  if (next === state.captureState) return;
  if (!CAPTURE_TRANSITIONS[state.captureState]?.has(next)) {
    throw new Error(`Illegal client capture transition: ${state.captureState} -> ${next}`);
  }
  state.captureState = next;
  renderCaptureControls();
}

function setModalState(next) {
  state.modalState = next;
  el("settingsModal").classList.toggle("hidden", next !== "settings");
  el("historyModal").classList.toggle("hidden", next !== "history");
}

function showToast(message, durationMs = 4200) {
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

function allowedEventsSet() {
  return new Set(state.session?.state_machines?.session?.allowed_events || []);
}

/* ---------------- WebSocket Transport State Machine ---------------- */

function connectWS() {
  setWsState(state.wsState === "reconnecting" ? "connecting" : state.wsState);
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/session`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = () => {
    setWsState("connected");
    while (state.outbox.length > 0 && ws.readyState === WebSocket.OPEN) {
      ws.send(state.outbox.shift());
    }
    if (state.captureState === "streaming") {
      sendCmd({ type: "start_capture" });
    }
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    handleServerMessage(msg);
  };

  ws.onclose = () => {
    setWsState("reconnecting");
    setTimeout(connectWS, 1400);
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

  if (msg.type === "transition_rejected") {
    showToast(`${msg.remediation || "Action unavailable in current state."}`);
    return;
  }

  if (msg.type === "error") {
    showToast(msg.message || "Error");
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
    if (state.session?.state_machines) {
      state.session.state_machines.jobs.calibration = "running";
      state.session.state_machines.session.phase = "calibrating";
      renderPhaseAndControls();
      renderCalibrationPanel();
    }
    activateRightTab("calibration");
    return;
  }

  if (msg.type === "calibration_ready") {
    if (state.session) {
      state.session.calibration_markdown = msg.markdown;
    }
    renderCalibrationPanel();
    return;
  }

  if (msg.type === "session_state") {
    const prevSelected = state.selectedQaId;
    const prevSessionId = state.session?.id;
    state.session = msg.session;
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
    }

    // Sync cumulative elapsed timer from server state when not mid-local tick
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
      showToast("Cards regrouped cleanly across full utterance history.", 2500);
    }

    syncConfigInputsFromSession();
    renderPhaseAndControls();
    renderQAList();
    renderProbesPanel();
    renderCalibrationPanel();
    renderMetricsBar();
    renderSpeakerToggle();
  }
}

/* ---------------- Shared-Clock Web Audio Capture ---------------- */

async function ensureSharedAudioContext() {
  if (!state.sharedAudioCtx || state.sharedAudioCtx.state === "closed") {
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule("/static/pcm-worklet.js");
    state.sharedAudioCtx = ctx;
  }
  if (state.sharedAudioCtx.state === "suspended") {
    await state.sharedAudioCtx.resume();
  }
  return state.sharedAudioCtx;
}

async function startCapture() {
  if (state.captureState !== "idle") return;

  if (!el("consentCheckbox").checked) {
    showToast(
      "Confirm upfront candidate consent first (check 'Candidate consent confirmed' above—use 'Copy Verbal Consent Prompt' if helpful)."
    );
    return;
  }

  const mode = el("modeSelect").value;
  pushSessionConfig();
  setCaptureState("requesting_permissions");

  try {
    let streamsToAttach;
    if (mode === "remote_meet") {
      streamsToAttach = await acquireRemoteMeetStreams();
    } else {
      streamsToAttach = await acquireSingleMicStreams();
    }

    // Notify server BEFORE worklets start streaming PCM frames so consent & phase transition lead
    sendCmd({ type: "start_capture" });
    state.localSegmentStartedAt = Date.now();
    setCaptureState("streaming");

    for (const item of streamsToAttach) {
      attachStreamToSharedContext(item.ctx, item.stream, item.channelId, item.meterId);
    }

    if (state.timerHandle) clearInterval(state.timerHandle);
    state.timerHandle = setInterval(updateTimer, 500);
  } catch (err) {
    stopAudioStreamsOnly();
    setCaptureState("idle");
    sendCmd({ type: "cancel_capture" });
    showToast(`Could not start capture: ${err.message || err}`, 5500);
  }
}

async function acquireRemoteMeetStreams() {
  // Invoke getDisplayMedia FIRST while the button click's transient user activation is fresh!
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
        ? "Tab selection cancelled."
        : err?.message || String(err);
    throw new Error(`Google Meet tab capture failed: ${detail}`);
  }

  tabStream.getVideoTracks().forEach((t) => t.stop());

  const tabAudioTracks = tabStream.getAudioTracks();
  if (tabAudioTracks.length === 0) {
    tabStream.getTracks().forEach((t) => t.stop());
    throw new Error(
      "No tab audio track shared! Select your Google Meet Chrome Tab and keep 'Share tab audio' checked."
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

  worklet.port.onmessage = (ev) => {
    const { pcmBuffer, rms } = ev.data;
    sendBinaryPcm(channelId, pcmBuffer);
    const pct = Math.min(100, Math.round(Math.pow(rms * 6.5, 0.65) * 100));
    const meterEl = el(meterElementId);
    if (meterEl) meterEl.style.width = `${pct}%`;
  };

  state.audioCleanupFns.push(() => {
    worklet.disconnect();
    source.disconnect();
    silentGain.disconnect();
    mediaStream.getTracks().forEach((t) => t.stop());
  });
}

function stopAudioStreamsOnly() {
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

function stopCaptureSession({ flush = true } = {}) {
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

function renderTimerPill(elapsedSec) {
  const pill = el("timerPill");
  pill.textContent = `${fmtClock(elapsedSec)} / 45:00`;
  pill.classList.toggle("warn", elapsedSec >= 38 * 60);
}

/* ---------------- State-Driven UI Rendering ---------------- */

function renderConnectionIndicator() {
  const dot = el("brandDot");
  dot.title = `WebSocket: ${state.wsState}`;
  dot.classList.toggle("disconnected", state.wsState !== "connected");
}

function renderCaptureControls() {
  const startBtn = el("startCaptureBtn");
  const stopBtn = el("stopCaptureBtn");
  const dot = el("brandDot");
  const cs = state.captureState;
  const phase = state.session?.state_machines?.session?.phase || "setup";

  dot.classList.toggle("recording", cs === "streaming");
  el("modeSelect").disabled = cs !== "idle";

  if (cs === "streaming") {
    startBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = false;
    stopBtn.textContent = "Stop & Calibrate";
  } else if (cs === "requesting_permissions") {
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    startBtn.disabled = true;
    startBtn.textContent = "Select Tab & Mic...";
  } else if (cs === "stopping") {
    startBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = true;
    stopBtn.textContent = "Flushing audio...";
  } else {
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    startBtn.disabled = phase === "calibrating";
    const isResumable =
      phase === "awaiting_self_rating" || phase === "rated" || phase === "calibrated";
    startBtn.textContent = isResumable ? "Resume Capture" : "Start Live Capture";
  }
}

function renderPhaseAndControls() {
  const sm = state.session?.state_machines;
  if (!sm) return;

  const phase = sm.session?.phase || "setup";
  const allowed = allowedEventsSet();
  const blocked = sm.session?.blocked_reasons || {};
  const regroupRunning = sm.jobs?.regroup === "running";

  const phaseBadge = el("phaseBadge");
  phaseBadge.textContent = PHASE_LABELS[phase] || `Phase: ${phase}`;
  phaseBadge.className = `status-badge ${
    phase === "capturing" || phase === "calibrated" ? "ready" : phase === "calibrating" ? "loading" : "phase-active"
  }`;

  const regroupBtn = el("regroupBtn");
  regroupBtn.disabled = !allowed.has("regroup_cards");
  regroupBtn.textContent = regroupRunning ? "Regrouping..." : "Regroup";
  regroupBtn.title = blocked.regroup_cards || "Clean up card boundaries across full utterance log";

  const newCardBtn = el("newCardBtn");
  newCardBtn.disabled = !allowed.has("force_new_card");

  renderCaptureControls();
}

function renderAsrBadge() {
  const badge = el("asrBadge");
  const st = state.asr.state;
  const shortModel = (state.asr.model_id || "").endsWith("-v2") ? "v2-en" : "v3";
  badge.className = `status-badge ${st === "ready" ? "ready" : "loading"}`;
  if (st === "ready") {
    const lat = state.asr.last_inference_ms ? ` · ${state.asr.last_inference_ms}ms` : "";
    badge.textContent = `Parakeet ${shortModel} · Ready${lat}`;
  } else if (st === "switching") {
    badge.textContent = `Parakeet · Switching to ${shortModel}...`;
  } else if (st === "loading") {
    badge.textContent = `Parakeet ${shortModel} · Warming up...`;
  } else {
    badge.textContent = "ASR Error";
    badge.title = state.asr.error || "";
  }
}

function renderSpeakerToggle() {
  const mode = el("modeSelect").value;
  const btn = el("speakerToggleBtn");
  const candMeter = el("candMeterWrap");
  if (mode === "in_person") {
    btn.classList.remove("hidden");
    candMeter.classList.add("hidden");
    el("manualSpeakerLabel").textContent =
      state.manualSpeaker === "interviewer" ? "You (Interviewer)" : "Candidate";
  } else {
    btn.classList.add("hidden");
    candMeter.classList.remove("hidden");
  }
}

function syncConfigInputsFromSession() {
  if (!state.session) return;
  const cfg = state.session.config;
  if (document.activeElement !== el("roleInput")) {
    el("roleInput").value = cfg.role_title;
  }
  if (document.activeElement !== el("typeSelect")) {
    el("typeSelect").value = cfg.interview_type;
  }
  if (document.activeElement !== el("rubricInput")) {
    el("rubricInput").value = cfg.rubric_notes || "";
  }
  el("consentCheckbox").checked = Boolean(cfg.consent_confirmed);
  el("saveAudioCheckbox").checked = Boolean(cfg.save_raw_audio);
  if (state.captureState === "idle" && document.activeElement !== el("modeSelect")) {
    el("modeSelect").value = cfg.mode;
  }

  const isPhase = cfg.interview_type === "system_design" || cfg.interview_type === "coding";
  el("leftPaneTitle").textContent = isPhase ? "Live Phase Cards" : "Live Q&A Cards";

  const sessionChanged = state.lastSyncedSessionId !== state.session.id;
  state.lastSyncedSessionId = state.session.id;

  if (state.session.self_rating) {
    const sr = state.session.self_rating;
    if (document.activeElement !== el("selfHireSignal")) el("selfHireSignal").value = sr.hire_signal;
    if (document.activeElement !== el("selfLevelInput")) el("selfLevelInput").value = sr.perceived_level || "";
    if (document.activeElement !== el("selfRationaleInput")) el("selfRationaleInput").value = sr.gut_rationale || "";
    if (document.activeElement !== el("selfConcernInput")) el("selfConcernInput").value = sr.biggest_concern || "";
  } else if (sessionChanged) {
    el("selfHireSignal").value = "lean_hire";
    el("selfLevelInput").value = "";
    el("selfRationaleInput").value = "";
    el("selfConcernInput").value = "";
    el("calibrationOutput").textContent =
      "Lock in your gut rating above and click 'Submit My Call & Reveal AI Audit'.";
  }
}

function renderMetricsBar() {
  const m = state.session?.metrics;
  if (!m || (m.interviewer_words === 0 && m.candidate_words === 0)) {
    el("ratioPill").textContent = "Talk Ratio: —";
    return;
  }
  const bleedHint = m.echo_bleed_dropped ? ` · ${m.echo_bleed_dropped} echo dropped` : "";
  el("ratioPill").textContent = `Time: You ${m.interviewer_time_pct}% · Cand ${m.candidate_time_pct}%${bleedHint}`;
}

function renderCardProbePill(probeState) {
  if (probeState === "ready_cached") {
    return `<span class="cached-pill" title="Probes pre-computed and current—0ms reveal">⚡ Instant</span>`;
  }
  if (probeState === "warming" || probeState === "loading") {
    return `<span class="warming-pill">⏳ Thinking...</span>`;
  }
  if (probeState === "stale_cached" || probeState === "revealed_stale") {
    return `<span class="stale-pill" title="New turns arrived since last probe">↻ Newer turns</span>`;
  }
  return "";
}

function renderPendingReviewsHtml(uttId) {
  const reviews = (state.session?.pending_reviews || []).filter((r) => r.utterance_id === uttId);
  if (!reviews.length) return "";
  return reviews
    .map((r) => {
      let label = "Suggested edit";
      if (r.kind === "flip_speaker") {
        const target = r.suggested_speaker === "interviewer" ? "YOU" : "CANDIDATE";
        label = `Flip speaker → ${target}`;
      } else if (r.kind === "split_mixed_turn") {
        const segCount = (r.suggested_segments || []).length || 2;
        label = `Split handoff (${segCount} speakers)`;
      } else if (r.kind === "split_card") {
        label = r.suggested_title ? `New card: “${r.suggested_title}”` : "Split into new card";
      }
      const rationale = r.rationale ? ` — ${r.rationale}` : "";
      return `
        <div class="review-chip" data-review-chip="${escapeHtml(r.id)}">
          <span class="review-chip-text">
            <strong>[${escapeHtml(r.confidence)}]</strong> ${escapeHtml(label)}${escapeHtml(rationale)}
          </span>
          <span class="review-chip-actions">
            <button type="button" class="btn btn-sm btn-accept-review" data-action="accept-review" data-review="${escapeHtml(r.id)}">Accept</button>
            <button type="button" class="btn btn-sm" data-action="dismiss-review" data-review="${escapeHtml(r.id)}">Dismiss</button>
          </span>
        </div>
      `;
    })
    .join("");
}

function renderWarmupRowsHtml(warmup) {
  return warmup
    .map(
      (u) => `
      <div class="utterance-row ${u.speaker} ${u.final ? "" : "draft-utt"}" data-utt-row="${escapeHtml(u.id)}">
        <div class="utt-meta">
          <span class="${u.speaker === "interviewer" ? "role-you" : "role-cand"}">
            <button type="button" class="role-flip-btn ${u.speaker === "interviewer" ? "role-you" : "role-cand"}" data-action="flip-speaker" data-utt="${escapeHtml(u.id)}" title="Click to flip speaker (You ↔ Candidate)">
              ${u.speaker === "interviewer" ? "YOU" : "CANDIDATE"}
            </button>
            · ${fmtMs(u.start_ms)} · <code>${escapeHtml(u.id)}</code>
            ${u.final ? "" : `<span class="draft-pill">live r${u.rev}</span>`}
          </span>
        </div>
        <div>${escapeHtml(u.text)}</div>
        ${renderPendingReviewsHtml(u.id)}
      </div>`
    )
    .join("");
}

function renderCardUtterancesHtml(qa, canSplitAny) {
  const utts = qa.utterances || [];
  if (!utts.length) {
    return `<div style="color:var(--text-dim); font-size:12.5px;">Waiting for next utterance...</div>`;
  }
  return utts
    .map((u, j) => {
      const canSplit = j > 0 && canSplitAny;
      return `
        <div class="utterance-row ${u.speaker} ${u.final ? "" : "draft-utt"}" data-utt-row="${escapeHtml(u.id)}">
          <div class="utt-meta">
            <span class="${u.speaker === "interviewer" ? "role-you" : "role-cand"}">
              <button type="button" class="role-flip-btn ${u.speaker === "interviewer" ? "role-you" : "role-cand"}" data-action="flip-speaker" data-utt="${escapeHtml(u.id)}" title="Click to flip speaker (You ↔ Candidate)">
                ${u.speaker === "interviewer" ? "YOU" : "CANDIDATE"}
              </button>
              · ${fmtMs(u.start_ms)} · <code>${escapeHtml(u.id)}</code>
              ${u.final ? "" : `<span class="draft-pill">live r${u.rev}</span>`}
            </span>
            ${
              canSplit
                ? `<button class="split-btn" data-action="split" data-qa="${qa.id}" data-utt="${u.id}" title="Split from this utterance into a new card">Split card ↵</button>`
                : ""
            }
          </div>
          <div>${escapeHtml(u.text)}</div>
          ${renderPendingReviewsHtml(u.id)}
        </div>
      `;
    })
    .join("");
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

  const activeEl = document.activeElement;
  if (
    activeEl &&
    container.contains(activeEl) &&
    (activeEl.classList.contains("qa-title-input") || activeEl.classList.contains("qa-note-input"))
  ) {
    // Keep live transcript flowing inside existing cards without disturbing focused inputs
    state.pendingQARerender = true;
    const warmList = container.querySelector('[data-warmup-list="true"]');
    if (warmList && warmup.length > 0) {
      warmList.innerHTML = renderWarmupRowsHtml(warmup);
    }
    for (const qa of qas) {
      const cardEl = container.querySelector(`[data-qa-card="${CSS.escape(qa.id)}"]`);
      if (!cardEl) continue;
      const uttListEl = cardEl.querySelector(".utterance-list");
      if (uttListEl) {
        uttListEl.innerHTML = renderCardUtterancesHtml(qa, canSplitAny);
      }
      const pillHost = cardEl.querySelector(".qa-probe-pill-host");
      if (pillHost) {
        pillHost.innerHTML = renderCardProbePill(qa.probe_state || "idle");
      }
    }
    return;
  }

  state.pendingQARerender = false;
  const prevScrollTop = container.scrollTop;
  const hadContent = container.querySelector(".qa-card") !== null;
  const wasNearBottom =
    !hadContent || container.scrollHeight - container.scrollTop - container.clientHeight < 48;

  if (qas.length === 0 && warmup.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>Ready for Your 45-Minute Mock Interview</h3>
        <ol>
          <li>Tell the candidate up front &amp; check <strong>Candidate consent confirmed</strong> above (put on headphones!).</li>
          <li>Click <strong>Start Live Capture</strong>, choose your <strong>Google Meet Chrome Tab</strong>, and check <strong>"Share tab audio"</strong>.</li>
          <li>Press <kbd>D</kbd> anytime—even mid-answer—for <strong>3 follow-up probes</strong>, or <kbd>N</kbd> to split a new card.</li>
        </ol>
      </div>
    `;
    return;
  }

  const parts = [];

  if (warmup.length > 0) {
    parts.push(`
      <div class="qa-card" style="opacity:0.86;">
        <div class="qa-card-header">
          <span class="qa-badge">INTRO</span>
          <span style="font-weight:600; flex:1;">Warm-up / Audio Check</span>
        </div>
        <div class="utterance-list" data-warmup-list="true">${renderWarmupRowsHtml(warmup)}</div>
      </div>
    `);
  }

  for (let i = 0; i < qas.length; i++) {
    const qa = qas[i];
    const isSelected = qa.id === state.selectedQaId;
    const probeCount = (qa.dive_deeper_history || []).length;
    const pState = qa.probe_state || "idle";

    parts.push(`
      <div class="qa-card ${isSelected ? "selected" : ""}" data-qa-card="${qa.id}">
        <div class="qa-card-header">
          <span class="qa-badge">${escapeHtml(qa.badge)}</span>
          <input
            class="qa-title-input"
            data-qa-title="${qa.id}"
            value="${escapeHtml(qa.title)}"
            title="Click to rename card title"
          />
          <div class="qa-card-actions">
            <span class="qa-probe-pill-host">${renderCardProbePill(pState)}</span>
            ${
              i > 0 && canMergeAny
                ? `<button class="btn btn-sm" data-action="merge" data-qa="${qa.id}" title="Merge into previous card">Merge ↑</button>`
                : ""
            }
            <button
              class="btn btn-sm btn-dive"
              data-action="dive"
              data-qa="${qa.id}"
              ${pState === "empty" || pState === "loading" ? "disabled" : ""}
            >
              ${pState === "loading" ? "Probing..." : `Dive Deeper${probeCount ? ` (${probeCount})` : ""}`}
            </button>
          </div>
        </div>
        <div class="utterance-list">
          ${renderCardUtterancesHtml(qa, canSplitAny)}
        </div>
        <input
          class="qa-note-input"
          data-qa-note="${qa.id}"
          placeholder="Quick interviewer note for ${escapeHtml(qa.badge)}..."
          value="${escapeHtml(qa.notes || "")}"
        />
      </div>
    `);
  }

  container.innerHTML = parts.join("");
  if (wasNearBottom) {
    container.scrollTop = container.scrollHeight;
  } else {
    container.scrollTop = prevScrollTop;
  }
}

function renderProbesPanel() {
  const qas = state.session?.qa_groups_resolved || [];
  const qa = qas.find((q) => q.id === state.selectedQaId) || qas[qas.length - 1] || null;

  const diveBtn = el("diveDeeperBtn");
  const refreshBtn = el("refreshProbesBtn");
  const label = el("selectedQaLabel");
  const content = el("probesContent");

  if (!qa) {
    label.textContent = "Select or start a card";
    diveBtn.disabled = true;
    diveBtn.innerHTML = `Dive Deeper <kbd>D</kbd>`;
    refreshBtn.classList.add("hidden");
    content.innerHTML = `
      <div style="color:var(--text-muted); font-size:13px; padding:6px 2px;">
        Start live capture or inject a sample exchange below, then press <kbd>D</kbd> on any card for 3 high-signal follow-up questions.
      </div>
    `;
    return;
  }

  state.selectedQaId = qa.id;
  const pState = qa.probe_state || "idle";
  label.textContent = `${qa.badge}: ${qa.title}`;
  const utts = qa.utterances || [];

  diveBtn.disabled = pState === "empty" || pState === "loading";
  diveBtn.innerHTML =
    pState === "ready_cached"
      ? `⚡ Dive Deeper (Instant) <kbd>D</kbd>`
      : pState === "loading"
      ? `Generating...`
      : `Dive Deeper <kbd>D</kbd>`;

  const history = qa.dive_deeper_history || [];
  refreshBtn.classList.toggle(
    "hidden",
    pState === "loading" || (history.length === 0 && pState !== "ready_cached" && pState !== "stale_cached")
  );

  if (pState === "loading") {
    content.innerHTML = `<div class="quick-read-box">Generating non-repeating follow-up probes for <strong>${escapeHtml(qa.badge)}</strong>...</div>`;
    return;
  }

  if (history.length === 0) {
    let statusHint = `Press <kbd>D</kbd> anytime (even while the candidate is mid-sentence) to generate 3 follow-up questions.`;
    if (pState === "ready_cached") {
      statusHint = `<span style="color:var(--candidate); font-weight:600;">Probes are pre-cached for all ${utts.length} turns—press <kbd>D</kbd> for instant 0ms reveal!</span>`;
    } else if (pState === "warming") {
      statusHint = `<span style="color:#a5b4fc;">Pre-warming background follow-up probes...</span>`;
    } else if (pState === "stale_cached") {
      statusHint = `<span style="color:var(--amber);">Candidate added newer turns since pre-warm—press <kbd>D</kbd> to probe latest turns.</span>`;
    }
    content.innerHTML = `
      <div style="color:var(--text-muted); font-size:13px; padding:6px 2px;">
        <strong>${escapeHtml(qa.badge)}</strong> selected (${utts.length} utterance${utts.length === 1 ? "" : "s"}).
        ${statusHint}
      </div>
    `;
    return;
  }

  const latest = history[history.length - 1];
  const staleBanner =
    pState === "revealed_stale"
      ? `<div style="font-size:12px; color:var(--amber);">↻ Candidate spoke additional turns since these probes—press <kbd>D</kbd> to probe the newer material.</div>`
      : "";

  const cardsHtml = (latest.probes || [])
    .map(
      (p) => `
      <div class="probe-card">
        <div class="probe-angle">${escapeHtml(p.angle)}</div>
        <div class="probe-question">“${escapeHtml(p.question)}”</div>
        <div class="probe-why">${escapeHtml(p.why)}</div>
      </div>
    `
    )
    .join("");

  content.innerHTML = `
    ${staleBanner}
    <div class="quick-read-box">
      <strong>Signal Read:</strong> ${escapeHtml(latest.quick_read)}
    </div>
    ${cardsHtml}
    <div style="font-size:11px; color:var(--text-dim); text-align:right;">
      Model: ${escapeHtml(latest.model_used || "default")} · Press <kbd>D</kbd> or ↻ for fresh non-repeating angles
    </div>
  `;
}

function formatInlineMarkdown(raw) {
  let s = escapeHtml(raw);
  const codeTokens = [];
  s = s.replace(/`([^`]+)`/g, (_, code) => {
    const idx = codeTokens.length;
    codeTokens.push(`<code>${code}</code>`);
    return `\u0000CODE${idx}\u0000`;
  });

  // Make [MM:SS, utt_id] citations clickable links into the live transcript
  s = s.replace(
    /\[(\d{1,2}:\d{2}),\s*(utt_[A-Za-z0-9_-]+)\]/g,
    (_, ts, uttId) =>
      `<button type="button" class="cite-link" data-cite-utt="${uttId}" title="Jump to ${uttId}">[${ts}, ${uttId}]</button>`
  );

  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  s = s.replace(/\u0000CODE(\d+)\u0000/g, (_, idx) => codeTokens[Number(idx)] || "");
  return s;
}

function renderMarkdownSafe(md) {
  const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let inUl = false;
  let inOl = false;
  let para = [];

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

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushPara();
      closeLists();
      continue;
    }
    const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      flushPara();
      closeLists();
      const level = Math.min(4, Math.max(2, hMatch[1].length));
      out.push(`<h${level}>${formatInlineMarkdown(hMatch[2])}</h${level}>`);
      continue;
    }
    if (/^[-*_]{3,}$/.test(trimmed)) {
      flushPara();
      closeLists();
      out.push(`<hr style="border:none; border-top:1px solid var(--border); margin:10px 0;" />`);
      continue;
    }
    const bqMatch = trimmed.match(/^>\s?(.*)$/);
    if (bqMatch) {
      flushPara();
      closeLists();
      out.push(`<blockquote>${formatInlineMarkdown(bqMatch[1])}</blockquote>`);
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

function jumpToUtteranceCitation(uttId) {
  if (!uttId) return;
  const container = el("qaList");
  const row = container.querySelector(`[data-utt-row="${CSS.escape(uttId)}"]`);
  if (!row) {
    showToast(`Utterance ${uttId} not found in current view.`);
    return;
  }
  const card = row.closest("[data-qa-card]");
  if (card) {
    const qaId = card.getAttribute("data-qa-card");
    if (qaId && state.selectedQaId !== qaId) {
      state.selectedQaId = qaId;
      for (const c of container.querySelectorAll(".qa-card.selected")) {
        c.classList.remove("selected");
      }
      card.classList.add("selected");
      renderProbesPanel();
    }
  }
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.remove("cite-highlight");
  void row.offsetWidth;
  row.classList.add("cite-highlight");
  setTimeout(() => row.classList.remove("cite-highlight"), 2200);
}

function renderCalibrationPanel() {
  const m = state.session?.metrics;
  const sm = state.session?.state_machines;
  const calibRunning = sm?.jobs?.calibration === "running";
  const allowed = allowedEventsSet();

  el("runCalibrationBtn").disabled = !allowed.has("generate_calibration") || calibRunning;
  el("runCalibrationBtn").textContent = calibRunning
    ? "Calibrating..."
    : "Submit My Call & Reveal AI Audit";

  if (m) {
    el("statQaCount").textContent = String(m.qa_count);
    el("statTalkRatio").textContent =
      m.interviewer_speak_seconds + m.candidate_speak_seconds > 0
        ? `${Math.round(m.interviewer_time_pct)}% / ${Math.round(m.candidate_time_pct)}%`
        : "—";
    el("statAvgDepth").textContent = `${Math.round(m.avg_candidate_words_per_qa)}w`;

    const perCardBox = el("perCardStatsBox");
    if (m.per_card && m.per_card.length > 0) {
      perCardBox.classList.remove("hidden");
      perCardBox.innerHTML =
        `<div style="font-weight:600; color:var(--text);">Code-Computed Time &amp; Words Per Card</div>` +
        m.per_card
          .map(
            (c) =>
              `<div><strong>${escapeHtml(c.badge)}</strong> (${escapeHtml(c.title)}): <strong>${fmtClock(c.span_seconds)}</strong> span · Cand ${c.candidate_speak_seconds}s (${c.candidate_words}w) · You ${c.interviewer_speak_seconds}s (${c.interviewer_words}w)</div>`
          )
          .join("");
    } else {
      perCardBox.classList.add("hidden");
    }
  }

  const outputEl = el("calibrationOutput");
  if (calibRunning) {
    outputEl.textContent =
      "Regrouping full transcript context & auditing your call against [MM:SS, utt_id] evidence...";
  } else if (state.session?.calibration_markdown) {
    outputEl.innerHTML = renderMarkdownSafe(state.session.calibration_markdown);
  } else {
    outputEl.textContent =
      "Lock in your gut rating above and click 'Submit My Call & Reveal AI Audit'.";
  }
}

function activateRightTab(which) {
  state.rightTab = which;
  const isProbes = which === "probes";
  el("tabProbes").classList.toggle("active", isProbes);
  el("tabCalibration").classList.toggle("active", !isProbes);
  el("probesView").classList.toggle("hidden", !isProbes);
  el("calibrationView").classList.toggle("hidden", isProbes);
}

/* ---------------- Settings & History Modals ---------------- */

async function openSettingsModal() {
  const res = await fetch("/api/status");
  const data = await res.json();
  state.llm = data.llm;

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
  presetSelect.innerHTML = Object.entries(presets)
    .map(([key, val]) => `<option value="${key}">${escapeHtml(val.label)}</option>`)
    .join("");

  if (presets[data.llm.provider]) {
    presetSelect.value = data.llm.provider;
  }
  el("llmBaseUrlInput").value = data.llm.base_url || "";
  el("llmFastModelInput").value = data.llm.fast_model || "";
  el("llmCalibModelInput").value = data.llm.calibration_model || "";
  el("llmApiKeyInput").value = "";
  const updateApiKeyHint = (providerKey) => {
    if (providerKey === "agy" || providerKey === "pi") {
      el("maskedKeyHint").textContent = `(Zero API key — uses local ${providerKey} CLI)`;
      el("llmApiKeyInput").placeholder = `Not required for local ${providerKey} CLI`;
    } else if (providerKey === "ollama") {
      el("maskedKeyHint").textContent = "(Optional for local Ollama)";
      el("llmApiKeyInput").placeholder = "Not required for local Ollama";
    } else {
      el("maskedKeyHint").textContent = data.llm.has_api_key
        ? `(Saved: ${data.llm.masked_api_key})`
        : "(None configured yet)";
      el("llmApiKeyInput").placeholder = "Paste API key (saved to .env.local)";
    }
  };
  updateApiKeyHint(presetSelect.value);

  presetSelect.onchange = () => {
    const p = presets[presetSelect.value];
    if (p) {
      el("llmBaseUrlInput").value = p.base_url;
      el("llmFastModelInput").value = p.fast_model;
      el("llmCalibModelInput").value = p.calibration_model;
    }
    updateApiKeyHint(presetSelect.value);
  };

  setModalState("settings");
}

async function saveSettings() {
  const payload = {
    provider: el("llmPresetSelect").value,
    base_url: el("llmBaseUrlInput").value,
    fast_model: el("llmFastModelInput").value,
    calibration_model: el("llmCalibModelInput").value,
    api_key: el("llmApiKeyInput").value || null,
    parakeet_model: el("parakeetModelSelect").value,
  };
  let res;
  try {
    res = await fetch("/api/settings/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    showToast(`Failed to save settings: ${err.message || "network error"}`);
    return;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = typeof err.detail === "string" ? err.detail : err.detail?.message;
    showToast(msg || `Could not save settings (${res.status}).`);
    return;
  }
  const data = await res.json();
  if (data.asr) {
    state.asr = data.asr;
    renderAsrBadge();
  }
  showToast("Settings saved to .env.local", 2200);
  setModalState("none");
}

async function openHistoryModal() {
  const res = await fetch("/api/sessions");
  const data = await res.json();
  const list = el("historyList");
  const sessions = data.sessions || [];
  if (!sessions.length) {
    list.innerHTML = `<div style="color:var(--text-muted); font-size:13px;">No saved sessions yet.</div>`;
  } else {
    list.innerHTML = sessions
      .map(
        (s) => `
        <div class="history-item">
          <div>
            <div style="font-weight:600;">${escapeHtml(s.role_title)} (${escapeHtml(s.interview_type)})</div>
            <div style="font-size:11.5px; color:var(--text-muted);">
              ${escapeHtml(s.created_at_iso)} · ${s.qa_count} cards · ${fmtClock(s.duration_seconds)}
            </div>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-sm" data-load-session="${escapeHtml(s.id)}">Open</button>
            <a class="btn btn-sm" href="/api/sessions/${encodeURIComponent(s.id)}/markdown?audience=interviewer" style="text-decoration:none;">Interviewer .md</a>
            <a class="btn btn-sm" href="/api/sessions/${encodeURIComponent(s.id)}/markdown?audience=candidate" style="text-decoration:none;">Candidate .md</a>
          </div>
        </div>
      `
      )
      .join("");
  }
  setModalState("history");
}

/* ---------------- Event Listeners & Hotkeys ---------------- */

function currentConfigPayload() {
  return {
    role_title: el("roleInput").value,
    interview_type: el("typeSelect").value,
    mode: el("modeSelect").value,
    rubric_notes: el("rubricInput").value,
    consent_confirmed: el("consentCheckbox").checked,
    save_raw_audio: el("saveAudioCheckbox").checked,
  };
}

function pushSessionConfig() {
  sendCmd({ type: "configure_session", config: currentConfigPayload() });
}

function initEvents() {
  el("startCaptureBtn").onclick = startCapture;
  el("stopCaptureBtn").onclick = () => stopCaptureSession({ flush: true });

  el("consentCheckbox").onchange = pushSessionConfig;
  el("saveAudioCheckbox").onchange = pushSessionConfig;
  el("copyConsentScriptBtn").onclick = async () => {
    const btn = el("copyConsentScriptBtn");
    try {
      await navigator.clipboard.writeText(CONSENT_SCRIPT);
      const orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = orig;
      }, 1400);
    } catch (_) {
      showToast("Clipboard blocked — read the consent note directly to the candidate.");
    }
  };

  el("modeSelect").onchange = () => {
    renderSpeakerToggle();
    pushSessionConfig();
  };
  el("roleInput").onchange = pushSessionConfig;
  el("typeSelect").onchange = pushSessionConfig;
  el("rubricInput").onchange = pushSessionConfig;

  el("speakerToggleBtn").onclick = () => {
    const next = state.manualSpeaker === "interviewer" ? "candidate" : "interviewer";
    sendCmd({ type: "set_manual_speaker", speaker: next });
  };

  el("newSessionBtn").onclick = () => {
    state.pendingNewSession = true;
    if (state.captureState === "streaming") {
      stopCaptureSession({ flush: false });
    }
    el("consentCheckbox").checked = false;
    state.selectedQaId = null;
    state.baseElapsedMs = 0;
    state.localSegmentStartedAt = null;
    renderTimerPill(0);
    activateRightTab("probes");
    sendCmd({
      type: "new_session",
      config: { ...currentConfigPayload(), consent_confirmed: false },
    });
  };

  el("newCardBtn").onclick = () => sendCmd({ type: "force_new_card" });
  el("regroupBtn").onclick = () => sendCmd({ type: "regroup_cards" });

  el("diveDeeperBtn").onclick = () => {
    if (state.selectedQaId) {
      sendCmd({ type: "dive_deeper", qa_id: state.selectedQaId, force_refresh: false });
    }
  };
  el("refreshProbesBtn").onclick = () => {
    if (state.selectedQaId) {
      sendCmd({ type: "dive_deeper", qa_id: state.selectedQaId, force_refresh: true });
    }
  };

  el("runCalibrationBtn").onclick = () => {
    const rating = {
      hire_signal: el("selfHireSignal").value,
      perceived_level: el("selfLevelInput").value.trim(),
      gut_rationale: el("selfRationaleInput").value.trim(),
      biggest_concern: el("selfConcernInput").value.trim(),
    };
    sendCmd({
      type: "generate_calibration",
      rating,
      regroup_first: true,
    });
  };

  el("calibrationOutput").onclick = (ev) => {
    const btn = ev.target.closest("[data-cite-utt]");
    if (!btn) return;
    jumpToUtteranceCitation(btn.getAttribute("data-cite-utt"));
  };

  el("exportInterviewerMdBtn").onclick = () => {
    if (state.session?.id) {
      window.location.href = `/api/sessions/${encodeURIComponent(state.session.id)}/markdown?audience=interviewer`;
    }
  };
  el("exportCandidateMdBtn").onclick = () => {
    if (state.session?.id) {
      window.location.href = `/api/sessions/${encodeURIComponent(state.session.id)}/markdown?audience=candidate`;
    }
  };

  el("tabProbes").onclick = () => activateRightTab("probes");
  el("tabCalibration").onclick = () => activateRightTab("calibration");

  el("settingsBtn").onclick = openSettingsModal;
  el("closeSettingsBtn").onclick = () => setModalState("none");
  el("saveSettingsBtn").onclick = saveSettings;

  el("historyBtn").onclick = openHistoryModal;
  el("closeHistoryBtn").onclick = () => setModalState("none");
  el("historyList").onclick = async (ev) => {
    const btn = ev.target.closest("[data-load-session]");
    if (!btn) return;
    if (!allowedEventsSet().has("load_historical_session") || state.captureState !== "idle") {
      showToast("Stop live capture before opening a saved session.");
      return;
    }
    const id = btn.getAttribute("data-load-session");
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/load`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const msg = typeof err.detail === "string" ? err.detail : err.detail?.message;
      showToast(msg || "Could not load session.");
      return;
    }
    const data = await res.json();
    state.session = data.session;
    state.baseElapsedMs = data.session.live_elapsed_ms || 0;
    renderTimerPill(state.baseElapsedMs / 1000);
    const qas = state.session.qa_groups_resolved || [];
    state.selectedQaId = qas.length ? qas[0].id : null;
    syncConfigInputsFromSession();
    renderSpeakerToggle();
    renderPhaseAndControls();
    renderQAList();
    renderProbesPanel();
    renderCalibrationPanel();
    renderMetricsBar();
    setModalState("none");
  };

  const qaList = el("qaList");
  qaList.onclick = (ev) => {
    const actionBtn = ev.target.closest("[data-action]");
    if (actionBtn) {
      ev.stopPropagation();
      const action = actionBtn.getAttribute("data-action");
      const qaId = actionBtn.getAttribute("data-qa");
      if (action === "dive") {
        state.selectedQaId = qaId;
        renderQAList();
        sendCmd({ type: "dive_deeper", qa_id: qaId, force_refresh: false });
      } else if (action === "merge") {
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

  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.modalState !== "none") {
      ev.preventDefault();
      setModalState("none");
      return;
    }

    const tag = (ev.target?.tagName || "").toUpperCase();
    const isTyping = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    if (isTyping || state.modalState !== "none") return;

    if (ev.key === "d" || ev.key === "D") {
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
    } else if (ev.key === "n" || ev.key === "N") {
      ev.preventDefault();
      if (allowedEventsSet().has("force_new_card")) {
        sendCmd({ type: "force_new_card" });
      }
    } else if (ev.key === "Tab" && el("modeSelect").value === "in_person") {
      ev.preventDefault();
      const next = state.manualSpeaker === "interviewer" ? "candidate" : "interviewer";
      sendCmd({ type: "set_manual_speaker", speaker: next });
    }
  });
}

initEvents();
renderSpeakerToggle();
renderCaptureControls();
connectWS();
