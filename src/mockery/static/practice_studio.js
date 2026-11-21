/* AI Mock Practice Studio — Solo AI Interviewer, Local Voice (TTS + Parakeet Mic),
 * Live CARL Coaching Meter, Supportive CARL Debrief, 45-Second Retry Drill,
 * and extensible Studio Slot for Milestone 3 (System Design) & Milestone 4 (Coding).
 */

const SOURCE_NOTES = {
  mixed:
    "Blends curated role-ranked prompts with questions tailored to your locally redacted resume & JD.",
  bank_only:
    "Uses only Mockery's curated question banks — ranked by relevance to your target job listing.",
  resume_jd_tailored:
    "Generates tailored prompts from your locally redacted resume and target job listing.",
};

const FORMAT_LABELS = {
  behavioral: "Behavioral",
  system_design: "System Design",
  coding: "Coding",
  general: "General",
};

const CARL_STATE_LABELS = {
  strong: "Strong",
  partial: "Partial",
  missing: "Pending",
};

function downsampleToInt16(floatChunks, inputRate, targetRate = 16000) {
  let totalLen = 0;
  for (const ch of floatChunks) totalLen += ch.length;
  if (totalLen === 0) return new Int16Array(0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const ch of floatChunks) {
    merged.set(ch, offset);
    offset += ch.length;
  }
  if (inputRate === targetRate) {
    const out = new Int16Array(merged.length);
    for (let i = 0; i < merged.length; i++) {
      const s = Math.max(-1, Math.min(1, merged[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }
  const ratio = inputRate / targetRate;
  // Lightweight box/triangular anti-alias pre-filter when downsampling
  let src = merged;
  if (ratio > 1.05 && merged.length >= 8) {
    const radius = Math.max(1, Math.min(6, Math.round(ratio)));
    const filtered = new Float32Array(merged.length);
    for (let i = 0; i < merged.length; i++) {
      const lo = Math.max(0, i - radius);
      const hi = Math.min(merged.length - 1, i + radius);
      let acc = 0;
      let wsum = 0;
      for (let j = lo; j <= hi; j++) {
        const w = radius + 1 - Math.abs(j - i);
        acc += merged[j] * w;
        wsum += w;
      }
      filtered[i] = wsum > 0 ? acc / wsum : merged[i];
    }
    src = filtered;
  }
  const outLen = Math.max(1, Math.round(src.length / ratio));
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const s0 = src[idx] || 0;
    const s1 = src[Math.min(idx + 1, src.length - 1)] || s0;
    const sample = Math.max(-1, Math.min(1, s0 + (s1 - s0) * frac));
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return out;
}

function fmtShortClock(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, "0")}`;
}

export function createPracticeStudioController({
  state,
  el,
  escapeHtml,
  showToast,
  onPracticeActiveChange,
}) {
  const local = {
    homeMode: "copilot", // "copilot" | "practice"
    questionSource: "mixed",
    targetLevel: "L5",
    session: null,
    sessionGen: 0,
    ttsStatus: null,
    busy: false,

    // Voice playback state (with generation token to prevent race conditions)
    voiceEnabled: true,
    isSpeaking: false,
    speakGen: 0,
    activeAudio: null,
    activeAudioUrl: null,

    // Local mic recording state
    recordingTarget: null, // null | "answer" | "retry"
    recGen: 0,
    recSamples: 0,
    recAutoStopping: false,
    recStream: null,
    recAudioCtx: null,
    recProcessor: null,
    recSource: null,
    recChunks: [],
    recStartedAt: 0,
    recTimerHandle: null,

    // 45-second retry drill timer (wall-clock anchored)
    retrySecondsLeft: 45,
    retryDeadlineMs: 0,
    retryTimerHandle: null,

    // Studio plugins (M3 System Design / M4 Coding)
    studioPlugins: new Map(),
    mountedPlugin: null,
    mountedPluginKey: null,
    workspaceSyncTimer: null,
  };

  function getSelectedFormat() {
    const activeBtn = el("setupTypePills")?.querySelector(".seg-opt.active");
    return activeBtn?.getAttribute("data-setup-type") || el("typeSelect")?.value || "behavioral";
  }

  function currentQuestionId() {
    const sess = local.session;
    if (!sess || !Array.isArray(sess.questions) || sess.questions.length === 0) return null;
    const idx = Math.min(sess.current_question_index || 0, sess.questions.length - 1);
    return sess.questions[idx]?.id || null;
  }

  function formatWithNewlines(text) {
    return escapeHtml(String(text || "")).replace(/\n/g, "<br>");
  }

  /* ---------------- Home Mode Switching & Add-On Gating ---------------- */

  function setHomeMode(mode) {
    const practiceAllowed = state.addons?.modules_map?.practice_mode !== false;
    const next = mode === "practice" && practiceAllowed ? "practice" : "copilot";
    local.homeMode = next;

    el("homeModeSwitch")
      ?.querySelectorAll("[data-home-mode]")
      .forEach((btn) => {
        const active = btn.getAttribute("data-home-mode") === next;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
      });

    const isPractice = next === "practice";
    el("copilotLaunchRows")?.classList.toggle("hidden", isPractice);
    el("copilotLaunchCta")?.classList.toggle("hidden", isPractice);
    el("practiceLaunchRows")?.classList.toggle("hidden", !isPractice);
    el("practiceLaunchCta")?.classList.toggle("hidden", !isPractice);
    el("setupSimulateBtn")?.classList.toggle("hidden", isPractice);

    const kicker = el("homeKickerLabel");
    if (kicker) kicker.textContent = isPractice ? "Practice target role" : "New interview";

    const cap = el("homeModeCaption");
    if (cap) {
      cap.textContent = isPractice
        ? "Solo AI interviewer with supportive CARL coaching & local voice"
        : "Real-time transcript & probes while you interview someone";
    }

    if (isPractice) {
      refreshQuestionPicker();
    }
  }

  function applyStatusSnapshot(data) {
    if (data?.tts) {
      local.ttsStatus = data.tts;
      const voiceLbl = el("practiceVoiceLabel");
      if (voiceLbl) {
        const eng =
          data.tts.engine === "kokoro"
            ? "Kokoro-82M neural voice"
            : data.tts.engine === "macos_say"
            ? "macOS local voice"
            : "browser voice";
        voiceLbl.textContent = `AI speaks questions aloud (${eng}, instant barge-in)`;
      }
    }
    if (data?.addons?.modules_map) {
      const practiceOn = data.addons.modules_map.practice_mode !== false;
      el("homeModeSwitchRow")?.classList.toggle("hidden", !practiceOn);
      if (!practiceOn && local.homeMode === "practice") {
        setHomeMode("copilot");
      }
      if (!practiceOn && local.session) {
        exitPracticeToHome();
      } else if (local.session) {
        syncStudioSlot();
      }
    }
  }

  async function refreshQuestionPicker() {
    const picker = el("practiceQuestionPicker");
    if (!picker) return;
    const fmt = getSelectedFormat();
    const catParam = fmt === "general" ? "" : `&category=${encodeURIComponent(fmt)}`;
    const lvlParam = local.targetLevel ? `&level=${encodeURIComponent(local.targetLevel)}` : "";
    try {
      const res = await fetch(`/api/banks?match_active_jd=true${catParam}${lvlParam}`);
      if (!res.ok) return;
      const data = await res.json();
      let questions = data.questions || [];
      if (questions.length === 0 && catParam) {
        const fallbackRes = await fetch(`/api/banks?match_active_jd=true${catParam}`);
        if (fallbackRes.ok) {
          questions = (await fallbackRes.json()).questions || [];
        }
      }
      const prevVal = picker.value;
      picker.innerHTML =
        `<option value="">Auto-select best match for your target JD</option>` +
        questions
          .slice(0, 24)
          .map((q) => {
            const tag = q.jd_relevance_score > 0 ? " ★ JD match" : "";
            const title = q.title || q.competency || q.prompt?.slice(0, 52) || q.id;
            return `<option value="${escapeHtml(q.id)}">${escapeHtml(title)}${tag}</option>`;
          })
          .join("");
      if ([...picker.options].some((o) => o.value === prevVal)) {
        picker.value = prevVal;
      }
    } catch {
      /* ignore */
    }
  }

  async function refreshPracticeHistory() {
    const section = el("homePracticeHistorySection");
    const listEl = el("homePracticeList");
    if (!section || !listEl) return;
    if (state.addons?.modules_map?.practice_mode === false) {
      section.classList.add("hidden");
      return;
    }
    try {
      const res = await fetch("/api/practice/sessions");
      if (!res.ok) return;
      const data = await res.json();
      const items = data.sessions || [];
      if (items.length === 0) {
        section.classList.add("hidden");
        listEl.innerHTML = "";
        return;
      }
      section.classList.remove("hidden");
      listEl.innerHTML = items
        .slice(0, 8)
        .map((s) => {
          const role = s.company ? `${s.role_title} · ${s.company}` : s.role_title;
          const fmt = FORMAT_LABELS[s.interview_type] || s.interview_type;
          const status = s.has_debrief ? "Debrief ready" : "In progress";
          return `
            <button type="button" class="home-item practice-history-item" data-open-practice="${escapeHtml(s.id)}">
              <div class="home-item-title">${escapeHtml(role)}</div>
              <div class="home-item-meta">${escapeHtml(fmt)} · ${escapeHtml(s.target_level || "L5")} · ${escapeHtml(status)}</div>
            </button>
          `;
        })
        .join("");
    } catch {
      /* ignore */
    }
  }

  /* ---------------- Local Voice Playback & Instant Barge-In ---------------- */

  function setSpeakingUI(speaking) {
    local.isSpeaking = speaking;
    el("practiceBargeInBtn")?.classList.toggle("hidden", !speaking);
  }

  function stopSpeaking() {
    local.speakGen += 1;
    if (local.activeAudio) {
      try {
        local.activeAudio.pause();
        local.activeAudio.src = "";
      } catch {
        /* ignore */
      }
      local.activeAudio = null;
    }
    if (local.activeAudioUrl) {
      try {
        URL.revokeObjectURL(local.activeAudioUrl);
      } catch {
        /* ignore */
      }
      local.activeAudioUrl = null;
    }
    if (window.speechSynthesis) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
    }
    setSpeakingUI(false);
  }

  async function speakText(text, { force = false } = {}) {
    const clean = String(text || "").trim();
    if (!clean) return;
    if (!local.voiceEnabled && !force) return;

    stopSpeaking();
    const myGen = local.speakGen;
    setSpeakingUI(true);

    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean, rate: 1.0 }),
      });
      if (myGen !== local.speakGen) return;
      if (res.ok) {
        const blob = await res.blob();
        if (myGen !== local.speakGen) return;
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        local.activeAudio = audio;
        local.activeAudioUrl = url;
        audio.onended = () => {
          if (local.activeAudio === audio) stopSpeaking();
        };
        audio.onerror = () => {
          if (local.activeAudio === audio) stopSpeaking();
        };
        await audio.play();
        return;
      }
    } catch {
      /* fall through to browser speech synthesis */
    }

    if (myGen !== local.speakGen) return;
    if (window.speechSynthesis) {
      try {
        const voices = window.speechSynthesis.getVoices?.() || [];
        const localVoice =
          voices.find((v) => v.localService === true && /^en\b/i.test(v.lang || "")) ||
          voices.find((v) => v.localService === true) ||
          null;
        if (!localVoice) {
          // Strict privacy: never silently route practice prompts through a remote cloud TTS voice
          setSpeakingUI(false);
          return;
        }
        const utt = new SpeechSynthesisUtterance(clean.slice(0, 1400));
        utt.voice = localVoice;
        utt.rate = 1.02;
        utt.onend = () => {
          if (myGen === local.speakGen) setSpeakingUI(false);
        };
        utt.onerror = () => {
          if (myGen === local.speakGen) setSpeakingUI(false);
        };
        window.speechSynthesis.speak(utt);
        return;
      } catch {
        /* ignore */
      }
    }
    setSpeakingUI(false);
  }

  /* ---------------- Local Mic Capture -> Parakeet ASR ---------------- */

  function updateMicUI(target, phase, note = "") {
    const isRetry = target === "retry";
    const btn = el(isRetry ? "practiceRetryMicBtn" : "practiceMicBtn");
    const lbl = el(isRetry ? "practiceRetryMicLabel" : "practiceMicLabel");
    const status = el(isRetry ? "practiceRetryMicStatus" : "practiceMicStatus");
    if (btn) btn.classList.toggle("is-recording", phase === "recording");
    if (lbl) {
      if (phase === "recording") lbl.textContent = "Stop & transcribe";
      else if (phase === "transcribing") lbl.textContent = "Transcribing…";
      else lbl.textContent = isRetry ? "Speak retry" : "Speak answer";
    }
    if (btn) btn.disabled = phase === "transcribing";
    if (status && note) status.textContent = note;
  }

  function cleanupMicResources() {
    if (local.recTimerHandle) {
      clearInterval(local.recTimerHandle);
      local.recTimerHandle = null;
    }
    if (local.recProcessor) {
      try {
        local.recProcessor.onaudioprocess = null;
        local.recProcessor.disconnect();
      } catch {
        /* ignore */
      }
      local.recProcessor = null;
    }
    if (local.recSource) {
      try {
        local.recSource.disconnect();
      } catch {
        /* ignore */
      }
      local.recSource = null;
    }
    if (local.recStream) {
      for (const t of local.recStream.getTracks()) {
        try {
          t.stop();
        } catch {
          /* ignore */
        }
      }
      local.recStream = null;
    }
    if (local.recAudioCtx) {
      try {
        local.recAudioCtx.close();
      } catch {
        /* ignore */
      }
      local.recAudioCtx = null;
    }
    local.recChunks = [];
    local.recSamples = 0;
    local.recAutoStopping = false;
  }

  function abortMicRecording() {
    local.recGen += 1;
    const prevTarget = local.recordingTarget;
    local.recordingTarget = null;
    cleanupMicResources();
    if (prevTarget) {
      updateMicUI(prevTarget, "idle", "Local mic ready");
    }
  }

  async function toggleMicRecording(target = "answer") {
    // Instant barge-in: pressing speak immediately silences the AI voice
    stopSpeaking();

    if (local.recordingTarget === target) {
      await stopMicAndTranscribe();
      return;
    }
    if (local.recordingTarget && local.recordingTarget !== target) {
      await stopMicAndTranscribe();
    }

    if (!local.session) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      showToast("Microphone capture is unavailable in this browser.", 3000);
      return;
    }

    local.recGen += 1;
    const myRecGen = local.recGen;
    const mySessionGen = local.sessionGen;
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      // Guard against exiting/switching session or double-tapping while awaiting permission
      if (myRecGen !== local.recGen || mySessionGen !== local.sessionGen || !local.session) {
        for (const t of stream.getTracks()) {
          try {
            t.stop();
          } catch {
            /* ignore */
          }
        }
        return;
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const maxSamples = Math.round((ctx.sampleRate || 48000) * 120);

      local.recChunks = [];
      local.recSamples = 0;
      local.recAutoStopping = false;
      local.recStream = stream;
      local.recAudioCtx = ctx;
      local.recSource = source;
      local.recProcessor = processor;
      local.recordingTarget = target;
      local.recStartedAt = Date.now();

      processor.onaudioprocess = (ev) => {
        if (myRecGen !== local.recGen || local.recAutoStopping) return;
        const ch = ev.inputBuffer.getChannelData(0);
        const remaining = maxSamples - local.recSamples;
        if (remaining <= 0) {
          local.recAutoStopping = true;
          stopMicAndTranscribe();
          return;
        }
        if (ch.length > remaining) {
          local.recChunks.push(new Float32Array(ch.subarray(0, remaining)));
          local.recSamples += remaining;
          local.recAutoStopping = true;
          showToast("Reached 2-minute clip limit — transcribing locally…", 2800);
          stopMicAndTranscribe();
          return;
        }
        local.recChunks.push(new Float32Array(ch));
        local.recSamples += ch.length;
      };
      source.connect(processor);
      processor.connect(ctx.destination);

      updateMicUI(target, "recording", "Listening locally · 0:00");
      local.recTimerHandle = setInterval(() => {
        const elapsed = (Date.now() - local.recStartedAt) / 1000;
        updateMicUI(target, "recording", `Listening locally · ${fmtShortClock(elapsed)}`);
      }, 400);
    } catch (err) {
      if (stream) {
        for (const t of stream.getTracks()) {
          try {
            t.stop();
          } catch {
            /* ignore */
          }
        }
      }
      if (myRecGen === local.recGen) {
        cleanupMicResources();
        local.recordingTarget = null;
      }
      showToast(`Microphone access denied: ${err.message || err}`, 3200);
    }
  }

  async function stopMicAndTranscribe() {
    const target = local.recordingTarget;
    if (!target) return;
    local.recGen += 1;
    const mySessionGen = local.sessionGen;
    const sampleRate = local.recAudioCtx?.sampleRate || 48000;
    const chunks = local.recChunks.slice();
    local.recordingTarget = null;
    cleanupMicResources();

    const int16 = downsampleToInt16(chunks, sampleRate, 16000);
    if (int16.length < 1600) {
      updateMicUI(target, "idle", "Recording was too short — try again or type below.");
      return;
    }

    updateMicUI(target, "transcribing", "Transcribing locally with Parakeet…");
    try {
      const res = await fetch("/api/asr/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: int16.buffer,
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) {
        throw new Error(data.detail || "Transcription failed");
      }
      const transcript = String(data.text || "").trim();
      const inputEl = el(target === "retry" ? "practiceRetryInput" : "practiceAnswerInput");
      if (inputEl && transcript) {
        const prefix = inputEl.value.trim();
        inputEl.value = prefix ? `${prefix} ${transcript}` : transcript;
        inputEl.focus();
      }
      const ms = data.inference_ms ? ` (${Math.round(data.inference_ms)} ms local)` : "";
      updateMicUI(
        target,
        "idle",
        transcript
          ? `Transcribed locally${ms} — edit or press Send`
          : "No speech detected — try speaking closer to the mic"
      );
    } catch (err) {
      if (mySessionGen !== local.sessionGen) return;
      updateMicUI(target, "idle", `Local transcription error: ${err.message || err}`);
      showToast(String(err.message || err), 3200);
    }
  }

  /* ---------------- Extensible Studio Slot (M3 & M4) ---------------- */

  function registerStudioPlugin(interviewType, factory) {
    local.studioPlugins.set(interviewType, factory);
    if (local.session && local.session.config?.interview_type === interviewType) {
      syncStudioSlot();
    }
  }

  function unmountStudioPlugin({ flush = true } = {}) {
    const hadPending = Boolean(local.workspaceSyncTimer || local.workspaceDirty);
    if (local.workspaceSyncTimer) {
      clearTimeout(local.workspaceSyncTimer);
      local.workspaceSyncTimer = null;
    }
    if (
      flush &&
      hadPending &&
      local.mountedPlugin &&
      local.session &&
      local.session.phase !== "completed"
    ) {
      const snap = getActiveWorkspaceContext();
      if (snap) {
        pushWorkspaceSnapshot(snap, { keepalive: true });
      }
    }
    local.workspaceDirty = false;
    if (local.mountedPlugin?.destroy) {
      try {
        local.mountedPlugin.destroy();
      } catch {
        /* ignore */
      }
    }
    local.mountedPlugin = null;
    local.mountedPluginKey = null;
    const slot = el("practiceStudioSlot");
    if (slot) {
      slot.innerHTML = "";
      slot.classList.add("hidden");
    }
  }

  function getActiveWorkspaceContext() {
    if (local.mountedPlugin?.getWorkspaceContext) {
      try {
        return local.mountedPlugin.getWorkspaceContext() || null;
      } catch {
        return null;
      }
    }
    return null;
  }

  async function pushWorkspaceSnapshot(contextOverride = null, { keepalive = false } = {}) {
    if (!local.session || local.session.phase === "completed") return;
    if (local.workspaceSyncTimer) {
      clearTimeout(local.workspaceSyncTimer);
      local.workspaceSyncTimer = null;
    }
    const sid = local.session.id;
    const mySessionGen = local.sessionGen;
    const ctx = contextOverride || getActiveWorkspaceContext();
    if (!ctx) return;
    local.workspaceDirty = false;
    // Synchronously mirror onto local.session so toggling add-on off/on never remounts stale data
    local.session.workspace_context = {
      ...(local.session.workspace_context || {}),
      ...ctx,
    };
    try {
      const res = await fetch(`/api/practice/${encodeURIComponent(sid)}/workspace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspace_context: ctx }),
        keepalive,
      });
      if (!res.ok) return;
      const data = await res.json();
      if (
        mySessionGen === local.sessionGen &&
        local.session &&
        local.session.id === sid &&
        data?.workspace_context
      ) {
        local.session.workspace_context = data.workspace_context;
      }
    } catch {
      /* ignore background sync error */
    }
  }

  function scheduleWorkspaceSync() {
    local.workspaceDirty = true;
    const snap = getActiveWorkspaceContext();
    if (snap && local.session && local.session.phase !== "completed") {
      local.session.workspace_context = {
        ...(local.session.workspace_context || {}),
        ...snap,
      };
    }
    if (local.workspaceSyncTimer) clearTimeout(local.workspaceSyncTimer);
    local.workspaceSyncTimer = setTimeout(() => {
      local.workspaceSyncTimer = null;
      pushWorkspaceSnapshot();
    }, 900);
  }

  function syncStudioSlot() {
    const slot = el("practiceStudioSlot");
    if (!slot || !local.session) {
      unmountStudioPlugin();
      return;
    }
    const fmt = local.session.config?.interview_type;
    const mods = state.addons?.modules_map || {};
    const moduleAllowed =
      fmt === "system_design"
        ? mods.system_design_studio !== false
        : fmt === "coding"
        ? mods.coding_studio !== false
        : false;
    const factory = moduleAllowed ? local.studioPlugins.get(fmt) : null;
    const isActive = local.session.phase === "active" || local.session.phase === "awaiting_debrief";

    if (!factory || !isActive) {
      unmountStudioPlugin();
      return;
    }

    const pluginKey = `${local.session.id}:${fmt}`;
    if (local.mountedPluginKey === pluginKey && local.mountedPlugin) {
      local.mountedPlugin.updateSession?.(local.session);
      slot.classList.remove("hidden");
      return;
    }

    unmountStudioPlugin();
    slot.classList.remove("hidden");
    local.mountedPluginKey = pluginKey;
    local.mountedPlugin = factory({
      containerEl: slot,
      session: local.session,
      onWorkspaceChange: () => scheduleWorkspaceSync(),
      onAskAiReview: (promptText) => {
        if (local.busy) return;
        submitTurn(promptText);
      },
    });
  }

  /* ---------------- Practice Session Lifecycle ---------------- */

  function setPracticeVisible(active) {
    state.practiceActive = active;
    el("appShell")?.classList.toggle("practice-mode", active);
    el("practiceWorkspace")?.classList.toggle("hidden", !active);
    onPracticeActiveChange?.(active);
  }

  function clearDraftInputs() {
    const ans = el("practiceAnswerInput");
    if (ans) ans.value = "";
    const ret = el("practiceRetryInput");
    if (ret) ret.value = "";
  }

  function exitPracticeToHome() {
    // Flush any pending whiteboard/code edits before incrementing sessionGen & clearing session
    unmountStudioPlugin({ flush: true });
    local.sessionGen += 1;
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    clearDraftInputs();
    local.session = null;
    setBusy(false);
    setPracticeVisible(false);
    refreshPracticeHistory();
  }

  function setBusy(busy) {
    local.busy = busy;
    const ids = [
      "practiceSendBtn",
      "practiceHintBtn",
      "practiceNextQBtn",
      "practiceDebriefBtn",
      "practiceRetrySubmitBtn",
    ];
    for (const id of ids) {
      const btn = el(id);
      if (btn) btn.disabled = busy;
    }
  }

  async function startNewPractice() {
    if (local.busy) return;
    local.sessionGen += 1;
    const mySessionGen = local.sessionGen;
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    clearDraftInputs();

    const btn = el("practiceStartBtn");
    const lbl = el("practiceStartBtnLabel");
    if (btn) btn.disabled = true;
    if (lbl) lbl.textContent = "Preparing your interview…";

    const roleTitle = (el("setupRoleInput")?.value || "").trim();
    const fmt = getSelectedFormat();
    const selectedQ = el("practiceQuestionPicker")?.value || null;
    const liveNudges = Boolean(el("practiceNudgesCheckbox")?.checked ?? true);
    const voiceOn = Boolean(el("practiceVoiceCheckbox")?.checked ?? true);
    local.voiceEnabled = voiceOn;

    try {
      const res = await fetch("/api/practice/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role_title: roleTitle,
          interview_type: fmt,
          question_source: local.questionSource,
          target_level: local.targetLevel,
          coach_style: "supportive",
          live_coach_nudges: liveNudges,
          voice_enabled: voiceOn,
          question_count: fmt === "system_design" || fmt === "coding" ? 1 : 4,
          selected_question_id: selectedQ || null,
        }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Could not start practice session");
      local.session = data.session;
      setPracticeVisible(true);
      renderPracticeSession();
      if (data.opening_turn?.text && local.voiceEnabled) {
        speakText(data.opening_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3500);
      }
    } finally {
      if (btn) btn.disabled = false;
      if (lbl) lbl.textContent = "Start AI mock interview";
    }
  }

  async function openPracticeSession(sessionId) {
    local.sessionGen += 1;
    const mySessionGen = local.sessionGen;
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    clearDraftInputs();
    try {
      const res = await fetch(`/api/practice/sessions/${encodeURIComponent(sessionId)}`);
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Could not load practice session");
      local.session = data.session;
      local.voiceEnabled = Boolean(local.session.config?.voice_enabled ?? true);
      setPracticeVisible(true);
      renderPracticeSession();
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3200);
      }
    }
  }

  async function submitTurn(explicitText = null) {
    if (!local.session || local.busy || local.session.phase !== "active") return;
    const mySessionGen = local.sessionGen;
    if (local.recordingTarget === "answer") {
      await stopMicAndTranscribe();
      if (mySessionGen !== local.sessionGen || !local.session) return;
    }
    stopSpeaking();

    const inputEl = el("practiceAnswerInput");
    const text = (explicitText !== null ? String(explicitText) : inputEl?.value || "").trim();
    if (!text) return;

    if (explicitText === null && inputEl) inputEl.value = "";
    setBusy(true);

    // Optimistic render of candidate's message while AI thinks
    const feed = el("practiceFeed");
    if (feed) {
      const pendingEl = document.createElement("div");
      pendingEl.className = "pturn pturn-candidate";
      pendingEl.innerHTML = `<div class="pturn-head">You</div><div class="pturn-body">${formatWithNewlines(text)}</div>`;
      feed.appendChild(pendingEl);
      const thinkingEl = document.createElement("div");
      thinkingEl.className = "pturn pturn-interviewer is-thinking";
      thinkingEl.innerHTML = `<div class="pturn-head">AI Interviewer</div><div class="pturn-body">Thinking…</div>`;
      feed.appendChild(thinkingEl);
      feed.scrollTop = feed.scrollHeight;
    }

    try {
      const wsCtx = getActiveWorkspaceContext();
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          workspace_context: wsCtx,
          expected_question_id: currentQuestionId(),
        }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Turn failed");
      local.session = data.session;
      renderPracticeSession();
      if (data.interviewer_turn?.text && local.voiceEnabled) {
        speakText(data.interviewer_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3200);
        renderPracticeSession();
      }
    } finally {
      if (mySessionGen === local.sessionGen) {
        setBusy(false);
        el("practiceAnswerInput")?.focus();
      }
    }
  }

  async function requestHint() {
    if (!local.session || local.busy || local.session.phase !== "active") return;
    const mySessionGen = local.sessionGen;
    stopSpeaking();
    setBusy(true);
    try {
      const wsCtx = getActiveWorkspaceContext();
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/hint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspace_context: wsCtx }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Could not get hint");
      local.session = data.session;
      renderPracticeSession();
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3000);
      }
    } finally {
      if (mySessionGen === local.sessionGen) setBusy(false);
    }
  }

  async function advanceNextQuestion() {
    if (!local.session || local.busy || local.session.phase !== "active") return;
    const mySessionGen = local.sessionGen;
    stopSpeaking();
    setBusy(true);
    try {
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/next`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_question_id: currentQuestionId() }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Could not advance question");
      local.session = data.session;
      renderPracticeSession();
      if (!data.has_more) {
        showToast("All questions complete — click End & CARL Debrief when ready!", 3400);
      } else if (data.interviewer_turn?.text && local.voiceEnabled) {
        speakText(data.interviewer_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3000);
      }
    } finally {
      if (mySessionGen === local.sessionGen) setBusy(false);
    }
  }

  async function runDebrief() {
    if (!local.session || local.busy) return;
    const mySessionGen = local.sessionGen;
    stopSpeaking();
    if (local.recordingTarget) {
      await stopMicAndTranscribe();
      if (mySessionGen !== local.sessionGen || !local.session) return;
    }
    setBusy(true);
    const btn = el("practiceDebriefBtn");
    if (btn) btn.textContent = "Generating CARL coaching…";
    try {
      await pushWorkspaceSnapshot();
      if (mySessionGen !== local.sessionGen || !local.session) return;
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/debrief`, {
        method: "POST",
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Could not generate debrief");
      local.session = data.session;
      renderPracticeSession();
      refreshPracticeHistory();
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3400);
      }
    } finally {
      if (btn) btn.textContent = "End & CARL Debrief";
      if (mySessionGen === local.sessionGen) setBusy(false);
    }
  }

  /* ---------------- 45-Second Retry Weak Part Drill ---------------- */

  function stopRetryTimer() {
    if (local.retryTimerHandle) {
      clearInterval(local.retryTimerHandle);
      local.retryTimerHandle = null;
    }
    local.retryDeadlineMs = 0;
  }

  function resetRetryTimer() {
    stopRetryTimer();
    local.retrySecondsLeft = 45;
    const clock = el("practiceRetryClock");
    const btn = el("practiceRetryTimerBtn");
    if (clock) {
      clock.textContent = "0:45";
      clock.classList.remove("is-warn");
    }
    if (btn) btn.textContent = "Start 45s timer";
  }

  function toggleRetryTimer() {
    const clock = el("practiceRetryClock");
    const btn = el("practiceRetryTimerBtn");
    if (local.retryTimerHandle) {
      resetRetryTimer();
      return;
    }
    local.retrySecondsLeft = 45;
    local.retryDeadlineMs = Date.now() + 45_000;
    if (clock) {
      clock.textContent = "0:45";
      clock.classList.remove("is-warn");
    }
    if (btn) btn.textContent = "Reset timer";
    local.retryTimerHandle = setInterval(() => {
      const remSec = Math.max(0, Math.ceil((local.retryDeadlineMs - Date.now()) / 1000));
      local.retrySecondsLeft = remSec;
      if (clock) {
        clock.textContent = fmtShortClock(remSec);
        clock.classList.toggle("is-warn", remSec <= 10);
      }
      if (remSec <= 0) {
        stopRetryTimer();
        if (btn) btn.textContent = "Restart 45s";
        if (local.recordingTarget === "retry") {
          stopMicAndTranscribe();
        }
        showToast("45 seconds up! Submit your retry to get immediate before→after feedback.", 3200);
      }
    }, 250);
  }

  async function submitRetry() {
    if (!local.session || local.busy) return;
    const mySessionGen = local.sessionGen;
    if (local.recordingTarget === "retry") {
      await stopMicAndTranscribe();
      if (mySessionGen !== local.sessionGen || !local.session) return;
    }
    stopRetryTimer();
    const input = el("practiceRetryInput");
    const text = (input?.value || "").trim();
    if (!text) {
      showToast("Record or type your 45-second retry answer first.", 2600);
      return;
    }
    setBusy(true);
    const btn = el("practiceRetrySubmitBtn");
    if (btn) btn.textContent = "Evaluating retry…";
    try {
      const drill = local.session.debrief?.retry_drill || {};
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/retry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer_text: text,
          question_id: drill.question_id || null,
          focus: drill.focus || "",
        }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (!res.ok) throw new Error(data.detail || "Could not evaluate retry");
      local.session = data.session;
      if (input) input.value = "";
      resetRetryTimer();
      renderPracticeSession();
      showToast("Nice rep! Check your before → after coaching below.", 2800);
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3200);
      }
    } finally {
      if (btn) btn.textContent = "Evaluate Retry";
      if (mySessionGen === local.sessionGen) setBusy(false);
    }
  }

  /* ---------------- Rendering ---------------- */

  function renderCarlMeter(carl) {
    const signals = carl || { context: "missing", action: "missing", result: "missing", learning: "missing" };
    let score = 0;
    for (const dim of ["context", "action", "result", "learning"]) {
      const val = signals[dim] || "missing";
      if (val === "strong") score += 1;
      else if (val === "partial") score += 0.5;
      const pill = el("practiceCarlPills")?.querySelector(`[data-carl-dim="${dim}"]`);
      if (pill) {
        pill.classList.remove("is-strong", "is-partial", "is-missing");
        pill.classList.add(`is-${val}`);
        const stEl = pill.querySelector(".carl-state");
        if (stEl) stEl.textContent = CARL_STATE_LABELS[val] || "Pending";
      }
    }
    const scoreEl = el("practiceCarlScore");
    if (scoreEl) {
      scoreEl.textContent = `${Number.isInteger(score) ? score : score.toFixed(1)} / 4`;
    }
  }

  function renderCarlBadgesInline(carl) {
    const dims = [
      ["C", "context"],
      ["A", "action"],
      ["R", "result"],
      ["L", "learning"],
    ];
    return `<span class="carl-inline-strip">${dims
      .map(([letter, key]) => {
        const val = (carl && carl[key]) || "missing";
        return `<span class="carl-chip is-${escapeHtml(val)}" title="${escapeHtml(key)}: ${escapeHtml(val)}">${letter}</span>`;
      })
      .join("")}</span>`;
  }

  function renderTurnCard(t) {
    if (t.role === "interviewer") {
      return `
        <article class="pturn pturn-interviewer">
          <div class="pturn-head">
            <span>AI Interviewer</span>
            <button type="button" class="link-quiet pturn-replay-btn" data-replay-turn="${escapeHtml(t.id)}" title="Replay spoken question">Replay audio</button>
          </div>
          <div class="pturn-body pturn-serif">${formatWithNewlines(t.text)}</div>
        </article>
      `;
    }
    if (t.role === "coach") {
      return `
        <article class="pturn pturn-coach">
          <div class="pturn-head">Supportive Coach Hint</div>
          <div class="pturn-body">${formatWithNewlines(t.text)}</div>
        </article>
      `;
    }
    const retryTag = t.is_retry ? `<span class="pturn-retry-tag">Retry</span>` : "";
    const carlStrip = t.carl_signals ? renderCarlBadgesInline(t.carl_signals) : "";
    const tipHtml = t.coach_tip
      ? `<div class="pturn-coach-note">Coach: ${escapeHtml(t.coach_tip)}</div>`
      : "";
    return `
      <article class="pturn pturn-candidate">
        <div class="pturn-head">
          <span>You ${retryTag}</span>
          ${carlStrip}
        </div>
        <div class="pturn-body">${formatWithNewlines(t.text)}</div>
        ${tipHtml}
      </article>
    `;
  }

  function renderDebriefView(sess) {
    const contentEl = el("practiceDebriefContent");
    const retryCard = el("practiceRetryCard");
    const debrief = sess.debrief;
    if (!contentEl || !debrief) return;

    const strengths = Array.isArray(debrief.top_strengths) ? debrief.top_strengths : [];
    const reviews = Array.isArray(debrief.question_reviews) ? debrief.question_reviews : [];

    contentEl.innerHTML = `
      <section class="debrief-hero">
        <div class="practice-kicker">Supportive CARL Coaching Debrief</div>
        <h2 class="debrief-encouragement">${escapeHtml(debrief.overall_encouragement || "Great practice session!")}</h2>
        ${
          debrief.carl_primer
            ? `<p class="debrief-primer">${escapeHtml(debrief.carl_primer)}</p>`
            : ""
        }
      </section>

      ${
        strengths.length > 0
          ? `<section class="debrief-section">
              <h3 class="debrief-section-title">What You Did Well</h3>
              <ul class="debrief-strengths">
                ${strengths.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}
              </ul>
            </section>`
          : ""
      }

      ${
        reviews.length > 0
          ? `<section class="debrief-section">
              <h3 class="debrief-section-title">Question-by-Question CARL Breakdown</h3>
              <div class="debrief-q-list">
                ${reviews
                  .map(
                    (qr, idx) => `
                  <article class="debrief-q-card">
                    <div class="debrief-q-head">
                      <div>
                        <span class="debrief-q-num">Q${idx + 1}</span>
                        <span class="debrief-q-title">${escapeHtml(qr.title || "Practice Question")}</span>
                      </div>
                      ${renderCarlBadgesInline(qr.carl)}
                    </div>
                    ${qr.prompt ? `<p class="debrief-q-prompt">${escapeHtml(qr.prompt)}</p>` : ""}
                    ${
                      qr.strength
                        ? `<div class="debrief-q-row"><strong>Strength:</strong> ${escapeHtml(qr.strength)}</div>`
                        : ""
                    }
                    ${
                      qr.coaching_tip
                        ? `<div class="debrief-q-row"><strong>Next-Level Tip:</strong> ${escapeHtml(qr.coaching_tip)}</div>`
                        : ""
                    }
                    ${
                      qr.carl_reframe
                        ? `<div class="debrief-q-reframe">
                            <span class="practice-kicker">Before → After CARL Reframe</span>
                            <p>${escapeHtml(qr.carl_reframe)}</p>
                          </div>`
                        : ""
                    }
                  </article>
                `
                  )
                  .join("")}
              </div>
            </section>`
          : ""
      }
    `;

    // Retry Weak Part Drill
    const drill = debrief.retry_drill || {};
    if (retryCard && drill.prompt) {
      retryCard.classList.remove("hidden");
      if (el("practiceRetryFocus")) {
        el("practiceRetryFocus").textContent = drill.focus || "Sharpen your CARL finish (45s)";
      }
      if (el("practiceRetryPrompt")) {
        el("practiceRetryPrompt").textContent = drill.prompt;
      }
      const histEl = el("practiceRetryHistory");
      if (histEl) {
        const retries = sess.retry_history || [];
        histEl.innerHTML = retries
          .map((r, idx) => {
            const fb = r.feedback || {};
            return `
              <div class="retry-result-card">
                <div class="retry-result-head">
                  <span>Retry #${idx + 1}</span>
                  ${fb.carl_after ? renderCarlBadgesInline(fb.carl_after) : ""}
                </div>
                <blockquote class="retry-quote">${escapeHtml(r.candidate_answer || "")}</blockquote>
                ${fb.encouragement ? `<p class="retry-encourage">${escapeHtml(fb.encouragement)}</p>` : ""}
                ${fb.what_improved ? `<p class="retry-detail"><strong>What improved:</strong> ${escapeHtml(fb.what_improved)}</p>` : ""}
                ${fb.one_more_polish ? `<p class="retry-detail"><strong>One final polish:</strong> ${escapeHtml(fb.one_more_polish)}</p>` : ""}
              </div>
            `;
          })
          .join("");
      }
    } else if (retryCard) {
      retryCard.classList.add("hidden");
    }

    // Full transcript drawer
    const transEl = el("practiceDebriefTranscript");
    if (transEl) {
      transEl.innerHTML = (sess.turns || []).map(renderTurnCard).join("");
    }
  }

  function renderPracticeSession() {
    const sess = local.session;
    if (!sess) return;

    const isCompleted = sess.phase === "completed" && Boolean(sess.debrief);
    const isAwaitingDebrief = sess.phase === "awaiting_debrief";
    const totalQ = Math.max(1, (sess.questions || []).length);
    const curIdx = Math.min(sess.current_question_index || 0, totalQ - 1);
    const curQ = (sess.questions || [])[curIdx] || null;

    if (el("practiceRoleTitle")) {
      el("practiceRoleTitle").textContent = sess.company
        ? `${sess.role_title} · ${sess.company}`
        : sess.role_title;
    }
    if (el("practiceMetaPill")) {
      const fmt = FORMAT_LABELS[sess.config?.interview_type] || sess.config?.interview_type || "Practice";
      el("practiceMetaPill").textContent = `${fmt} · ${sess.config?.target_level || "L5"}`;
    }
    if (el("practiceProgressBadge")) {
      el("practiceProgressBadge").textContent = isCompleted
        ? "Completed · CARL Debrief"
        : isAwaitingDebrief
        ? "All Questions Done · Ready for Debrief"
        : `Q${curIdx + 1} of ${totalQ}`;
    }
    if (el("practiceVoiceToggleLabel")) {
      el("practiceVoiceToggleLabel").textContent = local.voiceEnabled ? "Voice: On" : "Voice: Off";
    }

    el("practiceHintBtn")?.classList.toggle("hidden", isCompleted || isAwaitingDebrief);
    el("practiceNextQBtn")?.classList.toggle(
      "hidden",
      isCompleted || isAwaitingDebrief || curIdx + 1 >= totalQ
    );
    el("practiceDebriefBtn")?.classList.toggle("hidden", isCompleted);

    const answerInput = el("practiceAnswerInput");
    const sendBtn = el("practiceSendBtn");
    const micBtn = el("practiceMicBtn");
    if (answerInput) {
      answerInput.disabled = isAwaitingDebrief;
      answerInput.placeholder = isAwaitingDebrief
        ? "All questions complete — click End & CARL Debrief above for your supportive coaching report."
        : "Type or speak your answer… (⌘/Ctrl+Enter to send)";
    }
    if (sendBtn && !local.busy) sendBtn.disabled = isAwaitingDebrief;
    if (micBtn && !local.recordingTarget) micBtn.disabled = isAwaitingDebrief;

    el("practiceActiveView")?.classList.toggle("hidden", isCompleted);
    el("practiceDebriefView")?.classList.toggle("hidden", !isCompleted);

    // Sync M3 / M4 embedded studio slot
    syncStudioSlot();

    // Render feed
    const feed = el("practiceFeed");
    if (feed && !isCompleted) {
      feed.innerHTML = (sess.turns || []).map(renderTurnCard).join("");
      feed.scrollTop = feed.scrollHeight;
    }

    if (isCompleted) {
      renderDebriefView(sess);
    }

    // Live CARL Meter
    renderCarlMeter(curQ?.carl_signals);

    // Live Supportive Coach card
    const coachMsgEl = el("practiceCoachMessage");
    const coachToneEl = el("practiceCoachTone");
    if (coachMsgEl && coachToneEl) {
      if (isCompleted) {
        coachToneEl.textContent = "Debrief Ready";
        coachMsgEl.textContent =
          "Review your strengths first, then try the 45-second Retry Drill to lock in the improvement.";
      } else if (isAwaitingDebrief) {
        coachToneEl.textContent = "Wrap Up";
        coachMsgEl.textContent =
          "Awesome job completing every question! Click End & CARL Debrief above for your supportive breakdown and 45-second retry drill.";
      } else if (sess.live_nudge?.message) {
        coachToneEl.textContent = sess.live_nudge.tone === "encourage" ? "Nice signal" : "Coach tip";
        coachMsgEl.textContent = sess.live_nudge.message;
      } else if (curQ?.carl_focus) {
        coachToneEl.textContent = "Focus";
        coachMsgEl.textContent = curQ.carl_focus;
      } else if (curQ?.core_requirements?.length) {
        coachToneEl.textContent = "Focus";
        coachMsgEl.textContent = `Key requirements: ${curQ.core_requirements.slice(0, 2).join(" · ")}`;
      } else {
        coachToneEl.textContent = "Ready";
        coachMsgEl.textContent =
          "Think out loud, anchor with Context, highlight what YOU drove, and close with measurable Result + Learning.";
      }
    }

    // Question Roadmap
    const roadmap = el("practiceRoadmap");
    if (roadmap) {
      roadmap.innerHTML = (sess.questions || [])
        .map((q, idx) => {
          const isCur = !isCompleted && !isAwaitingDebrief && idx === curIdx;
          const cls = isCur ? "is-active" : q.status === "completed" ? "is-done" : "";
          const badge = q.source === "tailored" ? "Tailored" : "Bank";
          return `
            <div class="roadmap-item ${cls}">
              <div class="roadmap-head">
                <span class="roadmap-num">Q${idx + 1}</span>
                <span class="roadmap-title">${escapeHtml(q.title || q.competency || "Question")}</span>
                <span class="roadmap-src">${escapeHtml(badge)}</span>
              </div>
              <div class="roadmap-carl">${renderCarlBadgesInline(q.carl_signals)}</div>
            </div>
          `;
        })
        .join("");
    }
  }

  /* ---------------- Wire Event Listeners ---------------- */

  function bindEvents() {
    el("homeModeSwitch")?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-home-mode]");
      if (!btn) return;
      setHomeMode(btn.getAttribute("data-home-mode"));
    });

    el("setupTypePills")?.addEventListener("click", () => {
      if (local.homeMode === "practice") {
        // Wait a tick for active class update
        setTimeout(() => refreshQuestionPicker(), 0);
      }
    });

    el("practiceSourcePills")?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-practice-source]");
      if (!btn) return;
      local.questionSource = btn.getAttribute("data-practice-source") || "mixed";
      el("practiceSourcePills")
        ?.querySelectorAll("[data-practice-source]")
        .forEach((b) => b.classList.toggle("active", b === btn));
      const desc = el("practiceSourceDesc");
      if (desc) desc.textContent = SOURCE_NOTES[local.questionSource] || SOURCE_NOTES.mixed;
    });

    el("practiceLevelPills")?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-practice-level]");
      if (!btn) return;
      local.targetLevel = btn.getAttribute("data-practice-level") || "L5";
      el("practiceLevelPills")
        ?.querySelectorAll("[data-practice-level]")
        .forEach((b) => b.classList.toggle("active", b === btn));
      refreshQuestionPicker();
    });

    el("practiceStartBtn")?.addEventListener("click", () => startNewPractice());

    el("homePracticeList")?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-open-practice]");
      if (!btn) return;
      openPracticeSession(btn.getAttribute("data-open-practice"));
    });

    el("practiceHomeBtn")?.addEventListener("click", () => exitPracticeToHome());
    el("practiceNewFromDebriefBtn")?.addEventListener("click", () => {
      exitPracticeToHome();
      setHomeMode("practice");
    });

    el("practiceBargeInBtn")?.addEventListener("click", () => stopSpeaking());

    el("practiceVoiceToggleBtn")?.addEventListener("click", () => {
      local.voiceEnabled = !local.voiceEnabled;
      if (!local.voiceEnabled) stopSpeaking();
      if (el("practiceVoiceToggleLabel")) {
        el("practiceVoiceToggleLabel").textContent = local.voiceEnabled ? "Voice: On" : "Voice: Off";
      }
    });

    el("practiceHintBtn")?.addEventListener("click", () => requestHint());
    el("practiceNextQBtn")?.addEventListener("click", () => advanceNextQuestion());
    el("practiceDebriefBtn")?.addEventListener("click", () => runDebrief());
    el("practiceSendBtn")?.addEventListener("click", () => submitTurn());

    el("practiceAnswerInput")?.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        submitTurn();
      }
    });

    el("practiceMicBtn")?.addEventListener("click", () => toggleMicRecording("answer"));
    el("practiceRetryMicBtn")?.addEventListener("click", () => toggleMicRecording("retry"));

    el("practiceRetryTimerBtn")?.addEventListener("click", () => toggleRetryTimer());
    el("practiceRetrySubmitBtn")?.addEventListener("click", () => submitRetry());

    el("practiceExportMdBtn")?.addEventListener("click", () => {
      if (!local.session) return;
      window.location.href = `/api/practice/${encodeURIComponent(local.session.id)}/markdown`;
    });

    el("practiceToggleTranscriptBtn")?.addEventListener("click", () => {
      const box = el("practiceDebriefTranscript");
      const btn = el("practiceToggleTranscriptBtn");
      if (!box) return;
      const nowHidden = box.classList.toggle("hidden");
      if (btn) btn.textContent = nowHidden ? "Show full transcript" : "Hide full transcript";
    });

    el("practiceFeed")?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-replay-turn]");
      if (!btn || !local.session) return;
      const tid = btn.getAttribute("data-replay-turn");
      const turn = (local.session.turns || []).find((t) => t.id === tid);
      if (turn?.text) speakText(turn.text, { force: true });
    });
  }

  bindEvents();

  return {
    applyStatusSnapshot,
    refreshPracticeHistory,
    refreshQuestionPicker,
    openPracticeSession,
    exitPracticeToHome,
    setHomeMode,
    registerStudioPlugin,
    pushWorkspaceSnapshot,
    isPracticeActive: () => Boolean(local.session),
    isHomePracticeMode: () => local.homeMode === "practice",
  };
}
