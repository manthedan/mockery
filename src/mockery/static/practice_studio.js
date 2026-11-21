/* AI Mock Practice Studio — Solo AI Interviewer, Local Voice (TTS + Parakeet Mic),
 * Live CARL Coaching Meter, Supportive CARL Debrief, 45-Second Retry Drill,
 * and extensible Studio Slot for Milestone 3 (System Design) & Milestone 4 (Coding).
 */

import {
  createPracticeMicMachine,
  createPracticeOpMachine,
  createRetryTimerMachine,
  createWorkspaceSaveMachine,
  formatApiDetail,
} from "./state_machines.js";

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

const KOKORO_VOICE_LABELS = {
  af_heart: "Heart (US · Warm)",
  af_bella: "Bella (US · Natural)",
  af_sarah: "Sarah (US · Clear)",
  am_michael: "Michael (US · Calm)",
  am_fenrir: "Fenrir (US · Deep)",
  bf_emma: "Emma (UK · Crisp)",
  bm_george: "George (UK · Warm)",
};

const PHONON_VOICE_LABELS = {
  Harper: "Harper (US · Clear)",
  Hadley: "Hadley (US · Warm)",
  Easton: "Easton (US · Calm)",
  Juniper: "Juniper (US · Bright)",
  Trevor: "Trevor (US · Steady)",
  Maeve: "Maeve (US · Expressive)",
  Miles: "Miles (US · Deep)",
  default: "Default (Offline Built-in)",
};

const VOICE_PREVIEW_TEXT =
  "Hi! I'm your local interview coach. Ready to walk through your next mock interview?";

const VOICE_PREVIEW_BY_ENGINE = {
  phonon: "Hi! I'm your Phonon interview coach running locally on your Mac. Ready to practice?",
  kokoro: "Hi! I'm your Kokoro interview coach running locally on your Mac. Ready to practice?",
};

export function createPracticeStudioController({
  state,
  el,
  escapeHtml,
  showToast,
  onPracticeActiveChange,
  clientMachines = null,
  packOps = null,
  onInstallPack = null,
}) {
  let savedVoice = "";
  try {
    savedVoice = window.localStorage?.getItem("mockery_tts_voice") || "";
  } catch {
    savedVoice = "";
  }

  const local = {
    homeMode: "copilot", // "copilot" | "practice"
    questionSource: "mixed",
    targetLevel: "L5",
    session: null,
    sessionGen: 0,
    ttsStatus: null,
    busy: false,

    // Voice playback state (with generation token + AbortController)
    voiceEnabled: true,
    selectedVoice: savedVoice,
    isSpeaking: false,
    speakingPreviewEngine: null, // null | "picker" | "phonon" | "kokoro"
    speakGen: 0,
    ttsAbort: null,
    activeAudio: null,
    activeAudioUrl: null,

    // Local mic recording state
    recordingTarget: null, // null | "answer" | "retry"
    micTarget: null,
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
    workspaceDirty: false,
    workspaceSaveSeq: 0,
    workspaceAckSeq: 0,
    debriefPollTimer: null,
    exitForceArmed: false,
    exitForceTimer: null,
  };

  const micMachine = createPracticeMicMachine(clientMachines);
  const opMachine = createPracticeOpMachine(clientMachines, (next) => {
    local.busy = next !== "idle";
    if (next === "idle") {
      clientMachines?.endOp("practice_op");
    } else {
      clientMachines?.beginOp("practice_op", { op: next, sessionId: local.session?.id || null });
    }
  });
  const retryTimerMachine = createRetryTimerMachine(clientMachines);
  const workspaceSaveMachine = createWorkspaceSaveMachine(clientMachines, () => {
    renderWorkspaceSaveStatus();
  });

  function clearExitForceArm() {
    local.exitForceArmed = false;
    if (local.exitForceTimer) {
      clearTimeout(local.exitForceTimer);
      local.exitForceTimer = null;
    }
  }

  function renderWorkspaceSaveStatus() {
    const pill = el("practiceWorkspaceSaveStatus");
    if (!pill) return;
    const show = Boolean(local.session && local.mountedPlugin && local.session.phase !== "completed");
    pill.classList.toggle("hidden", !show);
    if (!show) return;
    const st = workspaceSaveMachine.state;
    pill.classList.remove("is-saving", "is-dirty", "is-error");
    if (st === "saving") {
      pill.textContent = "Saving…";
      pill.classList.add("is-saving");
      pill.disabled = true;
      pill.title = "Saving workspace changes locally…";
    } else if (st === "dirty") {
      pill.textContent = "Unsaved";
      pill.classList.add("is-dirty");
      pill.disabled = false;
      pill.title = "Changes will autosave momentarily — click to save now";
    } else if (st === "error") {
      pill.textContent = "Save failed · Retry";
      pill.classList.add("is-error");
      pill.disabled = false;
      pill.title = "Click to retry saving workspace";
    } else {
      pill.textContent = "Saved";
      pill.disabled = true;
      pill.title = "Workspace synced locally";
    }
  }

  function resetWorkspaceSaveState() {
    if (local.workspaceSyncTimer) {
      clearTimeout(local.workspaceSyncTimer);
      local.workspaceSyncTimer = null;
    }
    clearExitForceArm();
    local.workspaceDirty = false;
    local.workspaceSaveSeq = 0;
    local.workspaceAckSeq = 0;
    workspaceSaveMachine.reset("saved", { event: "session_reset" });
    renderWorkspaceSaveStatus();
  }

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

  function buildAvailableVoiceGroups(tts = local.ttsStatus) {
    if (!tts) return [];
    const groups = [];
    const engines = tts.engines || {};
    const phononVoices =
      engines.phonon?.ready && Array.isArray(engines.phonon.voices) && engines.phonon.voices.length > 0
        ? engines.phonon.voices
        : tts.engine === "phonon" && Array.isArray(tts.voices) && tts.voices.length > 0
          ? tts.voices
          : [];
    const kokoroVoices =
      engines.kokoro?.ready && Array.isArray(engines.kokoro.voices) && engines.kokoro.voices.length > 0
        ? engines.kokoro.voices
        : tts.engine === "kokoro" && Array.isArray(tts.voices) && tts.voices.length > 0
          ? tts.voices
          : [];

    if (phononVoices.length > 0) {
      groups.push({
        engine: "phonon",
        label: "Phonon-31M (62 MB · Fast)",
        voices: phononVoices.map((v) => ({
          value: `phonon:${v}`,
          raw: v,
          label: PHONON_VOICE_LABELS[v] || v,
        })),
      });
    }
    if (kokoroVoices.length > 0) {
      groups.push({
        engine: "kokoro",
        label: "Kokoro-82M (183 MB · Studio)",
        voices: kokoroVoices.map((v) => ({
          value: `kokoro:${v}`,
          raw: v,
          label: KOKORO_VOICE_LABELS[v] || v,
        })),
      });
    }
    return groups;
  }

  function resolveEffectiveVoice() {
    const tts = local.ttsStatus;
    if (!tts) return "";
    const groups = buildAvailableVoiceGroups(tts);
    if (groups.length > 0) {
      const all = groups.flatMap((g) => g.voices);
      const saved = String(local.selectedVoice || "").trim();
      if (saved) {
        const exact = all.find((opt) => opt.value === saved);
        if (exact) return exact.value;
        // Backward compatibility with previously saved unprefixed voice IDs (e.g. "af_heart")
        const rawMatch = all.find((opt) => opt.raw === saved);
        if (rawMatch) return rawMatch.value;
      }
      const primaryGroup = groups.find((g) => g.engine === tts.engine) || groups[0];
      return primaryGroup?.voices[0]?.value || "";
    }
    if (!Array.isArray(tts.voices) || tts.voices.length === 0) return "";
    if (local.selectedVoice && tts.voices.includes(local.selectedVoice)) {
      return local.selectedVoice;
    }
    return tts.default_voice || tts.voices[0] || "";
  }

  function renderVoiceControls() {
    const tts = local.ttsStatus;
    const groups = buildAvailableVoiceGroups(tts);
    const eff = resolveEffectiveVoice();
    const activeEngine = eff.startsWith("phonon:")
      ? "phonon"
      : eff.startsWith("kokoro:")
        ? "kokoro"
        : tts?.engine || "browser";

    const voiceLbl = el("practiceVoiceLabel");
    if (voiceLbl && tts) {
      if (activeEngine === "kokoro") {
        voiceLbl.textContent = "AI speaks questions aloud (Kokoro-82M studio voice, instant barge-in)";
      } else if (activeEngine === "phonon") {
        voiceLbl.textContent = "AI speaks questions aloud (Phonon-31M fast neural voice, instant barge-in)";
      } else if (activeEngine === "macos_say") {
        voiceLbl.textContent = "AI speaks questions aloud (macOS fallback voice, instant barge-in)";
      } else {
        voiceLbl.textContent = "AI speaks questions aloud (browser voice, instant barge-in)";
      }
    }

    const upgradeRow = el("practiceKokoroUpgradeRow");
    const installPhononBtn = el("practiceInstallPhononBtn");
    const installKokoroBtn = el("practiceInstallKokoroBtn");
    const pickerRow = el("practiceVoicePickerRow");
    const voiceSel = el("practiceVoiceSelect");
    const previewBtn = el("practicePreviewVoiceBtn");
    const phononOp = packOps?.get?.("phonon_tts") || null;
    const kokoroOp = packOps?.get?.("kokoro_tts") || null;

    if (upgradeRow && tts) {
      const anyNeuralInstalled = Boolean(
        tts.neural_installed || tts.kokoro_installed || tts.phonon_installed
      );
      const anyNeuralAvailable = Boolean(tts.kokoro_available || tts.phonon_available);
      const showUpgrade = anyNeuralAvailable && !anyNeuralInstalled;
      upgradeRow.classList.toggle("hidden", !showUpgrade);

      if (installPhononBtn) {
        installPhononBtn.classList.toggle("hidden", !tts.phonon_available);
        const busy = phononOp === "installing" || phononOp === "uninstalling";
        installPhononBtn.disabled = busy;
        installPhononBtn.textContent =
          phononOp === "installing"
            ? "Downloading Phonon (62 MB)…"
            : phononOp === "uninstalling"
              ? "Removing…"
              : "Install Phonon-31M (62 MB · Fast)";
      }
      if (installKokoroBtn) {
        installKokoroBtn.classList.toggle("hidden", !tts.kokoro_available);
        const busy = kokoroOp === "installing" || kokoroOp === "uninstalling";
        installKokoroBtn.disabled = busy;
        installKokoroBtn.textContent =
          kokoroOp === "installing"
            ? "Downloading Kokoro (183 MB)…"
            : kokoroOp === "uninstalling"
              ? "Removing…"
              : "Install Kokoro-82M (183 MB · Studio)";
      }
    }

    if (pickerRow && tts) {
      const hasNeuralVoices = groups.length > 0;
      pickerRow.classList.toggle("hidden", !hasNeuralVoices);
      if (hasNeuralVoices && voiceSel) {
        const sig = groups.map((g) => `${g.engine}:${g.voices.map((v) => v.raw).join(",")}`).join("|");
        if (voiceSel.dataset.voicesSig !== sig) {
          voiceSel.innerHTML = groups
            .map(
              (g) =>
                `<optgroup label="${escapeHtml(g.label)}">${g.voices
                  .map(
                    (opt) =>
                      `<option value="${escapeHtml(opt.value)}">${escapeHtml(opt.label)}</option>`
                  )
                  .join("")}</optgroup>`
            )
            .join("");
          voiceSel.dataset.voicesSig = sig;
        }
        if (eff && voiceSel.value !== eff) {
          voiceSel.value = eff;
        }
      }
    }

    if (previewBtn) {
      previewBtn.textContent =
        local.isSpeaking && local.speakingPreviewEngine === "picker" ? "Stop preview" : "Preview voice";
    }
  }

  function applyStatusSnapshot(data) {
    if (data?.tts) {
      local.ttsStatus = data.tts;
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
        local.mountedPlugin?.updateAddons?.(data.addons);
      }
    } else if (data?.addons && local.mountedPlugin?.updateAddons) {
      local.mountedPlugin.updateAddons(data.addons);
    }
    renderVoiceControls();
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

  function setSpeakingUI(speaking, previewTag = null) {
    local.isSpeaking = speaking;
    local.speakingPreviewEngine = speaking ? previewTag : null;
    el("practiceBargeInBtn")?.classList.toggle("hidden", !speaking);
    const prevBtn = el("practicePreviewVoiceBtn");
    if (prevBtn) {
      prevBtn.textContent =
        speaking && local.speakingPreviewEngine === "picker" ? "Stop preview" : "Preview voice";
    }
    const obPhononBtn = el("onboardPhononPreviewBtn");
    if (obPhononBtn) {
      obPhononBtn.textContent =
        speaking && local.speakingPreviewEngine === "phonon" ? "Stop preview" : "Preview Phonon";
    }
    const obKokoroBtn = el("onboardKokoroPreviewBtn");
    if (obKokoroBtn) {
      obKokoroBtn.textContent =
        speaking && local.speakingPreviewEngine === "kokoro" ? "Stop preview" : "Preview Kokoro";
    }
  }

  function stopSpeaking() {
    local.speakGen += 1;
    if (local.ttsAbort) {
      try {
        local.ttsAbort.abort();
      } catch {
        /* ignore */
      }
      local.ttsAbort = null;
    }
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
    setSpeakingUI(false, null);
  }

  async function speakText(text, { force = false, voice = null, engine = null, previewTag = null } = {}) {
    const clean = String(text || "").trim();
    if (!clean) return;
    if (!local.voiceEnabled && !force) return;

    stopSpeaking();
    const myGen = local.speakGen;
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    local.ttsAbort = ctrl;
    setSpeakingUI(true, previewTag);

    const payload = { text: clean, rate: 1.0 };
    if (engine) {
      payload.engine = engine;
      if (voice) payload.voice = voice;
    } else {
      const chosenVoice = voice || resolveEffectiveVoice();
      if (chosenVoice) payload.voice = chosenVoice;
    }

    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: ctrl?.signal,
      });
      if (myGen !== local.speakGen) return;
      if (local.ttsAbort === ctrl) local.ttsAbort = null;
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
    } catch (err) {
      if (local.ttsAbort === ctrl) local.ttsAbort = null;
      if (err?.name === "AbortError" || myGen !== local.speakGen) return;
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
      if (phase === "requesting_mic") lbl.textContent = "Requesting mic…";
      else if (phase === "recording") lbl.textContent = "Stop & transcribe";
      else if (phase === "transcribing") lbl.textContent = "Transcribing…";
      else lbl.textContent = isRetry ? "Speak retry" : "Speak answer";
    }
    if (btn) {
      const lockedByPhase =
        !isRetry &&
        local.session &&
        (local.session.phase === "awaiting_debrief" || local.session.phase === "debriefing");
      btn.disabled = phase === "transcribing" || phase === "requesting_mic" || Boolean(lockedByPhase);
    }
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
    const prevTarget = local.micTarget || local.recordingTarget;
    local.recordingTarget = null;
    local.micTarget = null;
    micMachine.reset("idle", { event: "abort_mic" });
    cleanupMicResources();
    if (prevTarget) {
      updateMicUI(prevTarget, "idle", "Local mic ready");
    }
  }

  async function toggleMicRecording(target = "answer") {
    // Instant barge-in: pressing speak immediately silences the AI voice
    stopSpeaking();

    if (opMachine.state === "exiting") return;
    if (micMachine.state === "requesting_mic" || micMachine.state === "transcribing") {
      return;
    }
    if (micMachine.state === "recording" && local.recordingTarget === target) {
      await stopMicAndTranscribe();
      return;
    }
    if (micMachine.state === "recording" && local.recordingTarget && local.recordingTarget !== target) {
      await stopMicAndTranscribe();
      if (micMachine.state !== "idle") return;
    }

    if (!local.session || opMachine.state === "exiting") return;

    if (!navigator.mediaDevices?.getUserMedia) {
      showToast("Microphone capture is unavailable in this browser.", 3000);
      return;
    }

    if (!micMachine.transition("requesting_mic", { event: "request_mic" })) return;
    local.micTarget = target;
    updateMicUI(target, "requesting_mic", "Requesting microphone access…");

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
      if (
        myRecGen !== local.recGen ||
        mySessionGen !== local.sessionGen ||
        !local.session ||
        opMachine.state === "exiting"
      ) {
        for (const t of stream.getTracks()) {
          try {
            t.stop();
          } catch {
            /* ignore */
          }
        }
        if (myRecGen === local.recGen) {
          local.micTarget = null;
          micMachine.reset("idle", { event: "mic_aborted_during_exit" });
          updateMicUI(target, "idle", "Local mic ready");
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
      micMachine.transition("recording", { event: "mic_granted" });

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
        local.micTarget = null;
        micMachine.reset("idle", { event: "mic_denied" });
        updateMicUI(target, "idle", "Microphone unavailable — type your answer below");
      }
      showToast(`Microphone access denied: ${err.message || err}`, 3200);
    }
  }

  async function stopMicAndTranscribe() {
    const target = local.recordingTarget;
    if (!target || micMachine.state !== "recording") return;
    local.recGen += 1;
    const myRecGen = local.recGen;
    const mySessionGen = local.sessionGen;
    const myQuestionId = target === "answer" ? currentQuestionId() : null;
    const sampleRate = local.recAudioCtx?.sampleRate || 48000;
    const chunks = local.recChunks.slice();
    local.recordingTarget = null;
    cleanupMicResources();

    const int16 = downsampleToInt16(chunks, sampleRate, 16000);
    if (int16.length < 1600) {
      local.micTarget = null;
      micMachine.transition("idle", { event: "clip_too_short" });
      updateMicUI(target, "idle", "Recording was too short — try again or type below.");
      return;
    }

    micMachine.transition("transcribing", { event: "start_transcribe" });
    updateMicUI(target, "transcribing", "Transcribing locally with Parakeet…");
    try {
      const res = await fetch("/api/asr/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: int16.buffer,
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || myRecGen !== local.recGen) return;
      if (!res.ok) {
        throw new Error(formatApiDetail(data.detail, "Transcription failed"));
      }
      // If the user advanced to a different question while transcribing, discard stale answer audio
      if (target === "answer" && myQuestionId && currentQuestionId() !== myQuestionId) {
        local.micTarget = null;
        micMachine.transition("idle", { event: "transcribe_discarded_question_changed" });
        updateMicUI(target, "idle", "Local mic ready");
        return;
      }
      const transcript = String(data.text || "").trim();
      const inputEl = el(target === "retry" ? "practiceRetryInput" : "practiceAnswerInput");
      if (inputEl && transcript) {
        const prefix = inputEl.value.trim();
        inputEl.value = prefix ? `${prefix} ${transcript}` : transcript;
        inputEl.focus();
      }
      local.micTarget = null;
      micMachine.transition("idle", { event: "transcribe_done" });
      const ms = data.inference_ms ? ` (${Math.round(data.inference_ms)} ms local)` : "";
      updateMicUI(
        target,
        "idle",
        transcript
          ? `Transcribed locally${ms} — edit or press Send`
          : "No speech detected — try speaking closer to the mic"
      );
    } catch (err) {
      if (mySessionGen !== local.sessionGen || myRecGen !== local.recGen) return;
      local.micTarget = null;
      micMachine.transition("idle", { event: "transcribe_error" });
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
    const hadPending = Boolean(
      local.workspaceSyncTimer ||
        local.workspaceDirty ||
        local.workspaceSaveSeq > local.workspaceAckSeq
    );
    if (local.workspaceSyncTimer) {
      clearTimeout(local.workspaceSyncTimer);
      local.workspaceSyncTimer = null;
    }
    let flushPromise = Promise.resolve(true);
    if (
      flush &&
      hadPending &&
      local.mountedPlugin &&
      local.session &&
      local.session.phase !== "completed" &&
      local.session.phase !== "debriefing"
    ) {
      const snap = getActiveWorkspaceContext();
      if (snap) {
        // Use a regular fetch (not keepalive) so workspaces > 64 KiB never get silently dropped
        flushPromise = pushWorkspaceSnapshot(snap, { keepalive: false });
      }
    }
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
    renderWorkspaceSaveStatus();
    return flushPromise;
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
    if (
      !local.session ||
      local.session.phase === "completed" ||
      local.session.phase === "debriefing"
    ) {
      return false;
    }
    if (local.workspaceSyncTimer) {
      clearTimeout(local.workspaceSyncTimer);
      local.workspaceSyncTimer = null;
    }
    const sid = local.session.id;
    const mySessionGen = local.sessionGen;
    const ctx = contextOverride || getActiveWorkspaceContext();
    if (!ctx) return false;
    const saveSeq =
      contextOverride || local.workspaceSaveSeq <= local.workspaceAckSeq
        ? ++local.workspaceSaveSeq
        : local.workspaceSaveSeq;
    local.workspaceDirty = true;
    workspaceSaveMachine.transition("saving", { event: "workspace_save_start" });
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
      if (!res.ok) {
        if (mySessionGen === local.sessionGen && saveSeq === local.workspaceSaveSeq) {
          workspaceSaveMachine.transition("error", { event: "workspace_save_http_error" });
        }
        return false;
      }
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return true;
      if (saveSeq > local.workspaceAckSeq) {
        local.workspaceAckSeq = saveSeq;
      }
      // Only clear dirty and accept server snapshot if no newer local edit superseded this request
      if (saveSeq === local.workspaceSaveSeq) {
        local.workspaceDirty = false;
        clearExitForceArm();
        workspaceSaveMachine.transition("saved", { event: "workspace_save_ok" });
        if (local.session && local.session.id === sid && data?.workspace_context) {
          local.session.workspace_context = data.workspace_context;
        }
      } else if (local.workspaceAckSeq < local.workspaceSaveSeq) {
        // Only show "dirty" if a newer edit is still unacknowledged; never regress an already-saved newer edit
        workspaceSaveMachine.transition("dirty", { event: "workspace_save_superseded" });
      }
      return true;
    } catch {
      // Retain workspaceDirty = true so subsequent sync/exit retries cleanly
      if (mySessionGen === local.sessionGen && saveSeq === local.workspaceSaveSeq) {
        workspaceSaveMachine.transition("error", { event: "workspace_save_network_error" });
      }
      return false;
    }
  }

  function scheduleWorkspaceSync() {
    if (
      !local.session ||
      local.session.phase === "completed" ||
      local.session.phase === "debriefing"
    ) {
      return;
    }
    clearExitForceArm();
    local.workspaceSaveSeq += 1;
    local.workspaceDirty = true;
    workspaceSaveMachine.transition("dirty", { event: "workspace_edit" });
    const snap = getActiveWorkspaceContext();
    if (snap) {
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
    if (opMachine.state === "exiting") return;
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
    const isActive =
      local.session.phase === "active" ||
      local.session.phase === "awaiting_debrief" ||
      local.session.phase === "debriefing";

    if (!factory || !isActive) {
      unmountStudioPlugin();
      return;
    }

    const pluginKey = `${local.session.id}:${fmt}`;
    if (local.mountedPluginKey === pluginKey && local.mountedPlugin) {
      local.mountedPlugin.updateSession?.(local.session);
      slot.classList.remove("hidden");
      renderWorkspaceSaveStatus();
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
        if (local.busy || local.session?.phase !== "active") return;
        submitTurn(promptText);
      },
    });
    renderWorkspaceSaveStatus();
  }

  function stopDebriefPoll() {
    if (local.debriefPollTimer) {
      clearTimeout(local.debriefPollTimer);
      local.debriefPollTimer = null;
    }
  }

  function ensureDebriefPoll() {
    if (local.debriefPollTimer || !local.session || local.session.phase !== "debriefing") {
      return;
    }
    const sid = local.session.id;
    const myGen = local.sessionGen;
    local.debriefPollTimer = setTimeout(async () => {
      local.debriefPollTimer = null;
      if (myGen !== local.sessionGen || !local.session || local.session.id !== sid) {
        return;
      }
      if (local.session.phase !== "debriefing") return;
      try {
        const res = await fetch(`/api/practice/sessions/${encodeURIComponent(sid)}`);
        const data = await res.json().catch(() => ({}));
        if (myGen !== local.sessionGen || !local.session || local.session.id !== sid) {
          return;
        }
        if (res.ok && data.session) {
          local.session = data.session;
          renderPracticeSession();
          return;
        }
      } catch {
        /* retry while still debriefing */
      }
      if (myGen === local.sessionGen && local.session?.id === sid && local.session.phase === "debriefing") {
        ensureDebriefPoll();
      }
    }, 1200);
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

  async function exitPracticeToHome({ force = false } = {}) {
    if (opMachine.state === "exiting") return false;
    const forceNow = Boolean(force || local.exitForceArmed);
    if (!opMachine.transition("exiting", { event: "exit_practice" })) {
      opMachine.reset("exiting", { event: "exit_practice" });
    }
    setBusy(true);
    const micBtn = el("practiceMicBtn");
    const retryMicBtn = el("practiceRetryMicBtn");
    if (micBtn) micBtn.disabled = true;
    if (retryMicBtn) retryMicBtn.disabled = true;

    stopDebriefPoll();
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();

    const myGen = local.sessionGen;
    const needsFlush = Boolean(
      local.mountedPlugin &&
        local.session &&
        local.session.phase !== "completed" &&
        local.session.phase !== "debriefing" &&
        (local.workspaceSyncTimer ||
          local.workspaceDirty ||
          local.workspaceSaveSeq > local.workspaceAckSeq ||
          workspaceSaveMachine.state === "error")
    );

    if (needsFlush && !forceNow) {
      let savedOk = false;
      try {
        savedOk = Boolean(
          await Promise.race([
            pushWorkspaceSnapshot(),
            new Promise((resolve) => setTimeout(() => resolve(false), 2000)),
          ])
        );
      } catch {
        savedOk = false;
      }
      if (myGen !== local.sessionGen) {
        if (opMachine.state === "exiting") {
          opMachine.reset("idle", { event: "exit_superseded" });
          setBusy(false);
        }
        return false;
      }
      if (!savedOk) {
        clearExitForceArm();
        local.exitForceArmed = true;
        local.exitForceTimer = setTimeout(() => {
          local.exitForceArmed = false;
          local.exitForceTimer = null;
        }, 8000);
        opMachine.reset("idle", { event: "exit_blocked_unsaved" });
        setBusy(false);
        renderPracticeSession();
        showToast(
          "Could not save your latest workspace edits — click Retry Save, or click Home again to leave without saving.",
          4500
        );
        return false;
      }
    }

    if (myGen !== local.sessionGen) {
      if (opMachine.state === "exiting") {
        opMachine.reset("idle", { event: "exit_superseded" });
        setBusy(false);
      }
      return false;
    }

    // Invalidate all in-flight session requests BEFORE unmounting so nothing can remount after exit
    local.sessionGen += 1;
    clearExitForceArm();
    unmountStudioPlugin({ flush: false });
    abortMicRecording();
    resetWorkspaceSaveState();
    clearDraftInputs();
    local.session = null;
    opMachine.reset("idle", { event: "exited_practice" });
    setBusy(false);
    setPracticeVisible(false);
    refreshPracticeHistory();
    return true;
  }

  function setBusy(busy) {
    local.busy = busy;
    const phase = local.session?.phase || "active";
    const isDebriefing = phase === "debriefing";
    const isInputLocked = phase === "awaiting_debrief" || isDebriefing;
    for (const id of ["practiceSendBtn", "practiceHintBtn", "practiceNextQBtn"]) {
      const btn = el(id);
      if (btn) btn.disabled = busy || isInputLocked;
    }
    const debriefBtn = el("practiceDebriefBtn");
    if (debriefBtn) debriefBtn.disabled = busy || isDebriefing;
    const retryBtn = el("practiceRetrySubmitBtn");
    if (retryBtn) retryBtn.disabled = busy;
  }

  async function startNewPractice() {
    if (state.captureState && state.captureState !== "idle") {
      showToast("Stop live Copilot capture before starting an AI mock interview.", 3200);
      return;
    }
    if (opMachine.state !== "idle") return;
    if (!opMachine.transition("starting", { event: "start_practice" })) return;

    local.sessionGen += 1;
    const mySessionGen = local.sessionGen;
    stopDebriefPoll();
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    resetWorkspaceSaveState();
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
      if (state.captureState && state.captureState !== "idle") {
        showToast("Live Copilot recording started — cancelled entering Practice Mode.", 3200);
        return;
      }
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not start practice session"));
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
      if (opMachine.state === "starting") {
        opMachine.transition("idle", { event: "start_practice_done" });
      }
      if (mySessionGen === local.sessionGen) {
        setBusy(false);
      }
    }
  }

  async function openPracticeSession(sessionId) {
    if (state.captureState && state.captureState !== "idle") {
      showToast("Stop live Copilot capture before opening a Practice session.", 3200);
      return;
    }
    if (opMachine.state !== "idle") return;
    if (!opMachine.transition("opening", { event: "open_practice" })) return;

    local.sessionGen += 1;
    const mySessionGen = local.sessionGen;
    stopDebriefPoll();
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    resetWorkspaceSaveState();
    clearDraftInputs();
    try {
      const res = await fetch(`/api/practice/sessions/${encodeURIComponent(sessionId)}`);
      const data = await res.json();
      if (mySessionGen !== local.sessionGen) return;
      if (state.captureState && state.captureState !== "idle") {
        showToast("Live Copilot recording started — cancelled opening Practice session.", 3200);
        return;
      }
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not load practice session"));
      local.session = data.session;
      local.voiceEnabled = Boolean(local.session.config?.voice_enabled ?? true);
      setPracticeVisible(true);
      renderPracticeSession();
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(String(err.message || err), 3200);
      }
    } finally {
      if (opMachine.state === "opening") {
        opMachine.transition("idle", { event: "open_practice_done" });
      }
      if (mySessionGen === local.sessionGen) {
        setBusy(false);
      }
    }
  }

  async function submitTurn(explicitText = null) {
    if (!local.session || opMachine.state !== "idle" || local.session.phase !== "active") return;
    if (micMachine.state === "requesting_mic" || micMachine.state === "transcribing") {
      showToast("Finishing voice capture — try again in a moment.", 2000);
      return;
    }
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("submitting_turn", { event: "submit_turn" })) return;
    setBusy(true);

    try {
      if (micMachine.state === "recording" && local.recordingTarget === "answer") {
        await stopMicAndTranscribe();
        if (mySessionGen !== local.sessionGen || !local.session || opMachine.state === "exiting") return;
      }
      stopSpeaking();

      const inputEl = el("practiceAnswerInput");
      const text = (explicitText !== null ? String(explicitText) : inputEl?.value || "").trim();
      if (!text) return;

      if (explicitText === null && inputEl) inputEl.value = "";

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
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Turn failed"));
      local.session = data.session;
      renderPracticeSession();
      if (data.interviewer_turn?.text && local.voiceEnabled) {
        speakText(data.interviewer_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(String(err.message || err), 3200);
        renderPracticeSession();
      }
    } finally {
      if (opMachine.state === "submitting_turn") {
        opMachine.transition("idle", { event: "submit_turn_done" });
      }
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        setBusy(false);
        el("practiceAnswerInput")?.focus();
      }
    }
  }

  async function requestHint() {
    if (!local.session || opMachine.state !== "idle" || local.session.phase !== "active") return;
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("requesting_hint", { event: "request_hint" })) return;
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
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not get hint"));
      local.session = data.session;
      renderPracticeSession();
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(String(err.message || err), 3000);
      }
    } finally {
      if (opMachine.state === "requesting_hint") {
        opMachine.transition("idle", { event: "request_hint_done" });
      }
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") setBusy(false);
    }
  }

  async function advanceNextQuestion() {
    if (!local.session || opMachine.state !== "idle" || local.session.phase !== "active") return;
    if (micMachine.state === "requesting_mic" || micMachine.state === "transcribing") {
      showToast("Finishing voice capture — try again in a moment.", 2000);
      return;
    }
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("advancing", { event: "advance_question" })) return;
    stopSpeaking();
    abortMicRecording();
    setBusy(true);
    try {
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/next`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_question_id: currentQuestionId() }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not advance question"));
      const ans = el("practiceAnswerInput");
      if (ans) ans.value = "";
      local.session = data.session;
      renderPracticeSession();
      if (!data.has_more) {
        showToast("All questions complete — click End & CARL Debrief when ready!", 3400);
      } else if (data.interviewer_turn?.text && local.voiceEnabled) {
        speakText(data.interviewer_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(String(err.message || err), 3000);
      }
    } finally {
      if (opMachine.state === "advancing") {
        opMachine.transition("idle", { event: "advance_question_done" });
      }
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") setBusy(false);
    }
  }

  async function runDebrief() {
    if (!local.session || opMachine.state !== "idle") return;
    if (local.session.phase === "debriefing" || local.session.phase === "completed") return;
    if (micMachine.state === "requesting_mic" || micMachine.state === "transcribing") {
      showToast("Finishing voice capture — try again in a moment.", 2000);
      return;
    }
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("debriefing", { event: "run_debrief" })) return;
    setBusy(true);
    stopSpeaking();
    const btn = el("practiceDebriefBtn");
    if (btn) btn.textContent = "Generating CARL coaching…";
    try {
      if (micMachine.state === "recording" && local.recordingTarget) {
        await stopMicAndTranscribe();
        if (mySessionGen !== local.sessionGen || !local.session || opMachine.state === "exiting") return;
      }
      if (local.mountedPlugin) {
        const savedOk = await pushWorkspaceSnapshot();
        if (mySessionGen !== local.sessionGen || !local.session || opMachine.state === "exiting") return;
        if (!savedOk && (local.workspaceDirty || workspaceSaveMachine.state === "error")) {
          showToast(
            "Could not save your latest workspace edits — retry saving before generating the debrief.",
            4000
          );
          return;
        }
      }
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/debrief`, {
        method: "POST",
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not generate debrief"));
      local.session = data.session;
      renderPracticeSession();
      refreshPracticeHistory();
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(String(err.message || err), 3400);
      }
    } finally {
      if (btn && local.session?.phase !== "debriefing") {
        btn.textContent = "End & CARL Debrief";
      }
      if (opMachine.state === "debriefing") {
        opMachine.transition("idle", { event: "run_debrief_done" });
      }
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") setBusy(false);
    }
  }

  /* ---------------- 45-Second Retry Weak Part Drill ---------------- */

  function stopRetryTimer({ nextState = "idle" } = {}) {
    if (local.retryTimerHandle) {
      clearInterval(local.retryTimerHandle);
      local.retryTimerHandle = null;
    }
    local.retryDeadlineMs = 0;
    if (retryTimerMachine.state !== nextState) {
      if (!retryTimerMachine.transition(nextState, { event: `retry_timer_${nextState}` })) {
        retryTimerMachine.reset(nextState, { event: `retry_timer_${nextState}` });
      }
    }
  }

  function resetRetryTimer() {
    stopRetryTimer({ nextState: "idle" });
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
    if (retryTimerMachine.state === "running" || retryTimerMachine.state === "warning") {
      resetRetryTimer();
      return;
    }
    if (!retryTimerMachine.transition("running", { event: "start_retry_timer" })) {
      retryTimerMachine.reset("running", { event: "start_retry_timer" });
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
      if (remSec <= 10 && remSec > 0 && retryTimerMachine.state === "running") {
        retryTimerMachine.transition("warning", { event: "retry_timer_warning" });
      }
      if (remSec <= 0) {
        stopRetryTimer({ nextState: "expired" });
        if (btn) btn.textContent = "Restart 45s";
        if (micMachine.state === "recording" && local.recordingTarget === "retry") {
          stopMicAndTranscribe();
        }
        showToast("45 seconds up! Submit your retry to get immediate before→after feedback.", 3200);
      }
    }, 250);
  }

  async function submitRetry() {
    if (!local.session || opMachine.state !== "idle") return;
    if (micMachine.state === "requesting_mic" || micMachine.state === "transcribing") {
      showToast("Finishing voice capture — try again in a moment.", 2000);
      return;
    }
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("submitting_retry", { event: "submit_retry" })) return;
    setBusy(true);
    const btn = el("practiceRetrySubmitBtn");
    try {
      if (micMachine.state === "recording" && local.recordingTarget === "retry") {
        await stopMicAndTranscribe();
        if (mySessionGen !== local.sessionGen || !local.session) return;
      }
      stopRetryTimer({ nextState: "idle" });
      const input = el("practiceRetryInput");
      const text = (input?.value || "").trim();
      if (!text) {
        showToast("Record or type your 45-second retry answer first.", 2600);
        return;
      }
      if (btn) btn.textContent = "Evaluating retry…";
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
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not evaluate retry"));
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
      if (opMachine.state === "submitting_retry") {
        opMachine.transition("idle", { event: "submit_retry_done" });
      }
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
    if (!sess || opMachine.state === "exiting") return;

    const isCompleted = sess.phase === "completed" && Boolean(sess.debrief);
    const isDebriefing = sess.phase === "debriefing";
    const isAwaitingDebrief = sess.phase === "awaiting_debrief";
    const isInputLocked = isAwaitingDebrief || isDebriefing;
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
        : isDebriefing
        ? "Generating CARL Debrief…"
        : isAwaitingDebrief
        ? "All Questions Done · Ready for Debrief"
        : `Q${curIdx + 1} of ${totalQ}`;
    }
    if (el("practiceVoiceToggleLabel")) {
      el("practiceVoiceToggleLabel").textContent = local.voiceEnabled ? "Voice: On" : "Voice: Off";
    }

    el("practiceHintBtn")?.classList.toggle("hidden", isCompleted || isInputLocked);
    el("practiceNextQBtn")?.classList.toggle(
      "hidden",
      isCompleted || isInputLocked || curIdx + 1 >= totalQ
    );
    const debriefBtn = el("practiceDebriefBtn");
    if (debriefBtn) {
      debriefBtn.classList.toggle("hidden", isCompleted);
      if (isDebriefing) {
        debriefBtn.disabled = true;
        debriefBtn.textContent = "Generating debrief…";
      } else if (opMachine.state === "idle") {
        debriefBtn.disabled = false;
        debriefBtn.textContent = "End & CARL Debrief";
      }
    }

    const answerInput = el("practiceAnswerInput");
    const sendBtn = el("practiceSendBtn");
    const micBtn = el("practiceMicBtn");
    if (answerInput) {
      answerInput.disabled = isInputLocked;
      answerInput.placeholder = isDebriefing
        ? "Generating your supportive CARL debrief — please hold on…"
        : isAwaitingDebrief
        ? "All questions complete — click End & CARL Debrief above for your supportive coaching report."
        : "Type or speak your answer… (⌘/Ctrl+Enter to send)";
    }
    if (sendBtn && opMachine.state === "idle") sendBtn.disabled = isInputLocked;
    // Never re-enable mic button while requesting permission or transcribing
    if (micBtn && micMachine.state === "idle") {
      micBtn.disabled = isInputLocked;
    }

    el("practiceActiveView")?.classList.toggle("hidden", isCompleted);
    el("practiceDebriefView")?.classList.toggle("hidden", !isCompleted);

    // Sync M3 / M4 embedded studio slot & save pill
    syncStudioSlot();
    renderWorkspaceSaveStatus();

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
      } else if (isDebriefing) {
        coachToneEl.textContent = "Debriefing";
        coachMsgEl.textContent =
          "Synthesizing your strengths, CARL breakdown, and 45-second retry drill…";
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

    if (isDebriefing) {
      ensureDebriefPoll();
    } else {
      stopDebriefPoll();
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

  function getStateSnapshot() {
    const sess = local.session;
    const mountedSnapshot =
      typeof local.mountedPlugin?.getStateSnapshot === "function"
        ? local.mountedPlugin.getStateSnapshot()
        : null;
    return {
      active: Boolean(sess),
      practiceEnabled: state.addons?.modules_map?.practice_mode !== false,
      homeMode: local.homeMode,
      sessionId: sess?.id || null,
      phase: sess?.phase || null,
      interviewType: sess?.config?.interview_type || null,
      opState: opMachine.state,
      micState: micMachine.state,
      micTarget: local.micTarget,
      voiceEnabled: local.voiceEnabled,
      isSpeaking: local.isSpeaking,
      retryTimerState: retryTimerMachine.state,
      retrySecondsLeft: local.retrySecondsLeft,
      workspaceSaveState: workspaceSaveMachine.state,
      workspaceDirty: local.workspaceDirty,
      mountedStudioKey: local.mountedPluginKey,
      mountedStudio: mountedSnapshot,
    };
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
    el("practiceNewFromDebriefBtn")?.addEventListener("click", async () => {
      const exited = await exitPracticeToHome();
      if (exited) setHomeMode("practice");
    });

    el("practiceWorkspaceSaveStatus")?.addEventListener("click", () => {
      if (workspaceSaveMachine.state === "error" || workspaceSaveMachine.state === "dirty") {
        pushWorkspaceSnapshot();
      }
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

    el("practiceInstallPhononBtn")?.addEventListener("click", () => {
      if (typeof onInstallPack === "function") {
        onInstallPack("phonon_tts");
      }
    });

    el("practiceInstallKokoroBtn")?.addEventListener("click", () => {
      if (typeof onInstallPack === "function") {
        onInstallPack("kokoro_tts");
      }
    });

    el("practiceVoiceSelect")?.addEventListener("change", (ev) => {
      const val = String(ev.target?.value || "").trim();
      if (!val) return;
      local.selectedVoice = val;
      try {
        window.localStorage?.setItem("mockery_tts_voice", val);
      } catch {
        /* ignore */
      }
      renderVoiceControls();
    });

    el("practicePreviewVoiceBtn")?.addEventListener("click", () => {
      previewVoice();
    });
  }

  function previewVoice({ engine = null } = {}) {
    const tag = engine || "picker";
    if (local.isSpeaking && local.speakingPreviewEngine === tag) {
      stopSpeaking();
      return;
    }
    const effEngine =
      engine ||
      (resolveEffectiveVoice().startsWith("phonon:")
        ? "phonon"
        : resolveEffectiveVoice().startsWith("kokoro:")
          ? "kokoro"
          : null);
    const text = VOICE_PREVIEW_BY_ENGINE[effEngine] || VOICE_PREVIEW_TEXT;
    speakText(text, { force: true, engine: engine || null, previewTag: tag });
  }

  if (packOps && typeof packOps.subscribe === "function") {
    packOps.subscribe(() => {
      renderVoiceControls();
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
    previewVoice,
    stopSpeaking,
    getStateSnapshot,
    isPracticeActive: () => Boolean(local.session),
    isPracticeStartingOrActive: () =>
      Boolean(local.session) || opMachine.state === "starting" || opMachine.state === "opening",
    isHomePracticeMode: () => local.homeMode === "practice",
  };
}
