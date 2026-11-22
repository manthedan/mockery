/* AI Mock Practice Studio — Solo AI Interviewer, Local Voice (TTS + Parakeet Mic),
 * Live CARL Coaching Meter, Supportive CARL Debrief, 45-Second Retry Drill,
 * and extensible Studio Slot for System Design & Coding studios.
 */

import {
  createPracticeMicMachine,
  createPracticeOpMachine,
  createRetryTimerMachine,
  createWorkspaceSaveMachine,
  formatApiDetail,
} from "./state_machines.js";
import { jdRoleLabel } from "./profile_addons.js";

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

// Once a session is over nothing is "pending" any more — an unfilled signal is simply missing.
const CARL_STATE_LABELS_FINAL = {
  strong: "Strong",
  partial: "Partial",
  missing: "Missing",
};

const CARL_DIMS = ["context", "action", "result", "learning"];

// The four-signal data model is shared by every format; only the words a person reads change.
const CARL_RUBRIC = {
  kicker: "Live CARL Meter",
  help: "Context → Action (what you drove) → Result (metric or outcome) → Learning.",
  dims: {
    context: ["C", "Context"],
    action: ["A", "Action"],
    result: ["R", "Result"],
    learning: ["L", "Learning"],
  },
};

const RUBRIC_BY_FORMAT = {
  behavioral: CARL_RUBRIC,
  general: CARL_RUBRIC,
  system_design: {
    kicker: "Live Design Signals",
    help: "Requirements (scope & constraints) → Architecture (components & data flow) → Scale (bottlenecks & numbers) → Tradeoffs (what you chose and why).",
    dims: {
      context: ["R", "Requirements"],
      action: ["A", "Architecture"],
      result: ["S", "Scale"],
      learning: ["T", "Tradeoffs"],
    },
  },
  coding: {
    kicker: "Live Coding Signals",
    help: "Approach (plan before code) → Correctness (working solution) → Complexity (time & space) → Edge cases (tested boundaries).",
    dims: {
      context: ["A", "Approach"],
      action: ["C", "Correctness"],
      result: ["O", "Complexity"],
      learning: ["E", "Edge cases"],
    },
  },
};

function rubricFor(fmt) {
  return RUBRIC_BY_FORMAT[fmt] || CARL_RUBRIC;
}

function carlScore(signals) {
  let score = 0;
  for (const dim of CARL_DIMS) {
    const val = signals?.[dim] || "missing";
    if (val === "strong") score += 1;
    else if (val === "partial") score += 0.5;
  }
  return score;
}

function fmtScore(score) {
  return Number.isInteger(score) ? String(score) : score.toFixed(1);
}

function fmtRelativeTime(iso) {
  const t = Date.parse(iso || "");
  if (!Number.isFinite(t)) return "";
  const diffSec = Math.max(0, (Date.now() - t) / 1000);
  if (diffSec < 60) return "now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 86400 * 7) return `${Math.floor(diffSec / 86400)}d ago`;
  try {
    return new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

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
  onHomeModeChange = null,
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

    // Voice playback state (with generation token + AbortController + Web Audio streaming)
    voiceEnabled: true,
    selectedVoice: savedVoice,
    isSpeaking: false,
    speakingState: "idle", // "idle" | "preparing" | "speaking"
    speakingPreviewEngine: null, // null | "picker" | "phonon" | "kokoro"
    speakGen: 0,
    ttsAbort: null,
    ttsAudioCtx: null,
    activeStreamSources: [],
    streamDoneTimer: null,
    activeAudio: null,
    activeAudioUrl: null,
    pendingOpTimer: null,
    pendingOpNodes: [],
    opAbortController: null,
    debriefStartedAt: 0,
    debriefConfirmTimer: null,
    debriefConfirmArmed: false,
    nextQConfirmTimer: null,
    nextQConfirmArmed: false,

    // Local mic recording state
    recordingTarget: null, // null | "answer" | "retry"
    micTarget: null,
    lastFailedClip: null, // { target, questionId, int16 }
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
    savePendingSince: 0, // when the current unsaved stretch began (0 = everything is saved)
    saveStatusTimer: null,
    workspaceSaveSeq: 0,
    workspaceAckSeq: 0,
    debriefPollTimer: null,

    // Split studio: collapsible conversation column (never persisted — the first
    // question and the debrief progress live there, so each session opens expanded)
    convCollapsed: false,
    convSeenTurns: 0,
    convSessionId: null,
    convPhase: null,

    // Home launch form
    levelTouched: false,
    pickerSeq: 0,
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

  const SAVE_STATUS_GRACE_MS = 1600;

  function renderWorkspaceSaveStatus() {
    const pill = el("practiceWorkspaceSaveStatus");
    if (!pill) return;
    const show = Boolean(local.session && local.mountedPlugin && local.session.phase !== "completed");
    pill.classList.toggle("hidden", !show);
    if (!show) return;
    const st = workspaceSaveMachine.state;
    pill.classList.remove("is-saving", "is-dirty", "is-error");

    // A routine autosave lands in about a second, so it is not news: the pill only
    // leaves "Saved" when a save has been pending for longer than a beat (a long
    // typing run, a slow write, a failure). No flicker on every pause in typing.
    const pending = st === "dirty" || st === "saving";
    if (!pending) {
      local.savePendingSince = 0;
      if (local.saveStatusTimer) {
        clearTimeout(local.saveStatusTimer);
        local.saveStatusTimer = null;
      }
    } else {
      if (!local.savePendingSince) local.savePendingSince = Date.now();
      const waited = Date.now() - local.savePendingSince;
      if (waited < SAVE_STATUS_GRACE_MS) {
        if (!local.saveStatusTimer) {
          local.saveStatusTimer = setTimeout(() => {
            local.saveStatusTimer = null;
            renderWorkspaceSaveStatus();
          }, SAVE_STATUS_GRACE_MS - waited + 16);
        }
        pill.textContent = "Saved";
        pill.disabled = true;
        pill.title = "Changes save automatically on this Mac";
        return;
      }
    }

    if (st === "saving") {
      pill.textContent = "Saving…";
      pill.classList.add("is-saving");
      pill.disabled = true;
      pill.title = "Saving workspace changes locally…";
    } else if (st === "dirty") {
      pill.textContent = "Autosaving…";
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
    local.savePendingSince = 0;
    if (local.saveStatusTimer) {
      clearTimeout(local.saveStatusTimer);
      local.saveStatusTimer = null;
    }
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

  // Interviewers and candidates name identifiers in markdown backticks. Escape first,
  // then show them as real inline code — never as raw ticks in serif prose.
  function formatInline(text) {
    return escapeHtml(String(text || "")).replace(
      /`([^`\n]+)`/g,
      '<code class="inline-code">$1</code>'
    );
  }

  function formatWithNewlines(text) {
    return formatInline(text).replace(/\n/g, "<br>");
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
    if (kicker) kicker.textContent = isPractice ? "Practicing for" : "New interview";

    const cap = el("homeModeCaption");
    if (cap) {
      cap.textContent = isPractice
        ? "A solo AI interviewer with supportive coaching"
        : "Live transcript & probes while you interview someone";
    }
    el("setupLaunchpad")?.setAttribute("data-home-mode", next);

    // Let the Home target-job line re-align role and level before the picker loads.
    if (typeof onHomeModeChange === "function") onHomeModeChange(next);
    if (isPractice) {
      refreshQuestionPicker();
    }
  }

  /* One owner for the target level: the pills, the question picker and the start
     payload all read it. An automatic alignment (from the active target job) never
     overrides a level the user picked by hand. Returns true when the level changed. */
  function setTargetLevel(level, { auto = false } = {}) {
    const pills = el("practiceLevelPills");
    const btn = pills?.querySelector(`[data-practice-level="${CSS.escape(String(level || ""))}"]`);
    if (!btn || (auto && local.levelTouched) || local.targetLevel === level) return false;
    local.targetLevel = level;
    pills.querySelectorAll("[data-practice-level]").forEach((b) => b.classList.toggle("active", b === btn));
    refreshQuestionPicker();
    return true;
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
        label: "Phonon · Fast",
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
        label: "Kokoro · Studio",
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
      // One calm line; the engine in use is shown by the status line or voice picker beneath it.
      voiceLbl.textContent = "Interviewer reads questions aloud";
      voiceLbl.dataset.engine = activeEngine;
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
      const voiceNote = upgradeRow.querySelector(".practice-voice-note");
      if (voiceNote) {
        voiceNote.textContent = activeEngine === "macos_say" ? "Using macOS voice" : "Using browser voice";
      }

      if (installPhononBtn) {
        installPhononBtn.classList.toggle("hidden", !tts.phonon_available);
        const busy = phononOp === "installing" || phononOp === "uninstalling";
        installPhononBtn.disabled = busy;
        installPhononBtn.classList.toggle("is-busy", busy);
        installPhononBtn.textContent =
          phononOp === "installing"
            ? "Downloading…"
            : phononOp === "uninstalling"
              ? "Removing…"
              : "Phonon · 62 MB";
      }
      if (installKokoroBtn) {
        installKokoroBtn.classList.toggle("hidden", !tts.kokoro_available);
        const busy = kokoroOp === "installing" || kokoroOp === "uninstalling";
        installKokoroBtn.disabled = busy;
        installKokoroBtn.classList.toggle("is-busy", busy);
        installKokoroBtn.textContent =
          kokoroOp === "installing"
            ? "Downloading…"
            : kokoroOp === "uninstalling"
              ? "Removing…"
              : "Kokoro · 183 MB";
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
    // Level and format can change back to back; only the latest request may paint.
    const seq = ++local.pickerSeq;
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
      if (seq !== local.pickerSeq) return;
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
        .slice(0, 12)
        .map((s) => {
          const role = s.company ? `${s.role_title} · ${s.company}` : s.role_title;
          const fmtKey = FORMAT_LABELS[s.interview_type] ? s.interview_type : "general";
          const fmt = FORMAT_LABELS[s.interview_type] || s.interview_type || "Practice";
          // The question theme is what tells two sessions for the same role apart.
          const theme = String(s.focus_title || "").trim();
          const title = theme || `${fmt} practice`;
          const qCount = Number(s.question_count || 0);
          const answered = Number(s.answered_count || 0);
          const score = Number.isFinite(s.score) ? Number(s.score) : null;
          let pillCls = "is-progress";
          let pillText = qCount > 1 ? `Q${Math.min(answered + 1, qCount)} of ${qCount}` : "In progress";
          if (s.has_debrief || s.phase === "completed") {
            pillCls = "is-done";
            pillText = score !== null ? `${fmtScore(score)} / 4` : "Debriefed";
          } else if (s.phase === "awaiting_debrief" || s.phase === "debriefing") {
            pillCls = "is-ready";
            pillText = "Ready to debrief";
          }
          const when = fmtRelativeTime(s.created_at_iso);
          const more = theme && qCount > 1 ? ` +${qCount - 1}` : "";
          return `
            <button type="button" class="home-item practice-history-item" data-open-practice="${escapeHtml(s.id)}" title="${escapeHtml(`${title} — ${role}`)}">
              <span class="home-item-top">
                <span class="home-item-title">${escapeHtml(title)}${more ? `<span class="ph-more">${escapeHtml(more)}</span>` : ""}</span>
                <span class="session-time">${escapeHtml(when)}</span>
              </span>
              <span class="home-item-meta ph-meta">
                <span class="ph-badge ph-badge-${escapeHtml(fmtKey)}">${escapeHtml(fmt)}</span>
                <span class="ph-pill ${pillCls}">${escapeHtml(pillText)}</span>
                <span class="session-meta-main">${escapeHtml(`${role} · ${s.target_level || "L5"}`)}</span>
              </span>
            </button>
          `;
        })
        .join("");
    } catch {
      /* ignore */
    }
  }

  /* ---------------- Local Voice Playback & Instant Barge-In ---------------- */

  function setSpeakingUI(mode, previewTag = null) {
    const stateMode =
      mode === "preparing" ? "preparing" : mode === true || mode === "speaking" ? "speaking" : "idle";
    const active = stateMode !== "idle";
    local.speakingState = stateMode;
    local.isSpeaking = active;
    local.speakingPreviewEngine = active ? previewTag : null;

    const bargeBtn = el("practiceBargeInBtn");
    if (bargeBtn) {
      bargeBtn.classList.toggle("hidden", !active);
      bargeBtn.classList.toggle("is-preparing", stateMode === "preparing");
    }
    const bargeLbl = el("practiceBargeInLabel");
    if (bargeLbl) {
      bargeLbl.textContent = stateMode === "preparing" ? "Preparing voice…" : "Stop speaking";
    }

    const previewMap = [
      ["practicePreviewVoiceBtn", "picker", "Preview voice"],
      ["onboardPhononPreviewBtn", "phonon", "Preview Phonon"],
      ["onboardKokoroPreviewBtn", "kokoro", "Preview Kokoro"],
    ];
    for (const [id, tag, idleLabel] of previewMap) {
      const btn = el(id);
      if (!btn) continue;
      const isThis = active && local.speakingPreviewEngine === tag;
      btn.classList.toggle("is-loading", isThis && stateMode === "preparing");
      btn.textContent = isThis
        ? stateMode === "preparing"
          ? "Loading voice…"
          : "Stop preview"
        : idleLabel;
    }
  }

  async function getTtsAudioContext() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    if (!local.ttsAudioCtx || local.ttsAudioCtx.state === "closed") {
      try {
        local.ttsAudioCtx = new AudioCtx();
      } catch {
        return null;
      }
    }
    if (local.ttsAudioCtx.state === "suspended") {
      try {
        await local.ttsAudioCtx.resume();
      } catch {
        /* autoplay policy may block until user gesture; caller can fall back */
      }
    }
    return local.ttsAudioCtx;
  }

  function stopSpeaking() {
    local.speakGen += 1;
    if (local.streamDoneTimer) {
      window.clearTimeout(local.streamDoneTimer);
      local.streamDoneTimer = null;
    }
    if (local.ttsAbort) {
      try {
        local.ttsAbort.abort();
      } catch {
        /* ignore */
      }
      local.ttsAbort = null;
    }
    if (local.activeStreamSources.length > 0) {
      const nodes = local.activeStreamSources.splice(0, local.activeStreamSources.length);
      for (const src of nodes) {
        try {
          src.onended = null;
          src.stop();
          src.disconnect();
        } catch {
          /* ignore */
        }
      }
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

  async function tryStreamSpeak(payload, myGen, ctrl) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx || typeof ReadableStream === "undefined") return false;

    const res = await fetch("/api/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctrl?.signal,
    });
    if (myGen !== local.speakGen) {
      try {
        res.body?.cancel();
      } catch {
        /* ignore */
      }
      return true;
    }
    if (!res.ok || !res.body) return false;

    const sampleRate = parseInt(res.headers.get("X-TTS-Sample-Rate") || "24000", 10) || 24000;
    const ctx = await getTtsAudioContext();
    if (!ctx || myGen !== local.speakGen) {
      try {
        res.body.cancel();
      } catch {
        /* ignore */
      }
      return myGen !== local.speakGen;
    }

    const reader = res.body.getReader();
    let carryByte = null;
    let nextStartTime = 0;
    let scheduledCount = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (myGen !== local.speakGen) {
          reader.cancel().catch(() => {});
          return true;
        }
        if (done) break;
        if (!value || value.byteLength === 0) continue;

        let chunk = value;
        if (carryByte !== null) {
          const merged = new Uint8Array(1 + value.byteLength);
          merged[0] = carryByte;
          merged.set(value, 1);
          chunk = merged;
          carryByte = null;
        }
        if ((chunk.byteLength & 1) === 1) {
          carryByte = chunk[chunk.byteLength - 1];
          chunk = chunk.subarray(0, chunk.byteLength - 1);
        }
        const numSamples = chunk.byteLength >> 1;
        if (numSamples === 0) continue;

        const floats = new Float32Array(numSamples);
        const dv = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
        for (let i = 0; i < numSamples; i++) {
          floats[i] = dv.getInt16(i * 2, true) / 32768.0;
        }

        const audioBuf = ctx.createBuffer(1, numSamples, sampleRate);
        audioBuf.copyToChannel(floats, 0);
        const src = ctx.createBufferSource();
        src.buffer = audioBuf;
        src.connect(ctx.destination);

        if (ctx.state === "suspended") {
          try {
            await ctx.resume();
          } catch {
            /* ignore */
          }
        }
        if (myGen !== local.speakGen) {
          reader.cancel().catch(() => {});
          return true;
        }

        const startAt = Math.max(ctx.currentTime + 0.015, nextStartTime);
        nextStartTime = startAt + audioBuf.duration;
        local.activeStreamSources.push(src);
        src.onended = () => {
          const idx = local.activeStreamSources.indexOf(src);
          if (idx >= 0) local.activeStreamSources.splice(idx, 1);
          try {
            src.disconnect();
          } catch {
            /* ignore */
          }
        };
        src.start(startAt);
        scheduledCount += 1;
        if (scheduledCount === 1 && myGen === local.speakGen) {
          setSpeakingUI("speaking", local.speakingPreviewEngine);
        }
      }
    } finally {
      if (local.ttsAbort === ctrl) local.ttsAbort = null;
    }

    if (myGen !== local.speakGen) return true;
    if (scheduledCount === 0) return false;

    const remainingMs = Math.max(0, Math.ceil((nextStartTime - ctx.currentTime) * 1000)) + 40;
    local.streamDoneTimer = window.setTimeout(() => {
      if (myGen === local.speakGen) stopSpeaking();
    }, remainingMs);
    return true;
  }

  async function speakText(text, { force = false, voice = null, engine = null, previewTag = null } = {}) {
    const clean = String(text || "").trim();
    if (!clean) return;
    if (!local.voiceEnabled && !force) return;

    stopSpeaking();
    const myGen = local.speakGen;
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    local.ttsAbort = ctrl;
    setSpeakingUI("preparing", previewTag);

    const payload = { text: clean, rate: 1.0 };
    if (engine) {
      payload.engine = engine;
      if (voice) payload.voice = voice;
    } else {
      const chosenVoice = voice || resolveEffectiveVoice();
      if (chosenVoice) payload.voice = chosenVoice;
    }

    // 1. Primary path: low-TTFA streaming PCM16 via Web Audio API
    try {
      const handled = await tryStreamSpeak(payload, myGen, ctrl);
      if (handled || myGen !== local.speakGen) return;
    } catch (err) {
      if (local.ttsAbort === ctrl) local.ttsAbort = null;
      if (err?.name === "AbortError" || myGen !== local.speakGen) return;
      /* fall through to buffered /api/tts/speak */
    }

    // 2. Fallback path: buffered WAV via /api/tts/speak
    if (myGen !== local.speakGen) return;
    const wavCtrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    local.ttsAbort = wavCtrl;
    try {
      const res = await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: wavCtrl?.signal,
      });
      if (myGen !== local.speakGen) return;
      if (local.ttsAbort === wavCtrl) local.ttsAbort = null;
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
        setSpeakingUI("speaking", local.speakingPreviewEngine);
        await audio.play();
        return;
      }
    } catch (err) {
      if (local.ttsAbort === wavCtrl) local.ttsAbort = null;
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
        setSpeakingUI("speaking", local.speakingPreviewEngine);
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
    if (btn) {
      btn.classList.toggle("is-recording", phase === "recording");
      btn.classList.toggle("is-transcribing", phase === "transcribing" || phase === "requesting_mic");
    }
    if (lbl) {
      if (phase === "requesting_mic") lbl.textContent = "Requesting mic…";
      else if (phase === "recording") lbl.textContent = "Done speaking";
      else if (phase === "transcribing") lbl.textContent = "Transcribing…";
      else lbl.textContent = isRetry ? "Speak retry" : "Speak answer";
    }
    const lockedByPhase = Boolean(
      !isRetry &&
        local.session &&
        (local.session.phase === "awaiting_debrief" || isDebriefInFlight())
    );
    if (btn) {
      // An idle mic also waits while an AI reply is in flight, so it never competes with "Thinking…".
      btn.disabled =
        phase === "transcribing" ||
        phase === "requesting_mic" ||
        lockedByPhase ||
        (phase === "idle" && Boolean(local.busy));
    }
    // While the dock is locked its status line belongs to syncComposerLock().
    if (status && note && !lockedByPhase) {
      status.textContent = note;
      delete status.dataset.lockNote;
    }
    if (isRetry) syncRetrySubmit();
  }

  /* True from the moment End & Debrief is clicked (the request is in flight) until the
     session leaves the debriefing phase — including a reload that resumes mid-debrief. */
  function isDebriefInFlight() {
    return local.session?.phase === "debriefing" || opMachine.state === "debriefing";
  }

  /* The answer dock steps back whenever something else has the floor: an in-flight
     AI reply (waiting) or the end of the interview (locked). One function owns that
     state so the mic, the input and the status line can never disagree. */
  function syncComposerLock() {
    const phase = local.session?.phase || "active";
    const isDebriefing = isDebriefInFlight();
    const locked = isDebriefing || phase === "awaiting_debrief";
    const waiting = Boolean(local.busy) && !locked;
    const micIdle = micMachine.state === "idle";

    const composer = el("practiceComposerWrap");
    if (composer) {
      composer.classList.toggle("is-locked", locked);
      // While the debrief is written the whole composer steps aside: the staged progress
      // card in the feed is the single status. Phase-driven, so an error or a new session
      // restores the dock by itself.
      composer.classList.toggle("is-debriefing", isDebriefing);
    }
    const input = el("practiceAnswerInput");
    if (input) input.disabled = locked;
    // Never touch a mic mid-capture: its button is the only way to stop the recording.
    const micBtn = el("practiceMicBtn");
    if (micBtn && micIdle) micBtn.disabled = locked || waiting;
    const retryMicBtn = el("practiceRetryMicBtn");
    if (retryMicBtn && micIdle) retryMicBtn.disabled = Boolean(local.busy);

    const status = el("practiceMicStatus");
    if (status) {
      if (locked) {
        status.textContent = "Every question answered — your debrief is ready when you are";
        status.dataset.lockNote = "1";
      } else if (status.dataset.lockNote) {
        status.textContent = "Transcribed privately on this Mac";
        delete status.dataset.lockNote;
      }
    }
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
    local.recordingTarget = null;
    local.micTarget = null;
    local.lastFailedClip = null;
    micMachine.reset("idle", { event: "abort_mic" });
    cleanupMicResources();
    // Always restore both status lines on session switch/reset so a prior mic-denied note never lingers.
    updateMicUI("answer", "idle", "Transcribed privately on this Mac");
    updateMicUI("retry", "idle", "One focused 45-second rep");
  }

  function renderMicRetryButton(target, errMsg) {
    const statusId = target === "retry" ? "practiceRetryMicStatus" : "practiceMicStatus";
    const statusEl = el(statusId);
    if (!statusEl) return;
    statusEl.innerHTML = `${escapeHtml(`Local transcription error: ${errMsg}`)} <button type="button" class="btn-inline-retry" data-retry-asr="${escapeHtml(target)}">Retry transcription</button>`;
    const btn = statusEl.querySelector("[data-retry-asr]");
    if (btn) {
      btn.addEventListener("click", () => {
        retryFailedTranscription();
      });
    }
  }

  async function transcribeInt16Clip(int16, { target, myQuestionId, mySessionGen, myRecGen }) {
    micMachine.transition("transcribing", { event: "start_transcribe" });
    updateMicUI(target, "transcribing", "Transcribing on this Mac…");
    try {
      const res = await fetch("/api/asr/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: int16.buffer,
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || myRecGen !== local.recGen) return false;
      if (!res.ok) {
        throw new Error(formatApiDetail(data.detail, "Transcription failed"));
      }
      local.lastFailedClip = null;
      // If the user advanced to a different question while transcribing, discard stale answer audio
      if (target === "answer" && myQuestionId && currentQuestionId() !== myQuestionId) {
        local.micTarget = null;
        micMachine.transition("idle", { event: "transcribe_discarded_question_changed" });
        updateMicUI(target, "idle", "Transcribed privately on this Mac");
        return true;
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
      return true;
    } catch (err) {
      if (mySessionGen !== local.sessionGen || myRecGen !== local.recGen) return false;
      local.micTarget = null;
      local.lastFailedClip = { target, questionId: myQuestionId, int16 };
      micMachine.transition("idle", { event: "transcribe_error" });
      updateMicUI(target, "idle", `Local transcription error: ${err.message || err}`);
      renderMicRetryButton(target, err.message || err);
      showToast(`${err.message || err} — click Retry transcription to try again.`, 3800);
      return false;
    }
  }

  async function retryFailedTranscription() {
    const clip = local.lastFailedClip;
    if (!clip || !clip.int16 || micMachine.state !== "idle" || opMachine.state === "exiting") return;
    local.recGen += 1;
    local.micTarget = clip.target;
    await transcribeInt16Clip(clip.int16, {
      target: clip.target,
      myQuestionId: clip.questionId,
      mySessionGen: local.sessionGen,
      myRecGen: local.recGen,
    });
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
          updateMicUI(target, "idle", "Transcribed privately on this Mac");
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

      local.lastFailedClip = null;
      updateMicUI(target, "recording", "Listening · 0:00");
      if (target === "retry") startRetryTimerIfIdle();
      local.recTimerHandle = setInterval(() => {
        const elapsed = (Date.now() - local.recStartedAt) / 1000;
        const left = Math.max(0, Math.ceil(120 - elapsed));
        const label =
          elapsed >= 90
            ? `Listening · ${fmtShortClock(elapsed)} (${left}s left)`
            : `Listening · ${fmtShortClock(elapsed)}`;
        updateMicUI(target, "recording", label);
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

    await transcribeInt16Clip(int16, {
      target,
      myQuestionId,
      mySessionGen,
      myRecGen,
    });
  }

  /* ---------------- Extensible Studio Slot (M3 & M4) ---------------- */

  function registerStudioPlugin(interviewType, factory) {
    local.studioPlugins.set(interviewType, factory);
    if (local.session && local.session.config?.interview_type === interviewType) {
      syncStudioSlot();
    }
  }

  function setStudioLayout(fmt) {
    const ws = el("practiceWorkspace");
    if (!ws) return;
    const changed = (ws.getAttribute("data-studio") || null) !== (fmt || null);
    if (fmt) ws.setAttribute("data-studio", fmt);
    else ws.removeAttribute("data-studio");
    // Only on a real layout change: a re-render must never fight an in-progress drag.
    if (changed) applySplitPct(fmt || null, fmt ? readSplitPct(fmt) : null);
  }

  /* ---------------- Split studio: resizable divider + collapsible conversation ----------------
     The conversation column's width is one CSS variable (--split-pct) on the workspace;
     styles.css clamps it and supplies the per-format default when it is unset. */

  const SPLIT_PCT_MIN = 22;
  const SPLIT_PCT_MAX = 48;
  const SPLIT_PX_MIN = 260;
  const SPLIT_PX_MAX = 560;
  const SPLIT_PCT_DEFAULT = { system_design: 30, coding: 34 };
  const clampNum = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
  const splitStorageKey = (fmt) => `mockery_split_pct_${fmt}`;

  function readSplitPct(fmt) {
    try {
      const n = Number.parseFloat(window.localStorage?.getItem(splitStorageKey(fmt)) ?? "");
      return Number.isFinite(n) ? clampNum(n, SPLIT_PCT_MIN, SPLIT_PCT_MAX) : null;
    } catch {
      return null;
    }
  }

  function storeSplitPct(fmt, pct) {
    try {
      if (pct === null) window.localStorage?.removeItem(splitStorageKey(fmt));
      else window.localStorage?.setItem(splitStorageKey(fmt), pct.toFixed(1));
    } catch {
      /* storage unavailable: the width simply lasts for this session */
    }
  }

  /* pct === null means "use the stylesheet default for this format". */
  function applySplitPct(fmt, pct) {
    const ws = el("practiceWorkspace");
    if (!ws) return;
    if (pct === null) ws.style.removeProperty("--split-pct");
    else ws.style.setProperty("--split-pct", `${pct.toFixed(2)}%`);
    el("practiceSplitHandle")?.setAttribute(
      "aria-valuenow",
      String(Math.round(pct ?? SPLIT_PCT_DEFAULT[fmt] ?? SPLIT_PCT_DEFAULT.system_design))
    );
  }

  function bindSplitHandle() {
    const handle = el("practiceSplitHandle");
    const ws = el("practiceWorkspace");
    if (!handle || !ws) return;
    let drag = null; // { pointerId, fmt, grabOffset, pct }

    const mainRect = () => handle.parentElement.getBoundingClientRect();
    const convRight = () => el("practiceActiveView")?.getBoundingClientRect().right ?? mainRect().left;
    const pctForEdge = (edgeX) => {
      const rect = mainRect();
      if (rect.width <= 0) return null;
      const px = clampNum(edgeX - rect.left, SPLIT_PX_MIN, SPLIT_PX_MAX);
      return clampNum((px / rect.width) * 100, SPLIT_PCT_MIN, SPLIT_PCT_MAX);
    };

    handle.addEventListener("pointerdown", (ev) => {
      const fmt = ws.getAttribute("data-studio");
      if (!fmt || ev.button !== 0) return;
      ev.preventDefault();
      // The handle owns this pointer until release, so the whiteboard and the editor
      // underneath never see the gesture.
      drag = { pointerId: ev.pointerId, fmt, grabOffset: ev.clientX - convRight(), pct: null };
      try {
        handle.setPointerCapture(ev.pointerId);
      } catch {
        /* synthetic pointers cannot be captured */
      }
      ws.classList.add("is-split-resizing");
    });

    handle.addEventListener("pointermove", (ev) => {
      if (!drag || ev.pointerId !== drag.pointerId) return;
      const pct = pctForEdge(ev.clientX - drag.grabOffset);
      if (pct === null) return;
      drag.pct = pct;
      applySplitPct(drag.fmt, pct);
    });

    const endDrag = (ev) => {
      if (!drag || ev.pointerId !== drag.pointerId) return;
      if (drag.pct !== null) storeSplitPct(drag.fmt, drag.pct);
      drag = null;
      ws.classList.remove("is-split-resizing");
    };
    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);
    handle.addEventListener("lostpointercapture", endDrag);

    const resetSplit = () => {
      const fmt = ws.getAttribute("data-studio");
      if (!fmt) return;
      storeSplitPct(fmt, null);
      applySplitPct(fmt, null);
    };
    handle.addEventListener("dblclick", resetSplit);

    handle.addEventListener("keydown", (ev) => {
      const fmt = ws.getAttribute("data-studio");
      if (!fmt) return;
      if (ev.key === "Enter") {
        ev.preventDefault();
        resetSplit();
        return;
      }
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      ev.preventDefault();
      const rect = mainRect();
      if (rect.width <= 0) return;
      const stepPx = (rect.width * 2) / 100;
      const pct = pctForEdge(convRight() + (ev.key === "ArrowLeft" ? -stepPx : stepPx));
      if (pct === null) return;
      applySplitPct(fmt, pct);
      storeSplitPct(fmt, pct);
    });
  }

  function renderConversationToggle() {
    const sess = local.session;
    const collapsed = local.convCollapsed;
    el("practiceWorkspace")?.toggleAttribute("data-conv-collapsed", collapsed);
    const turnCount = sess?.turns?.length || 0;
    if (!collapsed) local.convSeenTurns = turnCount;
    const btn = el("practiceConvToggleBtn");
    if (!btn) return;
    const label = collapsed ? "Show conversation" : "Hide conversation — focus on your workspace";
    btn.setAttribute("aria-pressed", collapsed ? "true" : "false");
    btn.setAttribute("aria-label", label);
    btn.title = label;
    // A quiet dot says the interviewer or coach replied while the panel was tucked away.
    btn.classList.toggle("has-unread", collapsed && turnCount > local.convSeenTurns);
  }

  function setConversationCollapsed(collapsed) {
    const next = Boolean(collapsed);
    if (local.convCollapsed === next) return;
    local.convCollapsed = next;
    renderConversationToggle();
    if (!next) {
      const feed = el("practiceFeed");
      if (feed) feed.scrollTop = feed.scrollHeight;
    }
  }

  /* A new session, or the interview wrapping up, always brings the conversation back:
     the opening question and the debrief progress are both read there. */
  function syncConversationPanel(sess) {
    const sessionChanged = sess.id !== local.convSessionId;
    const wrappingUp = sess.phase !== local.convPhase && sess.phase !== "active";
    local.convSessionId = sess.id;
    local.convPhase = sess.phase;
    if (sessionChanged) local.convSeenTurns = 0;
    if (local.convCollapsed && (sessionChanged || wrappingUp)) local.convCollapsed = false;
    renderConversationToggle();
  }

  function unmountStudioPlugin({ flush = true } = {}) {
    setStudioLayout(null);
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
      setStudioLayout(fmt);
      renderWorkspaceSaveStatus();
      return;
    }

    unmountStudioPlugin();
    slot.classList.remove("hidden");
    setStudioLayout(fmt);
    local.mountedPluginKey = pluginKey;
    local.mountedPlugin = factory({
      containerEl: slot,
      session: local.session,
      onWorkspaceChange: () => {
        scheduleWorkspaceSync();
        renderRoadmapCard();
      },
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
    syncRetrySubmit();
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
    local.debriefStartedAt = 0;
    cancelActiveOp();
    clearConfirmArms();
    clearExitForceArm();
    clearPendingOpUI({ force: true });
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

  const DEFAULT_PRACTICE_START_HINT = "Answer out loud or by typing. A supportive debrief follows.";

  function setButtonLoading(btnId, loading, loadingLabel = null, idleLabel = null) {
    const btn = el(btnId);
    if (!btn) return;
    btn.classList.toggle("is-loading", Boolean(loading));
    btn.setAttribute("aria-busy", loading ? "true" : "false");
    if (loading && loadingLabel !== null) {
      btn.textContent = loadingLabel;
    } else if (!loading && idleLabel !== null) {
      btn.textContent = idleLabel;
    }
  }

  function setCoachRailThinking(active, toneText = "Analyzing", messageText = "", { quiet = false } = {}) {
    const toneEl = el("practiceCoachTone");
    const msgEl = el("practiceCoachMessage");
    const card = toneEl?.closest(".practice-coach-card") || msgEl?.closest(".practice-coach-card");
    if (card) {
      card.classList.toggle("is-thinking", Boolean(active));
      // Quiet: something else on screen already carries the progress (the debrief card).
      card.classList.toggle("is-quiet", Boolean(active && quiet));
    }
    // While a new evaluation is in flight, the score on screen is the previous reading.
    const meterCard = el("practiceMeterCard");
    if (meterCard) {
      meterCard.classList.toggle("is-updating", Boolean(active));
      if (active) meterCard.setAttribute("aria-busy", "true");
      else meterCard.removeAttribute("aria-busy");
    }
    if (active) {
      if (toneEl) toneEl.textContent = toneText;
      if (msgEl && messageText) msgEl.textContent = messageText;
    }
  }

  function clearPendingOpUI({ force = false } = {}) {
    if (local.pendingOpTimer) {
      clearInterval(local.pendingOpTimer);
      local.pendingOpTimer = null;
    }
    if (local.pendingOpNodes.length > 0) {
      for (const node of local.pendingOpNodes.splice(0, local.pendingOpNodes.length)) {
        try {
          node.remove();
        } catch {
          /* ignore */
        }
      }
    }
    setButtonLoading("practiceSendBtn", false, null, "Send");
    setButtonLoading("practiceHintBtn", false, null, "Hint");
    setButtonLoading("practiceNextQBtn", false, null, "Next Q");
    const keepDebriefing = !force && local.session?.phase === "debriefing";
    if (!keepDebriefing) {
      setButtonLoading("practiceDebriefBtn", false, null, "End & Debrief");
      setCoachRailThinking(false);
    }
    setButtonLoading("practiceRetrySubmitBtn", false, null, "Evaluate retry");
  }

  function startPendingTimer(onTick, startedAtMs = Date.now()) {
    if (local.pendingOpTimer) {
      clearInterval(local.pendingOpTimer);
      local.pendingOpTimer = null;
    }
    const startedAt = Number.isFinite(startedAtMs) && startedAtMs > 0 ? startedAtMs : Date.now();
    onTick(Math.max(0, (Date.now() - startedAt) / 1000));
    local.pendingOpTimer = window.setInterval(() => {
      onTick(Math.max(0, (Date.now() - startedAt) / 1000));
    }, 250);
  }

  function beginOpAbort() {
    if (local.opAbortController) {
      try {
        local.opAbortController.abort();
      } catch {
        /* ignore */
      }
    }
    const ctrl = new AbortController();
    local.opAbortController = ctrl;
    return ctrl;
  }

  function cancelActiveOp() {
    if (!local.opAbortController) return false;
    const ctrl = local.opAbortController;
    local.opAbortController = null;
    try {
      ctrl.abort();
    } catch {
      /* ignore */
    }
    return true;
  }

  function clearConfirmArms() {
    if (local.debriefConfirmTimer) {
      clearTimeout(local.debriefConfirmTimer);
      local.debriefConfirmTimer = null;
    }
    if (local.debriefConfirmArmed) {
      local.debriefConfirmArmed = false;
      const btn = el("practiceDebriefBtn");
      if (btn && !btn.classList.contains("is-loading")) btn.textContent = "End & Debrief";
    }
    if (local.nextQConfirmTimer) {
      clearTimeout(local.nextQConfirmTimer);
      local.nextQConfirmTimer = null;
    }
    if (local.nextQConfirmArmed) {
      local.nextQConfirmArmed = false;
      const btn = el("practiceNextQBtn");
      if (btn && !btn.classList.contains("is-loading")) btn.textContent = "Next Q";
    }
  }

  function showPendingTurnCard({
    container,
    roleClass = "pturn-interviewer",
    title = "Interviewer",
    stages = [],
    coachRailMsg = "",
    onCancel = null,
  }) {
    if (!container) return null;
    const card = document.createElement("div");
    card.className = `pturn ${roleClass} is-thinking`;
    card.setAttribute("role", "status");
    card.setAttribute("aria-live", "polite");
    const initialText = stages[0]?.text || "Thinking…";
    const cancelHtml = onCancel
      ? `<button type="button" class="btn-inline-cancel" data-cancel-op>Cancel</button>`
      : "";
    card.innerHTML = `
      <div class="pturn-head">
        <span>${escapeHtml(title)}</span>
        <span class="pturn-head-actions">
          <span class="thinking-elapsed" aria-hidden="true"></span>
          ${cancelHtml}
        </span>
      </div>
      <div class="pturn-body">
        <span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
        <span class="thinking-stage-text">${escapeHtml(initialText)}</span>
      </div>
    `;
    if (onCancel) {
      card.querySelector("[data-cancel-op]")?.addEventListener("click", () => onCancel());
    }
    container.appendChild(card);
    local.pendingOpNodes.push(card);
    container.scrollTop = container.scrollHeight;

    if (coachRailMsg) {
      setCoachRailThinking(true, "Analyzing", coachRailMsg);
    }

    const stageEl = card.querySelector(".thinking-stage-text");
    const elapsedEl = card.querySelector(".thinking-elapsed");
    startPendingTimer((sec) => {
      if (!card.isConnected) {
        if (local.pendingOpTimer) {
          clearInterval(local.pendingOpTimer);
          local.pendingOpTimer = null;
        }
        return;
      }
      if (elapsedEl) {
        if (sec >= 1.0) {
          elapsedEl.textContent = `${sec.toFixed(1)}s`;
          elapsedEl.classList.add("is-visible");
        }
      }
      if (stageEl && stages.length > 0) {
        let activeText = stages[0].text;
        for (const st of stages) {
          if (sec >= st.at) activeText = st.text;
        }
        if (stageEl.textContent !== activeText) {
          stageEl.textContent = activeText;
        }
      }
    });
    return card;
  }

  function showPendingDebriefCard(container, startedAtMs = Date.now()) {
    if (!container) return null;
    for (const stale of container.querySelectorAll(".debrief-pending-card")) {
      stale.remove();
    }
    const card = document.createElement("div");
    card.className = "debrief-pending-card";
    card.setAttribute("role", "status");
    card.setAttribute("aria-live", "polite");
    card.innerHTML = `
      <div class="debrief-pending-head">
        <span class="debrief-pending-title">
          <span class="inline-spinner" aria-hidden="true"></span>
          <span>Writing your coaching debrief…</span>
        </span>
        <span class="thinking-elapsed" aria-hidden="true"></span>
      </div>
      <ul class="debrief-pending-steps">
        <li data-step="0" class="is-active"><span class="debrief-step-dot"></span><span>Reviewing your transcript &amp; workspace</span></li>
        <li data-step="1"><span class="debrief-step-dot"></span><span>Scoring your four signals</span></li>
        <li data-step="2"><span class="debrief-step-dot"></span><span>Choosing one thing to retry</span></li>
      </ul>
      <div class="debrief-skeleton-lines" aria-hidden="true">
        <div class="debrief-skeleton-line" style="width: 88%"></div>
        <div class="debrief-skeleton-line" style="width: 72%"></div>
        <div class="debrief-skeleton-line" style="width: 54%"></div>
      </div>
    `;
    container.appendChild(card);
    local.pendingOpNodes.push(card);
    container.scrollTop = container.scrollHeight;

    setCoachRailThinking(
      true,
      "Debriefing",
      "Pulling together your strengths, a question-by-question breakdown, and one retry drill…",
      { quiet: true }
    );

    const elapsedEl = card.querySelector(".thinking-elapsed");
    const stepEls = Array.from(card.querySelectorAll(".debrief-pending-steps li"));
    startPendingTimer((sec) => {
      if (!card.isConnected) {
        if (local.pendingOpTimer) {
          clearInterval(local.pendingOpTimer);
          local.pendingOpTimer = null;
        }
        return;
      }
      if (elapsedEl && sec >= 1.0) {
        elapsedEl.textContent = `${sec.toFixed(1)}s`;
        elapsedEl.classList.add("is-visible");
      }
      const curStep = sec >= 5.5 ? 2 : sec >= 2.2 ? 1 : 0;
      stepEls.forEach((li, idx) => {
        li.classList.toggle("is-done", idx < curStep);
        li.classList.toggle("is-active", idx === curStep);
      });
    }, startedAtMs);
    return card;
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
    syncRetrySubmit();
    syncComposerLock();
  }

  /* Evaluate retry is only live when there is something to evaluate: typed or
     transcribed words, or a retry recording it can stop and submit. */
  function syncRetrySubmit() {
    const btn = el("practiceRetrySubmitBtn");
    if (!btn) return;
    const hasText = Boolean((el("practiceRetryInput")?.value || "").trim());
    const recordingRetry = micMachine.state === "recording" && local.recordingTarget === "retry";
    btn.disabled = Boolean(local.busy) || !(hasText || recordingRetry);
  }

  async function startNewPractice() {
    if (state.captureState && state.captureState !== "idle") {
      showToast("Stop live Copilot capture before starting an AI mock interview.", 3200);
      return;
    }
    if (opMachine.state !== "idle") return;
    if (!opMachine.transition("starting", { event: "start_practice" })) return;

    local.sessionGen += 1;
    local.debriefStartedAt = 0;
    const mySessionGen = local.sessionGen;
    stopDebriefPoll();
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    clearPendingOpUI({ force: true });
    resetWorkspaceSaveState();
    clearDraftInputs();

    const btn = el("practiceStartBtn");
    const lbl = el("practiceStartBtnLabel");
    const hintEl = el("practiceStartHint");
    if (btn) {
      btn.disabled = true;
      btn.classList.add("is-loading");
      btn.setAttribute("aria-busy", "true");
    }
    if (lbl) lbl.textContent = "Preparing your interview…";
    if (hintEl) hintEl.classList.add("is-loading");

    // The Home headline shows the JD role as "Title (Company)". The server already pairs the
    // session with the active JD's company, so send the bare title — otherwise the studio bar
    // and the interviewer's greeting read "Title (Company) · Company".
    const typedRole = (el("setupRoleInput")?.value || "").trim();
    const activeJd = state.profileSummary?.active_jd;
    const roleTitle =
      activeJd?.title && activeJd.company && typedRole === jdRoleLabel(activeJd) ? activeJd.title : typedRole;
    const fmt = getSelectedFormat();
    const selectedQ = el("practiceQuestionPicker")?.value || null;
    const liveNudges = Boolean(el("practiceNudgesCheckbox")?.checked ?? true);
    const voiceOn = Boolean(el("practiceVoiceCheckbox")?.checked ?? true);
    local.voiceEnabled = voiceOn;

    const isTailored = local.questionSource !== "bank_only";
    startPendingTimer((sec) => {
      if (!hintEl) return;
      const suffix = sec >= 1.5 ? ` (${sec.toFixed(1)}s)` : "";
      if (sec >= 6.0) {
        hintEl.textContent = `Still working — waiting on model response to open your session…${suffix}`;
      } else if (sec >= 2.2 && isTailored) {
        hintEl.textContent = `Tailoring interview questions to your redacted resume & JD…${suffix}`;
      } else {
        hintEl.textContent = `Selecting prompts and warming up your AI interviewer…${suffix}`;
      }
    });

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
      clearPendingOpUI();
      if (btn) {
        btn.disabled = false;
        btn.classList.remove("is-loading");
        btn.setAttribute("aria-busy", "false");
      }
      if (lbl) lbl.textContent = "Start AI mock interview";
      if (hintEl) {
        hintEl.classList.remove("is-loading");
        hintEl.textContent = DEFAULT_PRACTICE_START_HINT;
      }
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
    local.debriefStartedAt = 0;
    const mySessionGen = local.sessionGen;
    stopDebriefPoll();
    stopSpeaking();
    abortMicRecording();
    stopRetryTimer();
    clearPendingOpUI({ force: true });
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
    clearConfirmArms();
    if (!opMachine.transition("submitting_turn", { event: "submit_turn" })) return;
    setBusy(true);

    let clearedDraft = false;
    let submittedText = "";
    let opCtrl = null;
    try {
      if (micMachine.state === "recording" && local.recordingTarget === "answer") {
        await stopMicAndTranscribe();
        if (mySessionGen !== local.sessionGen || !local.session || opMachine.state === "exiting") return;
      }
      stopSpeaking();

      const inputEl = el("practiceAnswerInput");
      const text = (explicitText !== null ? String(explicitText) : inputEl?.value || "").trim();
      if (!text) return;
      submittedText = text;

      if (explicitText === null && inputEl) {
        inputEl.value = "";
        clearedDraft = true;
      }

      setButtonLoading("practiceSendBtn", true, "Thinking…");
      opCtrl = beginOpAbort();

      // Optimistic render of candidate's message + staged AI Interviewer thinking card
      const feed = el("practiceFeed");
      if (feed) {
        const pendingEl = document.createElement("div");
        pendingEl.className = "pturn pturn-candidate";
        pendingEl.innerHTML = `<div class="pturn-head">You</div><div class="pturn-body">${formatWithNewlines(text)}</div>`;
        feed.appendChild(pendingEl);
        showPendingTurnCard({
          container: feed,
          roleClass: "pturn-interviewer",
          title: "Interviewer",
          stages: [
            { at: 0, text: "Reviewing your answer…" },
            { at: 2.2, text: "Weighing your answer and shaping a follow-up…" },
            { at: 6.5, text: "Still thinking — waiting on model response…" },
          ],
          coachRailMsg: "Reading your answer for its strongest signals…",
          onCancel: () => cancelActiveOp(),
        });
      }

      const wsCtx = getActiveWorkspaceContext();
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: opCtrl.signal,
        body: JSON.stringify({
          text,
          workspace_context: wsCtx,
          expected_question_id: currentQuestionId(),
        }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Turn failed"));
      clearedDraft = false;
      local.session = data.session;
      renderPracticeSession();
      if (data.interviewer_turn?.text && local.voiceEnabled) {
        speakText(data.interviewer_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        const inputEl = el("practiceAnswerInput");
        if (clearedDraft && submittedText && inputEl && !inputEl.value.trim()) {
          inputEl.value = submittedText;
        }
        const msg =
          err?.name === "AbortError"
            ? "Cancelled — your answer is still in the box."
            : `${err.message || err} — your answer was kept in the box.`;
        showToast(msg, 3600);
        renderPracticeSession();
      }
    } finally {
      if (opCtrl && local.opAbortController === opCtrl) {
        local.opAbortController = null;
      }
      clearPendingOpUI();
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
    clearConfirmArms();
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("requesting_hint", { event: "request_hint" })) return;
    stopSpeaking();
    setBusy(true);
    setButtonLoading("practiceHintBtn", true, "Hint…");
    const opCtrl = beginOpAbort();
    showPendingTurnCard({
      container: el("practiceFeed"),
      roleClass: "pturn-coach",
      title: "Supportive Coach",
      stages: [
        { at: 0, text: "Crafting a gentle hint without giving it away…" },
        { at: 2.5, text: "Checking your progress & workspace context…" },
        { at: 6.5, text: "Still thinking — waiting on model response…" },
      ],
      coachRailMsg: "Finding the most helpful nudge for where you are…",
      onCancel: () => cancelActiveOp(),
    });
    try {
      const wsCtx = getActiveWorkspaceContext();
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/hint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: opCtrl.signal,
        body: JSON.stringify({ workspace_context: wsCtx }),
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not get hint"));
      local.session = data.session;
      renderPracticeSession();
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(err?.name === "AbortError" ? "Hint request cancelled." : String(err.message || err), 3000);
        renderPracticeSession();
      }
    } finally {
      if (local.opAbortController === opCtrl) local.opAbortController = null;
      clearPendingOpUI();
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
    const hasPendingDraft =
      micMachine.state === "recording" || Boolean((el("practiceAnswerInput")?.value || "").trim());
    if (hasPendingDraft && !local.nextQConfirmArmed) {
      clearConfirmArms();
      local.nextQConfirmArmed = true;
      const btn = el("practiceNextQBtn");
      if (btn) btn.textContent = "Skip draft?";
      local.nextQConfirmTimer = setTimeout(() => clearConfirmArms(), 5000);
      showToast("You have an unsent answer — send it first, or click Next Q again to skip it.", 3800);
      return;
    }
    clearConfirmArms();
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("advancing", { event: "advance_question" })) return;
    stopSpeaking();
    abortMicRecording();
    setBusy(true);
    setButtonLoading("practiceNextQBtn", true, "Loading…");
    const opCtrl = beginOpAbort();
    showPendingTurnCard({
      container: el("practiceFeed"),
      roleClass: "pturn-interviewer",
      title: "Interviewer",
      stages: [
        { at: 0, text: "Preparing the next question…" },
        { at: 2.5, text: "Tailoring the transition from your last answer…" },
        { at: 6.5, text: "Still thinking — waiting on model response…" },
      ],
      onCancel: () => cancelActiveOp(),
    });
    try {
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/next`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: opCtrl.signal,
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
        showToast("All questions complete — click End & Debrief when you are ready.", 3400);
      } else if (data.interviewer_turn?.text && local.voiceEnabled) {
        speakText(data.interviewer_turn.text);
      }
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(err?.name === "AbortError" ? "Cancelled." : String(err.message || err), 3000);
        renderPracticeSession();
      }
    } finally {
      if (local.opAbortController === opCtrl) local.opAbortController = null;
      clearPendingOpUI();
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
    const qTotal = Array.isArray(local.session.questions) ? local.session.questions.length : 0;
    const qIdx = Number(local.session.current_question_index || 0);
    const hasRemainingQuestions = local.session.phase === "active" && qTotal > 0 && qIdx < qTotal - 1;
    if (hasRemainingQuestions && !local.debriefConfirmArmed) {
      clearConfirmArms();
      local.debriefConfirmArmed = true;
      const btn = el("practiceDebriefBtn");
      if (btn) btn.textContent = "Confirm End?";
      local.debriefConfirmTimer = setTimeout(() => clearConfirmArms(), 5000);
      const leftCount = qTotal - 1 - qIdx;
      showToast(
        `${leftCount} question${leftCount === 1 ? "" : "s"} remaining — click End & Debrief again to finish and grade now.`,
        4000
      );
      return;
    }
    clearConfirmArms();
    const mySessionGen = local.sessionGen;
    if (!opMachine.transition("debriefing", { event: "run_debrief" })) return;
    setBusy(true);
    stopSpeaking();
    const opCtrl = beginOpAbort();
    // End & Debrief simply goes quiet (disabled by setBusy): the progress card is the status.
    local.debriefStartedAt = Date.now();
    setConversationCollapsed(false);
    showPendingDebriefCard(el("practiceFeed"), local.debriefStartedAt);
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
      // If the candidate typed or just spoke a final answer, include it in the transcript before debriefing!
      const inputEl = el("practiceAnswerInput");
      const unsentDraft = (inputEl?.value || "").trim();
      if (unsentDraft && local.session.phase === "active") {
        const turnRes = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/turn`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: opCtrl.signal,
          body: JSON.stringify({
            text: unsentDraft,
            workspace_context: getActiveWorkspaceContext(),
            expected_question_id: currentQuestionId(),
          }),
        });
        const turnData = await turnRes.json();
        if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
        if (!turnRes.ok) {
          throw new Error(formatApiDetail(turnData.detail, "Could not save your final answer before debrief"));
        }
        local.session = turnData.session;
        if (inputEl) inputEl.value = "";
      }
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/debrief`, {
        method: "POST",
        signal: opCtrl.signal,
      });
      const data = await res.json();
      if (mySessionGen !== local.sessionGen || opMachine.state === "exiting") return;
      if (!res.ok) throw new Error(formatApiDetail(data.detail, "Could not generate debrief"));
      local.session = data.session;
      renderPracticeSession();
      refreshPracticeHistory();
    } catch (err) {
      if (mySessionGen === local.sessionGen && opMachine.state !== "exiting") {
        showToast(err?.name === "AbortError" ? "Debrief cancelled." : String(err.message || err), 3400);
        renderPracticeSession();
      }
    } finally {
      if (local.opAbortController === opCtrl) local.opAbortController = null;
      if (local.session?.phase !== "debriefing") {
        clearPendingOpUI();
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

  const RETRY_SECONDS = 45;

  function paintRetryTimer(remSec, { running = false } = {}) {
    const clock = el("practiceRetryClock");
    const ring = el("practiceRetryRing");
    const card = el("practiceRetryCard");
    const warn = running && remSec <= 10;
    if (clock) {
      clock.textContent = fmtShortClock(remSec);
      clock.classList.toggle("is-warn", warn);
    }
    if (ring) {
      const exact = running && local.retryDeadlineMs
        ? Math.max(0, (local.retryDeadlineMs - Date.now()) / 1000)
        : remSec;
      ring.style.setProperty("--p", String(Math.max(0, Math.min(100, (exact / RETRY_SECONDS) * 100))));
      ring.classList.toggle("is-warn", warn);
    }
    if (card) {
      card.classList.toggle("is-timing", running);
      card.classList.toggle("is-warn", warn);
    }
  }

  function resetRetryTimer() {
    stopRetryTimer({ nextState: "idle" });
    local.retrySecondsLeft = RETRY_SECONDS;
    paintRetryTimer(RETRY_SECONDS);
    const btn = el("practiceRetryTimerBtn");
    if (btn) btn.textContent = "Start timer";
  }

  function startRetryTimerIfIdle() {
    if (retryTimerMachine.state === "running" || retryTimerMachine.state === "warning") return;
    if (local.session?.phase !== "completed") return;
    toggleRetryTimer();
  }

  function toggleRetryTimer() {
    const btn = el("practiceRetryTimerBtn");
    if (retryTimerMachine.state === "running" || retryTimerMachine.state === "warning") {
      resetRetryTimer();
      return;
    }
    if (!retryTimerMachine.transition("running", { event: "start_retry_timer" })) {
      retryTimerMachine.reset("running", { event: "start_retry_timer" });
    }
    local.retrySecondsLeft = RETRY_SECONDS;
    local.retryDeadlineMs = Date.now() + RETRY_SECONDS * 1000;
    paintRetryTimer(RETRY_SECONDS, { running: true });
    if (btn) btn.textContent = "Reset";
    local.retryTimerHandle = setInterval(() => {
      const remSec = Math.max(0, Math.ceil((local.retryDeadlineMs - Date.now()) / 1000));
      local.retrySecondsLeft = remSec;
      paintRetryTimer(remSec, { running: true });
      if (remSec <= 10 && remSec > 0 && retryTimerMachine.state === "running") {
        retryTimerMachine.transition("warning", { event: "retry_timer_warning" });
      }
      if (remSec <= 0) {
        stopRetryTimer({ nextState: "expired" });
        paintRetryTimer(0);
        if (btn) btn.textContent = "Restart";
        if (micMachine.state === "recording" && local.recordingTarget === "retry") {
          stopMicAndTranscribe();
        }
        showToast("Time. Submit your retry to see what changed.", 3200);
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
    try {
      if (micMachine.state === "recording" && local.recordingTarget === "retry") {
        await stopMicAndTranscribe();
        if (mySessionGen !== local.sessionGen || !local.session) return;
      }
      stopRetryTimer({ nextState: "idle" });
      const input = el("practiceRetryInput");
      const text = (input?.value || "").trim();
      if (!text) {
        showToast("Speak or type your retry first.", 2600);
        return;
      }
      setButtonLoading("practiceRetrySubmitBtn", true, "Evaluating retry…");
      const opCtrl = beginOpAbort();
      showPendingTurnCard({
        container: el("practiceRetryHistory"),
        roleClass: "pturn-coach",
        title: "Coach",
        stages: [
          { at: 0, text: "Comparing your retry against your original answer…" },
          { at: 2.5, text: "Writing before → after coaching feedback…" },
          { at: 6.5, text: "Still working — waiting on model response…" },
        ],
        onCancel: () => cancelActiveOp(),
      });
      const drill = local.session.debrief?.retry_drill || {};
      const res = await fetch(`/api/practice/${encodeURIComponent(local.session.id)}/retry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: opCtrl.signal,
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
      showToast("Nice rep. Your before → after is below.", 2800);
    } catch (err) {
      if (mySessionGen === local.sessionGen) {
        showToast(err?.name === "AbortError" ? "Retry evaluation cancelled." : String(err.message || err), 3200);
      }
    } finally {
      local.opAbortController = null;
      clearPendingOpUI();
      if (opMachine.state === "submitting_retry") {
        opMachine.transition("idle", { event: "submit_retry_done" });
      }
      if (mySessionGen === local.sessionGen) setBusy(false);
    }
  }

  /* ---------------- Rendering ---------------- */

  function sessionFormat(sess = local.session) {
    return sess?.config?.interview_type || "behavioral";
  }

  function carlStateLabel(val, final = false) {
    const table = final ? CARL_STATE_LABELS_FINAL : CARL_STATE_LABELS;
    return table[val] || table.missing;
  }

  function renderCarlMeter(carl, { fmt = sessionFormat(), final = false, kicker = null } = {}) {
    const rubric = rubricFor(fmt);
    const signals = carl || { context: "missing", action: "missing", result: "missing", learning: "missing" };
    for (const dim of CARL_DIMS) {
      const val = signals[dim] || "missing";
      const pill = el("practiceCarlPills")?.querySelector(`[data-carl-dim="${dim}"]`);
      if (!pill) continue;
      const [letter, name] = rubric.dims[dim];
      pill.classList.remove("is-strong", "is-partial", "is-missing");
      pill.classList.add(`is-${val}`);
      pill.title = `${name}: ${carlStateLabel(val, final)}`;
      const letterEl = pill.querySelector(".carl-letter");
      if (letterEl && letterEl.textContent !== letter) letterEl.textContent = letter;
      const nameEl = pill.querySelector(".carl-name");
      if (nameEl && nameEl.textContent !== name) nameEl.textContent = name;
      const stEl = pill.querySelector(".carl-state");
      if (stEl) stEl.textContent = carlStateLabel(val, final);
    }
    const kickerEl = el("practiceMeterKicker");
    if (kickerEl) kickerEl.textContent = kicker || rubric.kicker;
    const helpEl = el("practiceMeterHelp");
    if (helpEl) helpEl.title = rubric.help;
    const scoreEl = el("practiceCarlScore");
    if (scoreEl) scoreEl.textContent = `${fmtScore(carlScore(signals))} / 4`;
  }

  function renderCarlBadgesInline(carl, { fmt = sessionFormat(), final = false } = {}) {
    const rubric = rubricFor(fmt);
    return `<span class="carl-inline-strip">${CARL_DIMS.map((key) => {
      const val = (carl && carl[key]) || "missing";
      const [letter, name] = rubric.dims[key];
      return `<span class="carl-chip is-${escapeHtml(val)}" title="${escapeHtml(
        `${name}: ${carlStateLabel(val, final)}`
      )}">${letter}</span>`;
    }).join("")}</span>`;
  }

  /* Post-session scoring: the debrief reflects the latest retry, not a stale snapshot. */

  function latestRetryEntry(sess) {
    const hist = Array.isArray(sess?.retry_history) ? sess.retry_history : [];
    for (let i = hist.length - 1; i >= 0; i--) {
      const after = hist[i]?.feedback?.carl_after;
      if (after && typeof after === "object") {
        const questionId = hist[i]?.question_id || null;
        return { after, questionId };
      }
    }
    return null;
  }

  function latestRetrySignals(sess) {
    return latestRetryEntry(sess)?.after || null;
  }

  function drillQuestionIndex(sess, preferredQuestionId = null) {
    const qs = Array.isArray(sess?.questions) ? sess.questions : [];
    if (qs.length === 0) return -1;
    const qid = preferredQuestionId || sess?.debrief?.retry_drill?.question_id;
    const byId = qid ? qs.findIndex((q) => q.id === qid) : -1;
    if (byId >= 0) return byId;
    let weakest = 0;
    for (let i = 1; i < qs.length; i++) {
      if (carlScore(originalQuestionSignals(sess, i)) < carlScore(originalQuestionSignals(sess, weakest))) weakest = i;
    }
    return weakest;
  }

  function originalQuestionSignals(sess, idx) {
    const q = sess?.questions?.[idx] || null;
    const reviews = Array.isArray(sess?.debrief?.question_reviews) ? sess.debrief.question_reviews : [];
    const byId = q?.id ? reviews.find((r) => r && r.question_id === q.id) : null;
    const review = byId || (!q?.id ? reviews[idx] : null);
    if (review?.carl && typeof review.carl === "object") return review.carl;
    return q?.carl_signals || null;
  }

  function effectiveQuestionSignals(sess) {
    const qs = Array.isArray(sess?.questions) ? sess.questions : [];
    const entry = latestRetryEntry(sess);
    const drillIdx = entry ? drillQuestionIndex(sess, entry.questionId) : -1;
    return qs.map((q, idx) => (idx === drillIdx ? entry.after : originalQuestionSignals(sess, idx)));
  }

  function aggregateSignals(list) {
    const out = {};
    const rows = list.filter(Boolean);
    for (const dim of CARL_DIMS) {
      if (rows.length === 0) {
        out[dim] = "missing";
        continue;
      }
      let total = 0;
      for (const row of rows) {
        total += row[dim] === "strong" ? 1 : row[dim] === "partial" ? 0.5 : 0;
      }
      const avg = total / rows.length;
      out[dim] = avg >= 0.75 ? "strong" : avg >= 0.25 ? "partial" : "missing";
    }
    return out;
  }

  function renderRetryOutcome(before, after, fmt) {
    if (!after) return "";
    const rubric = rubricFor(fmt);
    const moves = [];
    for (const dim of CARL_DIMS) {
      const from = (before && before[dim]) || "missing";
      const to = after[dim] || "missing";
      if (from === to) continue;
      const rank = { missing: 0, partial: 1, strong: 2 };
      const dir = (rank[to] || 0) > (rank[from] || 0) ? "is-up" : "is-down";
      moves.push(
        `<span class="retry-delta ${dir}"><span class="retry-delta-dim">${escapeHtml(rubric.dims[dim][1])}</span>` +
          `<span class="retry-delta-from">${escapeHtml(carlStateLabel(from, true))}</span>` +
          `<span class="retry-delta-arrow" aria-hidden="true">→</span>` +
          `<span class="retry-delta-to">${escapeHtml(carlStateLabel(to, true))}</span></span>`
      );
    }
    const scoreLine = `${fmtScore(carlScore(before))} → ${fmtScore(carlScore(after))} / 4`;
    if (moves.length === 0) {
      return `<div class="retry-outcome"><span class="retry-delta is-flat">Held steady at ${escapeHtml(
        `${fmtScore(carlScore(after))} / 4`
      )}</span></div>`;
    }
    return `<div class="retry-outcome">${moves.join("")}<span class="retry-outcome-score">${escapeHtml(scoreLine)}</span></div>`;
  }

  function renderTurnCard(t, { showCoachNote = false } = {}) {
    if (t.role === "interviewer") {
      return `
        <article class="pturn pturn-interviewer">
          <div class="pturn-head">
            <span>Interviewer</span>
            <button type="button" class="link-quiet pturn-replay-btn" data-replay-turn="${escapeHtml(t.id)}" title="Hear this question again" aria-label="Replay this question aloud"><svg class="icon-xs" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M5 3.4v9.2a.5.5 0 00.77.42l7.1-4.6a.5.5 0 000-.84l-7.1-4.6A.5.5 0 005 3.4z" fill="currentColor"/></svg><span>Replay</span></button>
          </div>
          <div class="pturn-body pturn-serif">${formatWithNewlines(t.text)}</div>
        </article>
      `;
    }
    if (t.role === "coach") {
      return `
        <article class="pturn pturn-coach">
          <div class="pturn-head"><span>Coach hint</span></div>
          <div class="pturn-body">${formatWithNewlines(t.text)}</div>
        </article>
      `;
    }
    const retryTag = t.is_retry ? `<span class="pturn-retry-tag">Retry</span>` : "";
    const carlStrip = t.carl_signals ? renderCarlBadgesInline(t.carl_signals) : "";
    // During a live interview the right-rail Supportive Coach already carries this tip;
    // it is only repeated inline in the post-session transcript, where the rail has moved on.
    const tipHtml =
      showCoachNote && t.coach_tip
        ? `<div class="pturn-coach-note">${escapeHtml(t.coach_tip)}</div>`
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

    const fmt = sessionFormat(sess);
    const isCarl = rubricFor(fmt) === CARL_RUBRIC;
    const strengths = Array.isArray(debrief.top_strengths) ? debrief.top_strengths : [];
    const reviews = Array.isArray(debrief.question_reviews) ? debrief.question_reviews : [];
    const drill = debrief.retry_drill || {};
    const retries = Array.isArray(sess.retry_history) ? sess.retry_history : [];
    const retryEntry = latestRetryEntry(sess);
    const effective = effectiveQuestionSignals(sess);
    const drillIdx = drillQuestionIndex(sess, retryEntry?.questionId || null);
    const drillQid = (drillIdx >= 0 && sess?.questions?.[drillIdx]?.id) || retryEntry?.questionId || drill.question_id || null;
    const sessionSignals = aggregateSignals(effective);
    const retried = Boolean(retryEntry);

    contentEl.innerHTML = `
      <section class="debrief-hero">
        <div class="practice-kicker">Coaching debrief</div>
        <h2 class="debrief-encouragement">${escapeHtml(debrief.overall_encouragement || "Great practice session!")}</h2>
        ${
          debrief.carl_primer
            ? `<p class="debrief-primer">${escapeHtml(debrief.carl_primer)}</p>`
            : ""
        }
        <div class="debrief-hero-actions">
          ${
            drill.prompt
              ? retries.length > 0
                ? `<button type="button" class="btn btn-outline btn-sm" data-jump-retry>View your retry ↓</button>`
                : `<button type="button" class="btn btn-primary btn-sm" data-jump-retry>Jump to 45s retry drill ↓</button>`
              : ""
          }
          <span class="debrief-hero-score">
            <span class="debrief-hero-score-num">${escapeHtml(fmtScore(carlScore(sessionSignals)))} / 4</span>
            <span>session score${retried ? " · after your retry" : ""}</span>
          </span>
        </div>
      </section>

      ${
        strengths.length > 0
          ? `<section class="debrief-section">
              <h3 class="debrief-section-title">What you did well</h3>
              <ul class="debrief-strengths">
                ${strengths.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}
              </ul>
            </section>`
          : ""
      }

      ${
        reviews.length > 0
          ? `<section class="debrief-section">
              <h3 class="debrief-section-title">Question-by-question breakdown</h3>
              <div class="debrief-q-list">
                ${reviews
                  .map((qr, idx) => {
                    const isDrillCard =
                      retried &&
                      (qr.question_id && drillQid ? qr.question_id === drillQid : idx === drillIdx);
                    const improvedSignals = isDrillCard ? retryEntry?.after || null : null;
                    return `
                  <article class="debrief-q-card">
                    <div class="debrief-q-head">
                      <div class="debrief-q-heading">
                        <span class="debrief-q-num">Q${idx + 1}</span>
                        <span class="debrief-q-title">${escapeHtml(qr.title || "Practice Question")}</span>
                      </div>
                      <span class="debrief-q-score">
                        ${improvedSignals ? `<span class="debrief-q-retried">After retry</span>` : ""}
                        ${renderCarlBadgesInline(improvedSignals || qr.carl, { fmt, final: true })}
                      </span>
                    </div>
                    ${qr.prompt ? `<p class="debrief-q-prompt">${formatInline(qr.prompt)}</p>` : ""}
                    ${
                      qr.strength
                        ? `<div class="debrief-q-row"><span class="debrief-q-label">What worked</span><span>${formatInline(qr.strength)}</span></div>`
                        : ""
                    }
                    ${
                      qr.coaching_tip
                        ? `<div class="debrief-q-row"><span class="debrief-q-label">Next level</span><span>${formatInline(qr.coaching_tip)}</span></div>`
                        : ""
                    }
                    ${
                      qr.carl_reframe
                        ? `<div class="debrief-q-reframe">
                            <span class="practice-kicker">${isCarl ? "Suggested CARL reframe" : "Suggested reframe"}</span>
                            <p>${formatInline(qr.carl_reframe)}</p>
                          </div>`
                        : ""
                    }
                  </article>
                `;
                  })
                  .join("")}
              </div>
            </section>`
          : ""
      }
    `;

    // Retry drill
    if (retryCard && drill.prompt) {
      retryCard.classList.remove("hidden");
      if (el("practiceRetryFocus")) {
        el("practiceRetryFocus").textContent = drill.focus || "Sharpen your finish";
      }
      if (el("practiceRetryPrompt")) {
        el("practiceRetryPrompt").textContent = drill.prompt;
      }
      // Result first: once a retry is scored it sits directly under the prompt, and the
      // composer — with its timer — becomes the quieter "another rep" block beneath it.
      // Before any retry, the composer leads and the timer rides in the header.
      const hasRetries = retries.length > 0;
      retryCard.classList.toggle("has-retries", hasRetries);
      el("practiceRetryAgainHint")?.classList.toggle("hidden", !hasRetries);
      const histEl = el("practiceRetryHistory");
      const repEl = el("practiceRetryRep");
      const timerEl = el("practiceRetryTimerWrap");
      if (histEl && repEl) {
        // Only move a node that is out of place, so focus and a running timer are never disturbed.
        if (hasRetries && histEl.nextElementSibling !== repEl) retryCard.insertBefore(histEl, repEl);
        else if (!hasRetries && repEl.nextElementSibling !== histEl) retryCard.insertBefore(histEl, repEl.nextSibling);
      }
      const timerHome = el(hasRetries ? "practiceRetryRepHead" : "practiceRetryHead");
      if (timerEl && timerHome && timerEl.parentElement !== timerHome) timerHome.appendChild(timerEl);
      syncRetrySubmit();
      if (histEl) {
        const beforeByQIdx = new Map();
        histEl.innerHTML = retries
          .map((r, idx) => {
            const fb = r.feedback || {};
            const after = fb.carl_after && typeof fb.carl_after === "object" ? fb.carl_after : null;
            const targetIdx = drillQuestionIndex(sess, r.question_id || null);
            const before = beforeByQIdx.has(targetIdx)
              ? beforeByQIdx.get(targetIdx)
              : originalQuestionSignals(sess, targetIdx);
            const outcome = renderRetryOutcome(before, after, fmt);
            if (after) beforeByQIdx.set(targetIdx, after);
            return `
              <div class="retry-result-card">
                <div class="retry-result-head">
                  <span>Retry ${idx + 1}</span>
                  ${after ? renderCarlBadgesInline(after, { fmt, final: true }) : ""}
                </div>
                ${outcome}
                <blockquote class="retry-quote">${escapeHtml(r.candidate_answer || "")}</blockquote>
                ${fb.encouragement ? `<p class="retry-encourage">${escapeHtml(fb.encouragement)}</p>` : ""}
                ${fb.what_improved ? `<p class="retry-detail"><span class="debrief-q-label">What improved</span><span>${escapeHtml(fb.what_improved)}</span></p>` : ""}
                ${fb.one_more_polish ? `<p class="retry-detail"><span class="debrief-q-label">One more polish</span><span>${escapeHtml(fb.one_more_polish)}</span></p>` : ""}
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
      transEl.innerHTML = (sess.turns || []).map((t) => renderTurnCard(t, { showCoachNote: true })).join("");
    }
  }

  function renderRoadmapCard(sess = local.session) {
    const roadmap = el("practiceRoadmap");
    const card = el("practiceRoadmapCard");
    const kicker = el("practiceRoadmapKicker");
    if (!roadmap || !sess) return;

    const fmt = sessionFormat(sess);
    const questions = Array.isArray(sess.questions) ? sess.questions : [];
    const isCompleted = sess.phase === "completed" && Boolean(sess.debrief);
    const isAwaitingDebrief = sess.phase === "awaiting_debrief";
    const totalQ = Math.max(1, questions.length);
    const curIdx = Math.min(sess.current_question_index || 0, totalQ - 1);

    // A one-question roadmap says nothing. Show the live requirements checklist instead, or step aside.
    if (questions.length === 1 && !isCompleted) {
      const reqs = Array.isArray(questions[0].core_requirements) ? questions[0].core_requirements : [];
      if (reqs.length === 0) {
        card?.classList.add("hidden");
        roadmap.innerHTML = "";
        return;
      }
      const liveCtx = local.mountedPlugin ? getActiveWorkspaceContext() : null;
      const checkedRaw = liveCtx?.checked_requirements || sess.workspace_context?.checked_requirements || [];
      const checked = new Set((Array.isArray(checkedRaw) ? checkedRaw : []).map((r) => String(r).trim()));
      const covered = reqs.filter((r) => checked.has(String(r).trim())).length;
      card?.classList.remove("hidden");
      if (kicker) kicker.textContent = `Requirements · ${covered} of ${reqs.length}`;
      roadmap.innerHTML = `<ul class="roadmap-reqs">${reqs
        .map((r) => {
          const done = checked.has(String(r).trim());
          return `<li class="roadmap-req ${done ? "is-done" : ""}"><span class="roadmap-req-mark" aria-hidden="true"></span><span>${escapeHtml(r)}</span></li>`;
        })
        .join("")}</ul>`;
      return;
    }

    card?.classList.remove("hidden");
    if (kicker) kicker.textContent = isCompleted ? "Questions" : "Question Roadmap";
    const effective = isCompleted ? effectiveQuestionSignals(sess) : null;
    roadmap.innerHTML = questions
      .map((q, idx) => {
        const isCur = !isCompleted && !isAwaitingDebrief && idx === curIdx;
        const cls = isCur ? "is-active" : q.status === "completed" ? "is-done" : "";
        const badge = q.source === "tailored" ? "Tailored" : "Bank";
        const signals = effective ? effective[idx] || q.carl_signals : q.carl_signals;
        return `
          <div class="roadmap-item ${cls}">
            <div class="roadmap-head">
              <span class="roadmap-num">Q${idx + 1}</span>
              <span class="roadmap-title">${escapeHtml(q.title || q.competency || "Question")}</span>
              <span class="roadmap-src">${escapeHtml(badge)}</span>
            </div>
            <div class="roadmap-carl">${renderCarlBadgesInline(signals, { fmt, final: isCompleted })}</div>
          </div>
        `;
      })
      .join("");
  }

  function renderVoiceToggle() {
    const btn = el("practiceVoiceToggleBtn");
    const lbl = el("practiceVoiceToggleLabel");
    if (lbl) lbl.textContent = local.voiceEnabled ? "Voice on" : "Voice off";
    if (btn) {
      btn.setAttribute("aria-pressed", local.voiceEnabled ? "true" : "false");
      btn.classList.toggle("is-off", !local.voiceEnabled);
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
    const fmtKey = sessionFormat(sess);

    el("practiceWorkspace")?.setAttribute("data-phase", isCompleted ? "completed" : sess.phase || "active");
    el("practiceWorkspace")?.setAttribute("data-format", fmtKey);

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
        ? "Debrief"
        : isDebriefing
        ? "Debrief"
        : isAwaitingDebrief
        ? "Ready for debrief"
        : `Q${curIdx + 1} of ${totalQ}`;
    }
    renderVoiceToggle();
    // Voice playback has nothing to say once the interview is over.
    el("practiceVoiceToggleBtn")?.classList.toggle("hidden", isCompleted);

    el("practiceHintBtn")?.classList.toggle("hidden", isCompleted || isInputLocked);
    el("practiceNextQBtn")?.classList.toggle(
      "hidden",
      isCompleted || isInputLocked || curIdx + 1 >= totalQ
    );
    const debriefBtn = el("practiceDebriefBtn");
    if (debriefBtn) {
      debriefBtn.classList.toggle("hidden", isCompleted);
      // Quiet outline while the interview is live; it only becomes the primary action once every question is done.
      debriefBtn.classList.toggle("is-ready", isAwaitingDebrief);
      if (isDebriefing) {
        // Muted and inert, with its own label: no second spinner beside the progress card.
        debriefBtn.disabled = true;
        setButtonLoading("practiceDebriefBtn", false, null, "End & Debrief");
      } else if (opMachine.state === "idle") {
        debriefBtn.disabled = false;
        setButtonLoading("practiceDebriefBtn", false, null, "End & Debrief");
      }
    }

    const answerInput = el("practiceAnswerInput");
    const sendBtn = el("practiceSendBtn");
    if (answerInput) {
      answerInput.placeholder = isAwaitingDebrief
        ? "All questions complete — click End & Debrief above when you are ready."
        : "Speak, or type your answer here…";
    }
    if (sendBtn && opMachine.state === "idle") sendBtn.disabled = isInputLocked;
    // Input, both mics and the dock's status line follow the phase and any in-flight reply.
    syncComposerLock();

    el("practiceActiveView")?.classList.toggle("hidden", isCompleted);
    el("practiceDebriefView")?.classList.toggle("hidden", !isCompleted);

    // Sync M3 / M4 embedded studio slot & save pill
    syncStudioSlot();
    renderWorkspaceSaveStatus();
    // Before the feed paints, so a panel that re-opens can scroll to its newest turn.
    syncConversationPanel(sess);

    // Render feed
    const feed = el("practiceFeed");
    if (feed && !isCompleted) {
      feed.innerHTML = (sess.turns || []).map((t) => renderTurnCard(t)).join("");
      if (isDebriefing) {
        if (!local.debriefStartedAt) local.debriefStartedAt = Date.now();
        showPendingDebriefCard(feed, local.debriefStartedAt);
      }
      feed.scrollTop = feed.scrollHeight;
    }

    if (isCompleted) {
      renderDebriefView(sess);
    }

    // Live meter while interviewing; one session score (reflecting any retry) once it is over.
    if (isCompleted) {
      renderCarlMeter(aggregateSignals(effectiveQuestionSignals(sess)), {
        fmt: fmtKey,
        final: true,
        kicker: "Session Score",
      });
    } else {
      renderCarlMeter(curQ?.carl_signals, { fmt: fmtKey });
    }

    // Live Supportive Coach card
    const coachMsgEl = el("practiceCoachMessage");
    const coachToneEl = el("practiceCoachTone");
    if (coachMsgEl && coachToneEl) {
      if (isCompleted) {
        const retried = Boolean(latestRetrySignals(sess));
        // Only suggest another rep while there is still a signal left to win.
        const maxedOut =
          retried && carlScore(aggregateSignals(effectiveQuestionSignals(sess))) >= CARL_DIMS.length;
        coachToneEl.textContent = retried ? "Retry scored" : "Debrief ready";
        coachMsgEl.textContent = maxedOut
          ? "All four signals landed cleanly — export your report or start another practice."
          : retried
          ? "Your session score now reflects the retry. Run it again to lock the stronger version in, or export the report."
          : "Start with what went well, then run the retry drill while the feedback is fresh.";
      } else if (isDebriefing) {
        coachToneEl.textContent = "Debriefing";
        coachMsgEl.textContent =
          "Pulling together your strengths, a question-by-question breakdown, and one retry drill…";
      } else if (isAwaitingDebrief) {
        coachToneEl.textContent = "Wrap up";
        coachMsgEl.textContent =
          "Every question answered — nicely done. Click End & Debrief above for your breakdown and a quick retry drill.";
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
          fmtKey === "system_design"
            ? "Think out loud. Pin down requirements first, then sketch the architecture and name your tradeoffs."
            : fmtKey === "coding"
            ? "Think out loud. State your approach and its complexity before you write the first line."
            : "Think out loud. Set the context, show what you drove, and close with a measurable result and what you learned.";
      }
    }

    if (isDebriefing) {
      ensureDebriefPoll();
    } else {
      local.debriefStartedAt = 0;
      stopDebriefPoll();
      if (opMachine.state === "idle") {
        clearPendingOpUI();
      }
    }

    renderRoadmapCard(sess);
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
      speakingState: local.speakingState,
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
      local.levelTouched = true;
      setTargetLevel(btn.getAttribute("data-practice-level") || "L5");
    });

    el("practiceStartBtn")?.addEventListener("click", () => startNewPractice());

    bindSplitHandle();
    el("practiceConvToggleBtn")?.addEventListener("click", () => {
      setConversationCollapsed(!local.convCollapsed);
    });

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
      renderVoiceToggle();
    });

    // The Studio Bar replaces the global title bar, so it carries its own Profile & Settings entry points.
    el("practiceProfileBtn")?.addEventListener("click", () => el("profileBtn")?.click());
    el("practiceSettingsBtn")?.addEventListener("click", () => el("settingsBtn")?.click());

    el("practiceDebriefContent")?.addEventListener("click", (ev) => {
      if (!ev.target.closest("[data-jump-retry]")) return;
      const card = el("practiceRetryCard");
      if (!card) return;
      const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
      card.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
      card.classList.add("is-spotlit");
      window.setTimeout(() => card.classList.remove("is-spotlit"), 1600);
      el("practiceRetryMicBtn")?.focus({ preventScroll: true });
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
    el("practiceRetryInput")?.addEventListener("input", () => syncRetrySubmit());
    el("practiceRetryInput")?.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter" && !el("practiceRetrySubmitBtn")?.disabled) {
        ev.preventDefault();
        submitRetry();
      }
    });

    el("practiceExportMdBtn")?.addEventListener("click", () => {
      if (!local.session) return;
      window.location.href = `/api/practice/${encodeURIComponent(local.session.id)}/markdown`;
    });

    el("practiceToggleTranscriptBtn")?.addEventListener("click", () => {
      const box = el("practiceDebriefTranscript");
      const btn = el("practiceToggleTranscriptBtn");
      if (!box) return;
      const nowHidden = box.classList.toggle("hidden");
      if (btn) {
        btn.textContent = nowHidden ? "Show full transcript" : "Hide full transcript";
        btn.setAttribute("aria-expanded", nowHidden ? "false" : "true");
      }
      if (!nowHidden) box.scrollIntoView({ block: "nearest" });
    });

    const onReplayClick = (ev) => {
      const btn = ev.target.closest("[data-replay-turn]");
      if (!btn || !local.session) return;
      const tid = btn.getAttribute("data-replay-turn");
      const turn = (local.session.turns || []).find((t) => t.id === tid);
      if (turn?.text) speakText(turn.text, { force: true });
    };
    el("practiceFeed")?.addEventListener("click", onReplayClick);
    el("practiceDebriefTranscript")?.addEventListener("click", onReplayClick);

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

  /* Question Banks → "Practice →": line the launcher up on one specific bank question. */
  async function selectBankQuestion({ id, category = "", title = "" } = {}) {
    if (!id) return false;
    if (local.session || opMachine.state === "starting" || opMachine.state === "opening") {
      showToast("Finish or leave your current session to practice this question.", 3000);
      return false;
    }
    if (state.addons?.modules_map?.practice_mode === false) {
      showToast("Turn on AI Mock Practice in Settings to practice this question.", 3000);
      return false;
    }
    setHomeMode("practice");
    const fmtBtn = category
      ? el("setupTypePills")?.querySelector(`[data-setup-type="${CSS.escape(category)}"]`)
      : null;
    if (fmtBtn && !fmtBtn.classList.contains("active")) fmtBtn.click();
    await refreshQuestionPicker();
    const picker = el("practiceQuestionPicker");
    if (picker) {
      if (![...picker.options].some((o) => o.value === id)) {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = title || "Selected bank question";
        picker.appendChild(opt);
      }
      picker.value = id;
      picker.classList.add("is-flash");
      window.setTimeout(() => picker.classList.remove("is-flash"), 1400);
    }
    el("practiceStartBtn")?.focus({ preventScroll: true });
    showToast(`Opening question set${title ? `: ${title}` : ""}. Press Start when you are ready.`, 3000);
    return true;
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
    setTargetLevel,
    selectBankQuestion,
    registerStudioPlugin,
    pushWorkspaceSnapshot,
    previewVoice,
    stopSpeaking,
    handleEscape: () => {
      if (local.isSpeaking) {
        stopSpeaking();
        return true;
      }
      if (local.opAbortController) {
        return cancelActiveOp();
      }
      return false;
    },
    toggleVoiceShortcut: () => {
      if (!local.session) return false;
      if (local.session.phase === "active") {
        toggleMicRecording("answer");
        return true;
      }
      if (local.session.phase === "completed" && el("practiceRetryMicBtn")) {
        toggleMicRecording("retry");
        return true;
      }
      return false;
    },
    hasUnsavedWork: () =>
      Boolean(
        local.session &&
          (micMachine.state === "recording" ||
            local.workspaceDirty ||
            Boolean((el("practiceAnswerInput")?.value || "").trim()))
      ),
    getStateSnapshot,
    isPracticeActive: () => Boolean(local.session),
    isPracticeStartingOrActive: () =>
      Boolean(local.session) || opMachine.state === "starting" || opMachine.state === "opening",
    isHomePracticeMode: () => local.homeMode === "practice",
  };
}
