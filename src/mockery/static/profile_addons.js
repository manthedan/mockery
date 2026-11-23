/* Candidate Profile (Resume + Rampart PII Redaction + Target JD + Question Banks)
 * and Xcode-style Add-Ons & Modules manager.
 */

import {
  createPackOpTracker,
  createRedactionQueueCoordinator,
  formatApiDetail,
} from "./state_machines.js";

const CATEGORY_LABELS = {
  behavioral: "Behavioral & Leadership",
  system_design: "System Design",
  coding: "Coding",
};

const LABEL_FRIENDLY = {
  PERSON_NAME: "Name",
  FIRST_NAME: "First Name",
  LAST_NAME: "Last Name",
  EMAIL: "Email",
  PHONE: "Phone",
  URL: "URL",
  STREET_ADDRESS: "Address",
  CITY: "City",
  STATE: "State",
  ZIP_CODE: "ZIP",
  SSN: "SSN",
  CREDIT_CARD: "Card",
  IP_ADDRESS: "IP",
  COMPANY: "Employer",
  SCHOOL: "School",
  CUSTOM: "Custom",
};

function friendlyLabel(raw) {
  return LABEL_FRIENDLY[raw] || String(raw || "PII").replace(/_/g, " ");
}

export function jdRoleLabel(jd) {
  if (!jd?.title) return "";
  return jd.company ? `${jd.title} (${jd.company})` : jd.title;
}

function fmtPackSize(mb) {
  const n = Number(mb);
  if (!Number.isFinite(n) || n <= 0) return "";
  return `${n >= 10 ? Math.round(n) : Math.round(n * 10) / 10} MB`;
}

function fmtBytesMb(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return "0 MB";
  const mb = n / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(1)} MB`;
}

function fmtEta(secs) {
  const s = Number(secs);
  if (!Number.isFinite(s) || s < 1) return "";
  if (s < 60) return `~${Math.round(s)}s left`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem > 0 ? `~${m}m ${rem}s left` : `~${m}m left`;
}

export function formatDownloadMeta(progress, fallbackDetail = "Downloading…") {
  if (!progress || typeof progress !== "object") {
    return fallbackDetail || "Downloading…";
  }
  if (progress.phase === "retrying" && progress.retry_note) {
    return progress.retry_note;
  }
  if (progress.phase === "verifying") return "Verifying checksum…";
  if (progress.phase === "loading") return "Initializing on Mac…";
  const parts = [];
  if (progress.bytes_done > 0 && progress.bytes_total > 0) {
    parts.push(`${fmtBytesMb(progress.bytes_done)} / ${fmtBytesMb(progress.bytes_total)}`);
  } else if (progress.bytes_done > 0) {
    parts.push(fmtBytesMb(progress.bytes_done));
  }
  if (progress.speed_bps > 1024) {
    const mbps = progress.speed_bps / (1024 * 1024);
    parts.push(mbps >= 1 ? `${mbps.toFixed(1)} MB/s` : `${Math.round(progress.speed_bps / 1024)} KB/s`);
  }
  const eta = fmtEta(progress.eta_seconds);
  if (eta) parts.push(eta);
  return parts.length > 0 ? parts.join(" · ") : fallbackDetail || "Downloading…";
}

function _escapeHtmlIntrinsic(val) {
  return String(val ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderDownloadProgressBarHtml(progress, fallbackDetail = "Downloading…") {
  const pct =
    progress && typeof progress.percent === "number" && Number.isFinite(progress.percent)
      ? Math.max(0, Math.min(100, progress.percent))
      : null;
  const indeterminate =
    pct === null || progress?.phase === "retrying" || progress?.phase === "verifying" || progress?.phase === "loading";
  const widthStyle = indeterminate ? "" : `style="width:${pct.toFixed(1)}%"`;
  const metaText = formatDownloadMeta(progress, fallbackDetail);
  const pctText = pct !== null && progress?.phase === "downloading" ? `${Math.round(pct)}%` : "";
  return `
    <div class="dl-bar-wrap">
      <div class="dl-bar"><div class="dl-bar-fill${indeterminate ? " is-indeterminate" : ""}" ${widthStyle}></div></div>
      <div class="dl-bar-meta">
        <span class="dl-bar-meta-text">${_escapeHtmlIntrinsic(metaText)}</span>
        <span class="dl-bar-meta-pct">${_escapeHtmlIntrinsic(pctText)}</span>
      </div>
    </div>
  `;
}

export function patchDownloadProgressBarEl(containerEl, progress, fallbackDetail = "Downloading…") {
  if (!containerEl) return;
  const fill = containerEl.querySelector(".dl-bar-fill");
  const metaEl = containerEl.querySelector(".dl-bar-meta-text");
  const pctEl = containerEl.querySelector(".dl-bar-meta-pct");
  if (!fill || !metaEl || !pctEl) {
    const barHtml = renderDownloadProgressBarHtml(progress, fallbackDetail);
    containerEl.innerHTML = `${barHtml}`;
    return;
  }
  const pct =
    progress && typeof progress.percent === "number" && Number.isFinite(progress.percent)
      ? Math.max(0, Math.min(100, progress.percent))
      : null;
  const indeterminate =
    pct === null || progress?.phase === "retrying" || progress?.phase === "verifying" || progress?.phase === "loading";
  fill.classList.toggle("is-indeterminate", indeterminate);
  fill.style.width = indeterminate ? "" : `${pct.toFixed(1)}%`;
  metaEl.textContent = formatDownloadMeta(progress, fallbackDetail);
  pctEl.textContent = pct !== null && progress?.phase === "downloading" ? `${Math.round(pct)}%` : "";
}

export function reconcileKeyedRows(container, items, { idAttr, sigAttr, getId, getSig, renderRowHtml, patchRow }) {
  if (!container) return;
  if (!items || items.length === 0) {
    container.innerHTML = "";
    return;
  }
  const existingById = new Map();
  for (const child of Array.from(container.children)) {
    const id = child.getAttribute?.(idAttr);
    if (id) existingById.set(id, child);
  }
  const desiredNodes = [];
  for (const item of items) {
    const id = String(getId(item));
    const sig = String(getSig(item));
    const prev = existingById.get(id);
    if (prev && prev.getAttribute(sigAttr) === sig) {
      if (typeof patchRow === "function") patchRow(prev, item);
      desiredNodes.push(prev);
      existingById.delete(id);
    } else {
      const tpl = document.createElement("template");
      tpl.innerHTML = renderRowHtml(item, sig).trim();
      const nextNode = tpl.content.firstElementChild;
      if (nextNode) {
        desiredNodes.push(nextNode);
      }
    }
  }
  for (const stale of existingById.values()) {
    stale.remove();
  }
  // Remove any non-keyed placeholder children (e.g. empty notes)
  const desiredSet = new Set(desiredNodes);
  for (const child of Array.from(container.children)) {
    if (!desiredSet.has(child)) child.remove();
  }
  // Reconcile order without moving nodes that are already in place (preserves focus)
  for (let i = 0; i < desiredNodes.length; i += 1) {
    const node = desiredNodes[i];
    const currentAtIdx = container.children[i];
    if (currentAtIdx !== node) {
      container.insertBefore(node, currentAtIdx || null);
    }
  }
}

export function createProfileAddonsController({
  state,
  el,
  escapeHtml,
  showToast,
  setModalState,
  onRoleApply,
  onLevelApply = null,
  isPracticeHome = null,
  onAddonsChange,
  onPracticeQuestion = null,
  clientMachines = null,
  packOps = null,
}) {
  const sharedPackOps = packOps || createPackOpTracker(clientMachines);
  const redactionCoord = createRedactionQueueCoordinator();

  const local = {
    profile: null,
    addons: null,
    latestServerEpoch: null,
    latestAddonsRevision: -1,
    activeTab: "resume",
    banksData: null,
    banksState: "idle", // "idle" | "loading" | "ready" | "error"
    banksError: "",
    banksAbort: null,
    banksReqId: 0,
    bankCategory: "",
    bankLevel: "",
    bankMatchJd: true,
    pasteResumeOpen: false,
    pasteJdOpen: false,
    expandedJdId: null,
    busyResumeMutation: false,
    busyJdImport: false,
    busyJdIds: new Set(),
    busyModules: new Set(),
    // Home role headline: hand edits end the automatic target-job alignment.
    roleEditedByHand: false,
    autoRoleLabel: "",
  };

  sharedPackOps.subscribe(() => {
    renderAddonsSection();
    renderResumeTab();
  });

  function getActiveJd(profile = local.profile) {
    if (!profile || !Array.isArray(profile.job_listings) || profile.job_listings.length === 0) {
      return null;
    }
    if (profile.active_jd_id) {
      const found = profile.job_listings.find((j) => j.id === profile.active_jd_id);
      if (found) return found;
    }
    return profile.job_listings[0];
  }

  /* Practice Home opens aligned with the active target job: while the role headline is
     still the untouched stock default (or the label this function last wrote), adopt
     the JD's role and level. A hand edit ends the auto-sync; "Use JD role" snaps back. */
  function autoAlignHomeToJd(jd) {
    const input = el("setupRoleInput");
    if (!jd?.title || !input || local.roleEditedByHand) return;
    if (typeof isPracticeHome !== "function" || !isPracticeHome()) return;
    if (document.activeElement === input) return;
    const current = input.value.trim();
    const label = jdRoleLabel(jd);
    if (current !== label) {
      const untouched =
        current === input.defaultValue || (local.autoRoleLabel !== "" && current === local.autoRoleLabel);
      if (!untouched) return;
      input.value = label;
    }
    // Already on the JD role (just written, or applied in an earlier visit): the level follows too.
    local.autoRoleLabel = label;
    if (typeof onLevelApply === "function") onLevelApply(jd.target_level, { auto: true });
  }

  function renderProfileSummaryIndicators(summary = null) {
    const sum = summary || buildSummaryFromProfile(local.profile);
    state.profileSummary = sum;

    // Topbar indicator dot
    const dot = el("profileReadyDot");
    if (dot) {
      const hasAny = Boolean(sum?.has_resume || sum?.active_jd);
      dot.classList.toggle("hidden", !hasAny);
    }

    // Home launchpad context summary
    const homeSummary = el("homeProfileSummary");
    const homeApplyBtn = el("homeApplyJdRoleBtn");
    const jd = sum?.active_jd || null;
    if (homeSummary) {
      const parts = [];
      if (sum?.has_resume) {
        const cnt = Number(sum.resume_redacted_count || 0);
        parts.push(
          cnt > 0
            ? `Resume added · ${cnt} personal detail${cnt === 1 ? "" : "s"} hidden`
            : "Resume added · kept private"
        );
      }
      if (jd && !sum?.has_resume) parts.push("Target job saved");
      if (parts.length > 0) {
        homeSummary.textContent = parts.join("  ·  ");
        homeSummary.classList.add("is-populated");
      } else {
        homeSummary.textContent = "Add your resume and a target job for tailored questions";
        homeSummary.classList.remove("is-populated");
      }
    }

    // Target job line under the role headline: shows what the session is aimed at,
    // and offers the one-tap "Use JD role" chip only while the role differs.
    const jdLine = el("homeJdTargetLine");
    const jdText = el("homeJdTargetText");
    autoAlignHomeToJd(jd);
    const roleVal = String(el("setupRoleInput")?.value || "").trim();
    const applied = Boolean(jd) && roleVal !== "" && roleVal === jdRoleLabel(jd);
    if (jdLine) {
      jdLine.classList.toggle("hidden", !jd);
      jdLine.classList.toggle("is-applied", applied);
    }
    if (jdText && jd) {
      const label = jd.company ? `${jd.title} · ${jd.company}` : jd.title;
      jdText.textContent = applied ? `From your active target job · ${jd.company || jd.title}` : `Target job: ${label}`;
      jdText.title = label;
    }
    if (homeApplyBtn) {
      homeApplyBtn.classList.toggle("hidden", !jd || applied);
    }
  }

  function buildSummaryFromProfile(profile) {
    if (!profile) return null;
    const jd = getActiveJd(profile);
    const r = profile.resume;
    return {
      has_resume: Boolean(r && r.char_count > 0),
      resume_filename: r?.filename || "",
      resume_redacted_count: r?.redacted_count || 0,
      resume_engine: r?.engine || "rules_only",
      active_jd: jd
        ? {
            id: jd.id,
            title: jd.title,
            company: jd.company,
            target_level: jd.target_level,
          }
        : null,
      jd_count: (profile.job_listings || []).length,
    };
  }

  function syncPackOpsFromAddons(addons) {
    const packs = addons?.packs || [];
    const activeById = new Map();
    for (const pk of packs) {
      if (pk.state === "installing") activeById.set(pk.id, "installing");
      else if (pk.state === "uninstalling") activeById.set(pk.id, "uninstalling");
    }
    const snap = sharedPackOps.snapshot();
    for (const [pid, op] of Object.entries(snap)) {
      if (!activeById.has(pid)) {
        sharedPackOps.end(pid);
      } else if (activeById.get(pid) !== op) {
        sharedPackOps.end(pid);
        sharedPackOps.begin(pid, activeById.get(pid));
      }
    }
    for (const [pid, op] of activeById.entries()) {
      if (!sharedPackOps.isBusy(pid)) {
        sharedPackOps.begin(pid, op);
      }
    }
  }

  function acceptAddonsSnapshot(addons) {
    if (!addons) return false;
    const epoch = typeof addons.server_epoch === "string" && addons.server_epoch ? addons.server_epoch : null;
    if (epoch && local.latestServerEpoch && epoch !== local.latestServerEpoch) {
      // Server restarted with a new epoch: reset revision tracking so fresh snapshots are accepted.
      local.latestAddonsRevision = -1;
    }
    if (epoch) {
      local.latestServerEpoch = epoch;
    }
    const rev = Number(addons.revision);
    if (Number.isFinite(rev)) {
      if (rev < local.latestAddonsRevision) {
        return false;
      }
      local.latestAddonsRevision = rev;
    }
    local.addons = addons;
    state.addons = addons;
    syncPackOpsFromAddons(addons);
    return true;
  }

  function applyStatusSnapshot(statusData) {
    if (statusData?.addons) {
      if (!acceptAddonsSnapshot(statusData.addons)) {
        return false;
      }
    }
    if (statusData?.profile_summary) {
      renderProfileSummaryIndicators(statusData.profile_summary);
    }
    if (statusData?.profile) {
      local.profile = statusData.profile;
      renderProfileSummaryIndicators();
      if (state.modalState === "profile") {
        renderProfileModal();
      }
    }
    if (statusData?.addons) {
      renderAddonsSection();
      if (typeof onAddonsChange === "function") {
        onAddonsChange(statusData.addons, statusData.tts || null);
      }
    }
    return true;
  }

  /* ---------------- Profile Modal Tabs ---------------- */

  function setProfileTab(tab, { skipLoadBanks = false } = {}) {
    local.activeTab = tab;
    for (const btn of document.querySelectorAll("[data-profile-tab]")) {
      const isAct = btn.getAttribute("data-profile-tab") === tab;
      btn.classList.toggle("active", isAct);
      btn.setAttribute("aria-selected", isAct ? "true" : "false");
    }
    el("profileTabResume")?.classList.toggle("hidden", tab !== "resume");
    el("profileTabJd")?.classList.toggle("hidden", tab !== "jd");
    el("profileTabBanks")?.classList.toggle("hidden", tab !== "banks");
    // The privacy footnote is about the resume; it has no business on Question Banks.
    el("profileFooterHint")?.classList.toggle("hidden", tab === "banks");
    el("profileModal")?.setAttribute("data-profile-tab-active", tab);

    if (tab === "banks" && !skipLoadBanks) {
      loadBanks();
    }
  }

  async function openProfileModal(initialTab = "resume") {
    setProfileTab(initialTab, { skipLoadBanks: true });
    setModalState("profile");
    await refreshProfile();
    if (local.activeTab === "banks") {
      await loadBanks();
    }
  }

  async function refreshProfile() {
    try {
      const res = await fetch("/api/profile");
      if (!res.ok) return;
      local.profile = await res.json();
      renderProfileModal();
      renderProfileSummaryIndicators();
    } catch {
      /* ignore */
    }
  }

  /* ---------------- Tab 1: Resume & Rampart PII Redaction ---------------- */

  function renderHighlightedRedactedText(redactedText, spans) {
    if (!redactedText) return "";
    // Build a lookup from placeholder -> span metadata so placeholders render as styled chips
    const placeholderMap = new Map();
    for (const sp of spans || []) {
      if (sp.active && sp.placeholder && !placeholderMap.has(sp.placeholder)) {
        placeholderMap.set(sp.placeholder, sp);
      }
    }
    if (placeholderMap.size === 0) {
      return escapeHtml(redactedText);
    }
    // Split text on `[LABEL_N]` tokens
    const parts = redactedText.split(/(\[[A-Z_]+_\d+\])/g);
    return parts
      .map((part) => {
        const sp = placeholderMap.get(part);
        if (!sp) return escapeHtml(part);
        const num = Number((part.match(/_(\d+)\]$/) || [])[1] || 1);
        const name = friendlyLabel(sp.label);
        return `<span class="pii-inline-tag" title="${escapeHtml(
          `Hidden — the AI only sees ${part}`
        )}">${escapeHtml(num > 1 ? `${name} ${num}` : name)}</span>`;
      })
      .join("");
  }

  function renderResumeTab() {
    const p = local.profile;
    const r = p?.resume || null;
    const rampart = p?.rampart || {};

    // Engine banner
    const engineBadge = el("rampartEngineBadge");
    const installBtn = el("installRampartQuickBtn");
    const rampartReady = Boolean(rampart.ready ?? (rampart.installed && !rampart.error));
    if (engineBadge) {
      if (rampartReady) {
        engineBadge.textContent = "Smart detection on";
        engineBadge.title = "On-device Rampart classifier + pattern rules";
        engineBadge.classList.add("is-ok");
      } else if (rampart.error) {
        engineBadge.textContent = "Basic detection";
        engineBadge.title = `Resume Privacy Scrubber could not start (${rampart.error}). Pattern rules are still protecting you.`;
        engineBadge.classList.remove("is-ok");
      } else {
        engineBadge.textContent = "Basic detection";
        engineBadge.title = "Pattern rules catch emails, phones, and links. Add Resume Privacy Scrubber for names and employers.";
        engineBadge.classList.remove("is-ok");
      }
    }
    if (installBtn) {
      const installing = sharedPackOps.isInstalling("rampart_pii");
      const uninstalling = sharedPackOps.isUninstalling("rampart_pii");
      const busy = installing || uninstalling;
      installBtn.classList.toggle("hidden", rampartReady && !busy);
      installBtn.disabled = busy;
      installBtn.textContent = installing
        ? "Downloading…"
        : uninstalling
          ? "Removing…"
          : rampart.installed
            ? "Retry scrubber"
            : "Add scrubber · 15 MB";
    }

    const emptyWrap = el("resumeEmptyState");
    const loadedWrap = el("resumeLoadedState");
    if (!r) {
      emptyWrap?.classList.remove("hidden");
      loadedWrap?.classList.add("hidden");
      el("resumePasteWrap")?.classList.toggle("hidden", !local.pasteResumeOpen);
      return;
    }

    emptyWrap?.classList.add("hidden");
    loadedWrap?.classList.remove("hidden");
    el("resumePasteWrap")?.classList.toggle("hidden", !local.pasteResumeOpen);

    el("resumeFileTitle").textContent = r.filename || "resume.txt";
    const detected = Number(r.total_detected || 0);
    el("resumeFileMeta").textContent =
      detected > 0
        ? `${r.char_count.toLocaleString()} characters · ${r.redacted_count} of ${detected} personal detail${
            detected === 1 ? "" : "s"
          } hidden`
        : `${r.char_count.toLocaleString()} characters · nothing personal found`;

    const prefs = r.prefs || {};
    const keepLocEl = el("prefKeepLocation");
    const hideOrgEl = el("prefHideOrgs");
    const customTermsEl = el("prefCustomTerms");
    if (keepLocEl) keepLocEl.checked = prefs.keep_location !== false;
    if (hideOrgEl) hideOrgEl.checked = Boolean(prefs.hide_employers_and_schools);
    if (customTermsEl && document.activeElement !== customTermsEl) {
      customTermsEl.value = (prefs.custom_terms || []).join(", ");
    }

    // Deduplicated clickable pills for detected spans using privacy-safe masked display_value
    const chipsEl = el("resumePiiChips");
    if (chipsEl) {
      const seen = new Map();
      for (const sp of r.spans || []) {
        const k = `${sp.placeholder}::${sp.active ? "1" : "0"}`;
        if (!seen.has(k)) {
          seen.set(k, sp);
        }
      }
      if (seen.size === 0) {
        chipsEl.innerHTML = `<span class="set-note">Nothing personal found in this resume.</span>`;
      } else {
        chipsEl.innerHTML = Array.from(seen.values())
          .map((sp) => {
            const activeCls = sp.active ? "is-redacted" : "is-kept";
            const actionHint = sp.active
              ? "Hidden from the AI — click to make visible"
              : "Visible to the AI — click to hide";
            const disp = String(sp.display_value || sp.placeholder || "");
            const previewVal = disp.length > 30 ? `${disp.slice(0, 27)}…` : disp;
            return `
              <button
                type="button"
                class="pii-chip ${activeCls}"
                data-toggle-span-id="${escapeHtml(sp.id)}"
                aria-pressed="${sp.active ? "true" : "false"}"
                title="${escapeHtml(`${friendlyLabel(sp.label)}: ${disp} — ${actionHint}`)}"
              >
                <span class="pii-chip-label">${escapeHtml(friendlyLabel(sp.label))}</span>
                <span class="pii-chip-orig">${escapeHtml(previewVal)}</span>
                <span class="pii-chip-state">${sp.active ? "Hidden" : "Visible"}</span>
              </button>
            `;
          })
          .join("");
      }
    }

    const previewEl = el("resumeRedactedPreview");
    if (previewEl) {
      previewEl.innerHTML = renderHighlightedRedactedText(r.redacted_text || "", r.spans || []);
    }
  }

  async function uploadResumeFile(file) {
    if (!file || local.busyResumeMutation) return;
    local.busyResumeMutation = true;
    clientMachines?.beginOp("resume_upload", { filename: file.name });
    const form = new FormData();
    form.append("file", file);
    showToast(`Scanning & scrubbing ${file.name} locally…`, 2500);
    try {
      const res = await fetch("/api/profile/resume/file", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showToast(formatApiDetail(data.detail, `Could not read resume (${res.status}).`), 4000);
        if (!redactionCoord.busy) await refreshProfile();
        return;
      }
      redactionCoord.invalidate();
      local.profile = data;
      local.pasteResumeOpen = false;
      renderProfileModal();
      renderProfileSummaryIndicators();
      showToast(
        `Resume scrubbed locally (${data.resume?.redacted_count || 0} PII items hidden).`,
        2600
      );
    } catch (err) {
      showToast(`Upload failed: ${err.message || "network error"}`);
      if (!redactionCoord.busy) await refreshProfile();
    } finally {
      local.busyResumeMutation = false;
      clientMachines?.endOp("resume_upload");
    }
  }

  async function submitPastedResume() {
    if (local.busyResumeMutation) return;
    const input = el("resumePasteTextarea");
    const text = input?.value?.trim() || "";
    if (!text) {
      showToast("Paste your resume text first.");
      return;
    }
    local.busyResumeMutation = true;
    clientMachines?.beginOp("resume_paste");
    try {
      const res = await fetch("/api/profile/resume/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: "pasted_resume.txt", text }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showToast(formatApiDetail(data.detail, "Could not process resume text."));
        if (!redactionCoord.busy) await refreshProfile();
        return;
      }
      redactionCoord.invalidate();
      local.profile = data;
      local.pasteResumeOpen = false;
      if (input) input.value = "";
      renderProfileModal();
      renderProfileSummaryIndicators();
      showToast(
        `Resume scrubbed locally (${data.resume?.redacted_count || 0} PII items hidden).`,
        2600
      );
    } catch (err) {
      showToast(`Could not process resume: ${err.message || "network error"}`);
      if (!redactionCoord.busy) await refreshProfile();
    } finally {
      local.busyResumeMutation = false;
      clientMachines?.endOp("resume_paste");
    }
  }

  function readRedactionPrefsFromDom() {
    const currentPrefs = local.profile?.resume?.prefs || {};
    const customRaw = el("prefCustomTerms")?.value || "";
    const customTerms = customRaw
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length >= 2);
    return {
      keep_location: Boolean(el("prefKeepLocation")?.checked),
      hide_employers_and_schools: Boolean(el("prefHideOrgs")?.checked),
      custom_terms: customTerms,
      unredacted_values: currentPrefs.unredacted_values || [],
      use_rampart: true,
    };
  }

  async function drainRedactionQueue() {
    clientMachines?.beginOp("resume_redaction");
    try {
      await redactionCoord.drain(async (op, isCurrent, isQueueEmpty) => {
        if (!local.profile?.resume || !isCurrent()) return;
        let res;
        if (op.type === "prefs") {
          const payload = {
            ...op.payload,
            unredacted_values: local.profile.resume.prefs?.unredacted_values || [],
          };
          res = await fetch("/api/profile/resume/redact", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
        } else if (op.type === "toggle") {
          const docId = op.docId ?? local.profile?.resume?.doc_id;
          const qs = docId ? `?doc_id=${encodeURIComponent(docId)}` : "";
          res = await fetch(`/api/profile/resume/toggle/${encodeURIComponent(op.spanId)}${qs}`, {
            method: "POST",
          });
        }
        if (!res) return;
        const data = await res.json().catch(() => ({}));
        // Discard stale response if resume was deleted or replaced while in flight
        if (!isCurrent()) return;
        if (!res.ok) {
          showToast(formatApiDetail(data.detail, "Could not update redaction."));
          if (res.status === 409) {
            // Drop any remaining queued toggles targeting the superseded resume before refreshing
            redactionCoord.invalidate();
            await refreshProfile();
          }
          return;
        }
        local.profile = data;
        // Only redraw DOM controls once no newer prefs update is waiting in the queue
        if (isQueueEmpty()) {
          renderProfileModal();
          renderProfileSummaryIndicators();
        }
      });
    } catch (err) {
      showToast(`Redaction update failed: ${err.message || "network error"}`);
    } finally {
      if (!redactionCoord.busy) {
        clientMachines?.endOp("resume_redaction");
      }
    }
  }

  async function saveRedactionPrefs() {
    if (!local.profile?.resume || local.busyResumeMutation) return;
    const payload = readRedactionPrefsFromDom();
    redactionCoord.enqueue({ type: "prefs", payload });
    await drainRedactionQueue();
  }

  async function toggleSpanRedaction(spanId) {
    if (!local.profile?.resume || !spanId || local.busyResumeMutation) return;
    redactionCoord.enqueue({
      type: "toggle",
      spanId,
      docId: local.profile.resume.doc_id || "",
    });
    await drainRedactionQueue();
  }

  async function deleteResume() {
    if (local.busyResumeMutation) return;
    local.busyResumeMutation = true;
    clientMachines?.beginOp("resume_delete");
    try {
      const res = await fetch("/api/profile/resume", { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showToast(formatApiDetail(data.detail, "Could not remove resume."));
        if (!redactionCoord.busy) await refreshProfile();
        return;
      }
      redactionCoord.invalidate();
      local.profile = await res.json();
      renderProfileModal();
      renderProfileSummaryIndicators();
      showToast("Resume removed from this Mac.", 2200);
    } catch (err) {
      showToast(`Could not remove resume: ${err.message || "network error"}`);
      if (!redactionCoord.busy) await refreshProfile();
    } finally {
      local.busyResumeMutation = false;
      clientMachines?.endOp("resume_delete");
    }
  }

  /* ---------------- Tab 2: Target Job Listings ---------------- */

  function renderJdTab() {
    const listEl = el("jdListContainer");
    el("jdPasteWrap")?.classList.toggle("hidden", !local.pasteJdOpen);
    if (!listEl) return;

    const listings = local.profile?.job_listings || [];
    const activeId = local.profile?.active_jd_id || listings[0]?.id || null;

    if (listings.length === 0) {
      listEl.innerHTML = `<div class="profile-empty-note">No target job listings saved yet. Paste a Greenhouse, Lever, Ashby, or careers URL above — or paste the job description text.</div>`;
      return;
    }

    listEl.innerHTML = listings
      .map((jd) => {
        const isActive = jd.id === activeId;
        const isExpanded = local.expandedJdId === jd.id;
        const subtitle = [jd.company, jd.target_level, `${jd.text.length.toLocaleString()} characters`]
          .filter(Boolean)
          .join(" · ");
        return `
          <div class="jd-card ${isActive ? "is-active" : ""}">
            <div class="jd-card-top">
              <div class="jd-card-info">
                <div class="jd-card-title">
                  ${escapeHtml(jd.title)}
                  ${isActive ? `<span class="jd-active-pill">Active</span>` : ""}
                </div>
                <div class="jd-card-meta">${escapeHtml(subtitle)}</div>
              </div>
              <div class="jd-card-actions">
                ${
                  !isActive
                    ? `<button type="button" class="btn btn-outline btn-sm" data-jd-select="${escapeHtml(
                        jd.id
                      )}">Set active</button>`
                    : ""
                }
                <button type="button" class="btn btn-ghost btn-sm" data-jd-apply-role="${escapeHtml(
                  jd.id
                )}" title="Fill the interview role field with this title">Use role</button>
                <button type="button" class="btn btn-ghost btn-sm" data-jd-toggle="${escapeHtml(
                  jd.id
                )}">${isExpanded ? "Hide" : "Preview"}</button>
                <button type="button" class="btn btn-ghost btn-sm is-danger-text" data-jd-delete="${escapeHtml(
                  jd.id
                )}">Remove</button>
              </div>
            </div>
            ${
              isExpanded
                ? `<pre class="jd-preview-text">${escapeHtml(jd.text.slice(0, 4000))}${
                    jd.text.length > 4000 ? "\n…" : ""
                  }</pre>`
                : ""
            }
          </div>
        `;
      })
      .join("");
  }

  async function importJobListing() {
    if (local.busyJdImport) return;
    const urlInput = el("jdUrlInput");
    const levelSelect = el("jdLevelSelect");
    const titleInput = el("jdPasteTitleInput");
    const companyInput = el("jdPasteCompanyInput");
    const textInput = el("jdPasteTextarea");
    const statusEl = el("jdImportStatus");
    const btn = el("jdImportBtn");

    const url = urlInput?.value?.trim() || "";
    const text = local.pasteJdOpen ? textInput?.value?.trim() || "" : "";
    const title = local.pasteJdOpen ? titleInput?.value?.trim() || "" : "";
    const company = local.pasteJdOpen ? companyInput?.value?.trim() || "" : "";
    const target_level = levelSelect?.value || "L5";

    if (!url && !text) {
      showToast("Paste a job listing URL or switch to Paste text.");
      return;
    }

    local.busyJdImport = true;
    if (btn) {
      btn.disabled = true;
      btn.textContent = url && !text ? "Importing…" : "Saving…";
    }
    if (statusEl) {
      statusEl.classList.remove("hidden", "is-ok", "is-err");
      statusEl.textContent = url && !text ? "Fetching & cleaning job listing…" : "Saving…";
    }

    try {
      const res = await fetch("/api/profile/jd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, text, title, company, target_level }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (statusEl) {
          statusEl.classList.add("is-err");
          statusEl.textContent = formatApiDetail(data.detail, `Import failed (${res.status})`);
        }
        return;
      }
      local.profile = data;
      const imported = data.imported_jd;
      if (urlInput) urlInput.value = "";
      if (textInput) textInput.value = "";
      if (titleInput) titleInput.value = "";
      if (companyInput) companyInput.value = "";
      local.pasteJdOpen = false;
      local.banksData = null; // Invalidate cached bank ranking for new active JD
      if (statusEl) {
        statusEl.classList.add("is-ok");
        statusEl.textContent = imported
          ? `Imported ${imported.title}${imported.company ? ` at ${imported.company}` : ""} (${imported.text.length.toLocaleString()} chars).`
          : "Job listing saved.";
      }
      renderProfileModal();
      renderProfileSummaryIndicators();
    } catch (err) {
      if (statusEl) {
        statusEl.classList.add("is-err");
        statusEl.textContent = err.message || "Network error";
      }
    } finally {
      local.busyJdImport = false;
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Import";
      }
    }
  }

  async function selectJobListing(jdId) {
    if (!jdId || local.busyJdIds.has(jdId)) return;
    local.busyJdIds.add(jdId);
    try {
      const res = await fetch(`/api/profile/jd/${encodeURIComponent(jdId)}/select`, {
        method: "POST",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showToast(formatApiDetail(data.detail, "Could not select job listing."));
        return;
      }
      local.profile = await res.json();
      local.banksData = null;
      renderProfileModal();
      renderProfileSummaryIndicators();
    } catch (err) {
      showToast(`Could not select job listing: ${err.message || "network error"}`);
    } finally {
      local.busyJdIds.delete(jdId);
    }
  }

  async function deleteJobListing(jdId) {
    if (!jdId || local.busyJdIds.has(jdId)) return;
    local.busyJdIds.add(jdId);
    try {
      const res = await fetch(`/api/profile/jd/${encodeURIComponent(jdId)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showToast(formatApiDetail(data.detail, "Could not remove job listing."));
        return;
      }
      local.profile = await res.json();
      local.banksData = null;
      renderProfileModal();
      renderProfileSummaryIndicators();
    } catch (err) {
      showToast(`Could not remove job listing: ${err.message || "network error"}`);
    } finally {
      local.busyJdIds.delete(jdId);
    }
  }

  /* ---------------- Tab 3: Curated Question Banks ---------------- */

  async function loadBanks() {
    if (local.banksAbort) {
      try {
        local.banksAbort.abort();
      } catch {
        /* ignore */
      }
      local.banksAbort = null;
    }
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    local.banksAbort = ctrl;
    local.banksReqId += 1;
    const reqId = local.banksReqId;
    local.banksState = "loading";
    local.banksError = "";

    const listEl = el("banksListContainer");
    if (listEl) {
      listEl.innerHTML = `<div class="profile-empty-note">Loading question banks…</div>`;
    }
    const params = new URLSearchParams();
    if (local.bankCategory) params.set("category", local.bankCategory);
    if (local.bankLevel) params.set("level", local.bankLevel);
    params.set("match_active_jd", local.bankMatchJd ? "true" : "false");

    try {
      const res = await fetch(`/api/banks?${params.toString()}`, ctrl ? { signal: ctrl.signal } : {});
      if (reqId !== local.banksReqId) return;
      const data = await res.json().catch(() => ({}));
      if (reqId !== local.banksReqId) return;
      if (!res.ok) {
        throw new Error(formatApiDetail(data.detail, `Could not load question banks (${res.status}).`));
      }
      local.banksData = data;
      local.banksState = "ready";
      renderBanksTab();
    } catch (err) {
      if (err?.name === "AbortError" || reqId !== local.banksReqId) return;
      local.banksData = null;
      local.banksState = "error";
      local.banksError = String(err?.message || "Could not load question banks.");
      if (listEl) {
        listEl.innerHTML = `
          <div class="profile-empty-note">
            ${escapeHtml(local.banksError)}
            <div style="margin-top:8px;">
              <button type="button" class="btn btn-outline btn-sm" data-retry-banks>Retry</button>
            </div>
          </div>
        `;
      }
    } finally {
      if (local.banksAbort === ctrl) {
        local.banksAbort = null;
      }
    }
  }

  function renderBanksTab() {
    const listEl = el("banksListContainer");
    const countEl = el("banksCountLabel");
    if (!listEl || !local.banksData) return;

    const questions = local.banksData.questions || [];
    if (countEl) {
      const activeJd = getActiveJd();
      const suffix =
        local.bankMatchJd && activeJd
          ? `, ranked for ${activeJd.title}${activeJd.company ? ` at ${activeJd.company}` : ""}`
          : "";
      countEl.textContent = `${questions.length} question${questions.length === 1 ? "" : "s"}${suffix}`;
    }

    if (questions.length === 0) {
      listEl.innerHTML = `<div class="profile-empty-note">No questions match those filters.</div>`;
      return;
    }

    listEl.innerHTML = questions
      .map((q) => {
        const catLabel = CATEGORY_LABELS[q.category] || q.category;
        const levels = Array.isArray(q.levels) ? q.levels.join(" / ") : "";
        const jdScore = Number(q.jd_relevance_score || 0);
        const title = q.title || q.competency || catLabel;
        const extras = [];
        if (q.carl_focus) extras.push(`Focus: ${q.carl_focus}`);
        if (Array.isArray(q.core_requirements) && q.core_requirements.length > 0) {
          extras.push(`Covers ${q.core_requirements.slice(0, 3).join(", ")}`);
        }
        if (Array.isArray(q.follow_ups) && q.follow_ups.length > 0) {
          extras.push(`${q.follow_ups.length} follow-up${q.follow_ups.length === 1 ? "" : "s"}`);
        }
        const practiceBtn = q.id
          ? `<button
               type="button"
               class="chip-action bank-practice-btn"
               data-bank-practice="${escapeHtml(q.id)}"
               data-bank-category="${escapeHtml(q.category || "")}"
               data-bank-title="${escapeHtml(title)}"
               title="Start a practice session with this question"
             >Practice <span aria-hidden="true">→</span></button>`
          : "";
        return `
          <article class="bank-item">
            <div class="bank-item-head">
              <span class="bank-item-title">${renderInlineCode(title)}</span>
              ${practiceBtn}
            </div>
            <div class="bank-item-meta">
              <span class="bank-cat-pill">${escapeHtml(catLabel)}</span>
              ${levels ? `<span class="bank-level-pill">${escapeHtml(levels)}</span>` : ""}
              ${q.difficulty ? `<span class="bank-level-pill">${escapeHtml(String(q.difficulty))}</span>` : ""}
              ${jdScore > 0 ? `<span class="bank-jd-match">Matches your target job</span>` : ""}
            </div>
            <p class="bank-item-prompt">${renderInlineCode(q.prompt)}</p>
            ${
              extras.length > 0
                ? `<div class="bank-item-foot">${extras
                    .map((x) => `<span>${renderInlineCode(x)}</span>`)
                    .join("")}</div>`
                : ""
            }
          </article>
        `;
      })
      .join("");
  }

  // Bank prompts are authored with inline markdown backticks. Render them as
  // real <code> (never raw backticks) and keep tokens like `->` unbroken.
  function renderInlineCode(text) {
    return escapeHtml(String(text ?? "")).replace(
      /`([^`\n]+)`/g,
      (_m, code) => `<code class="md-code">${code}</code>`
    );
  }

  function renderProfileModal() {
    renderResumeTab();
    renderJdTab();
    if (local.activeTab === "banks" && local.banksState === "ready" && local.banksData) {
      renderBanksTab();
    }
  }

  /* ---------------- Settings Modal: Xcode-style Add-Ons & Modules ---------------- */

  async function refreshAddons() {
    try {
      const res = await fetch("/api/addons");
      if (!res.ok) return;
      const fresh = await res.json();
      applyStatusSnapshot({ addons: fresh });
    } catch {
      /* ignore */
    }
  }

  function renderAddonsSection() {
    const modList = el("addonsModulesList");
    const packList = el("addonsPacksList");
    if (!local.addons) return;

    if (modList) {
      const modules = local.addons.modules || [];
      const existingInputs = Array.from(modList.querySelectorAll("[data-addon-module]"));
      const canPatchInPlace =
        existingInputs.length === modules.length &&
        modules.every((m, idx) => existingInputs[idx]?.getAttribute("data-addon-module") === m.id);
      if (canPatchInPlace) {
        // Patch checked/disabled state in place to preserve keyboard focus across toggles
        modules.forEach((m, idx) => {
          const inp = existingInputs[idx];
          inp.checked = Boolean(m.enabled);
          inp.disabled = local.busyModules.has(m.id);
        });
      } else {
        modList.innerHTML = modules
          .map(
            (m) => `
          <label class="addon-row">
            <span class="addon-row-main">
              <span class="addon-row-title">${escapeHtml(m.label)}</span>
              <span class="addon-row-desc">${escapeHtml(m.description)}</span>
            </span>
            <input
              type="checkbox"
              role="switch"
              class="switch addon-mod-toggle"
              data-addon-module="${escapeHtml(m.id)}"
              ${m.enabled ? "checked" : ""}
              ${local.busyModules.has(m.id) ? "disabled" : ""}
            />
          </label>
        `
          )
          .join("");
      }
    }

    if (packList) {
      const packs = local.addons.packs || [];
      reconcileKeyedRows(packList, packs, {
        idAttr: "data-addon-row-id",
        sigAttr: "data-addon-row-sig",
        getId: (pk) => pk.id,
        getSig: (pk) => {
          const installing = sharedPackOps.isInstalling(pk.id) || pk.state === "installing";
          const uninstalling = sharedPackOps.isUninstalling(pk.id) || pk.state === "uninstalling";
          const cancellable = pk.progress?.cancellable !== false;
          return `${pk.id}|${pk.state}|${installing}|${uninstalling}|${cancellable}|${Boolean(pk.installed)}|${pk.error || ""}`;
        },
        patchRow: (rowEl, pk) => {
          const installing = sharedPackOps.isInstalling(pk.id) || pk.state === "installing";
          if (installing) {
            const progEl = rowEl.querySelector(".addon-pack-progress");
            patchDownloadProgressBarEl(progEl, pk.progress, pk.status_detail || "Downloading…");
          }
        },
        renderRowHtml: (pk, sig) => {
          const installing = sharedPackOps.isInstalling(pk.id) || pk.state === "installing";
          const uninstalling = sharedPackOps.isUninstalling(pk.id) || pk.state === "uninstalling";
          const cancellable = pk.progress?.cancellable !== false;
          const sizeLabel = fmtPackSize(pk.size_mb);
          let actionHtml = "";
          let progressHtml = "";
          if (installing) {
            progressHtml = `<div class="addon-pack-progress">${renderDownloadProgressBarHtml(
              pk.progress,
              pk.status_detail || "Downloading…"
            )}</div>`;
            actionHtml = cancellable
              ? `<button type="button" class="btn btn-ghost btn-sm" data-addon-cancel="${escapeHtml(
                  pk.id
                )}">Cancel</button>`
              : `<span class="addon-state-badge">Finalizing…</span>`;
          } else if (uninstalling) {
            actionHtml = `<button type="button" class="btn btn-outline btn-sm" disabled>Removing…</button>`;
          } else if (pk.state === "installed") {
            actionHtml = `
              <span class="addon-state-badge is-ok">Installed</span>
              <button type="button" class="btn btn-ghost btn-sm addon-remove-btn" data-addon-uninstall="${escapeHtml(
                pk.id
              )}" title="Remove from this Mac${sizeLabel ? ` · frees ${sizeLabel}` : ""}">Remove</button>
            `;
          } else if (pk.state === "error") {
            const retryBtn =
              pk.available !== false
                ? `<button type="button" class="btn btn-outline btn-sm" data-addon-install="${escapeHtml(
                    pk.id
                  )}">Retry</button>`
                : "";
            const removeBtn = pk.installed
              ? `<button type="button" class="btn btn-ghost btn-sm" data-addon-uninstall="${escapeHtml(
                  pk.id
                )}">Remove</button>`
              : "";
            actionHtml = `${retryBtn}${removeBtn}`;
          } else if (pk.state === "coming_soon") {
            actionHtml = `<span class="addon-state-badge">Available soon</span>`;
          } else {
            actionHtml = `<button type="button" class="btn btn-outline btn-sm" data-addon-install="${escapeHtml(
              pk.id
            )}">Install${sizeLabel ? ` · ${sizeLabel}` : ""}</button>`;
          }
          return `
            <div class="addon-row" data-addon-row-id="${escapeHtml(pk.id)}" data-addon-row-sig="${escapeHtml(sig)}">
              <div class="addon-row-main">
                <div class="addon-row-title">
                  <span>${escapeHtml(pk.name)}</span>
                  ${pk.model ? `<span class="addon-model-id">${escapeHtml(pk.model)}</span>` : ""}
                </div>
                <div class="addon-row-desc">${escapeHtml(pk.description)}</div>
                ${progressHtml}
                ${pk.error ? `<div class="addon-row-err">${escapeHtml(pk.error)}</div>` : ""}
              </div>
              <div class="addon-row-actions">${actionHtml}</div>
            </div>
          `;
        },
      });
    }
  }

  async function toggleModule(modId, enabled) {
    if (!modId || local.busyModules.has(modId)) return;
    local.busyModules.add(modId);
    const modList = el("addonsModulesList");
    const targetInput = modList?.querySelector(
      `[data-addon-module="${CSS.escape ? CSS.escape(modId) : modId}"]`
    );
    const hadFocus = Boolean(targetInput && document.activeElement === targetInput);
    if (targetInput) {
      targetInput.disabled = true;
    } else {
      renderAddonsSection();
    }
    try {
      const res = await fetch("/api/addons/modules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modules: { [modId]: enabled } }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showToast(formatApiDetail(data.detail, "Could not update add-on module."));
        return;
      }
      const updatedAddons = await res.json();
      applyStatusSnapshot({ addons: updatedAddons });
    } catch (err) {
      showToast(`Could not update add-on module: ${err.message || "network error"}`);
    } finally {
      local.busyModules.delete(modId);
      renderAddonsSection();
      if (hadFocus) {
        const refocused = el("addonsModulesList")?.querySelector(
          `[data-addon-module="${CSS.escape ? CSS.escape(modId) : modId}"]`
        );
        refocused?.focus?.();
      }
    }
  }

  async function runPackOp(packId, { op, url, method, startToast, okToast, okMs, failFallback, errPrefix, background = false }) {
    if (!packId || !sharedPackOps.begin(packId, op)) return false;
    if (startToast) showToast(startToast, 3000);
    let refreshedFromResponse = false;
    let runningInBackground = false;
    try {
      const res = await fetch(url, { method });
      const data = await res.json().catch(() => ({}));
      if (data.addons || data.profile) {
        refreshedFromResponse = true;
        applyStatusSnapshot({
          addons: data.addons || null,
          tts: data.tts || null,
          profile: data.profile || null,
        });
      }
      if (!res.ok) {
        showToast(formatApiDetail(data.detail, failFallback(res.status)), 4000);
        return false;
      }
      // Check authoritative post-merge pack state so a late 202 never revives a completed pack.
      const authoritativePk = (local.addons?.packs || []).find((p) => p.id === packId);
      if (background && authoritativePk?.state === "installing") {
        runningInBackground = true;
        renderAddonsSection();
        return true;
      }
      if (!background && okToast) showToast(okToast, okMs);
      return true;
    } catch (err) {
      showToast(`${errPrefix}: ${err.message || "network error"}`);
      return false;
    } finally {
      if (!runningInBackground) {
        const authoritativePk = (local.addons?.packs || []).find((p) => p.id === packId);
        if (authoritativePk?.state !== "installing" && authoritativePk?.state !== "uninstalling") {
          sharedPackOps.end(packId);
        }
      }
      if (!refreshedFromResponse) {
        await refreshAddons();
      }
    }
  }

  function installPack(packId) {
    const label =
      (local.addons?.packs || []).find((p) => p.id === packId)?.name ||
      String(packId || "").replace(/_/g, " ");
    return runPackOp(packId, {
      op: "installing",
      url: `/api/addons/${encodeURIComponent(packId)}/install?background=true`,
      method: "POST",
      startToast: `Downloading ${label} in the background…`,
      okToast: `${label} installed and ready.`,
      okMs: 2400,
      failFallback: (status) => `Install failed (${status})`,
      errPrefix: "Install failed",
      background: true,
    });
  }

  async function cancelPack(packId) {
    if (!packId) return false;
    try {
      const res = await fetch(`/api/addons/${encodeURIComponent(packId)}/cancel`, {
        method: "POST",
      });
      const data = await res.json().catch(() => ({}));
      if (data.addons || data.profile) {
        applyStatusSnapshot({
          addons: data.addons || null,
          tts: data.tts || null,
          profile: data.profile || null,
        });
      }
      if (!res.ok) {
        showToast(formatApiDetail(data.detail, "Could not cancel download."), 3200);
        return false;
      }
      showToast("Cancelling download…", 2000);
      return true;
    } catch (err) {
      showToast(`Cancel failed: ${err.message || "network error"}`);
      return false;
    }
  }

  function uninstallPack(packId) {
    return runPackOp(packId, {
      op: "uninstalling",
      url: `/api/addons/${encodeURIComponent(packId)}`,
      method: "DELETE",
      okToast: "Add-on removed from local cache.",
      okMs: 2200,
      failFallback: () => "Could not remove add-on.",
      errPrefix: "Remove failed",
    });
  }

  function getStateSnapshot() {
    return {
      activeTab: local.activeTab,
      banksState: local.banksState,
      redactionBusy: redactionCoord.busy,
      redactionEpoch: redactionCoord.epoch,
      redactionPendingCount: redactionCoord.pendingCount,
      busyResumeMutation: local.busyResumeMutation,
      busyJdImport: local.busyJdImport,
      packOps: sharedPackOps.snapshot(),
    };
  }

  /* ---------------- Event Binding ---------------- */

  function bindEvents() {
    el("profileBtn")?.addEventListener("click", () => openProfileModal("resume"));
    el("homeEditProfileBtn")?.addEventListener("click", () => openProfileModal("resume"));
    el("homeApplyJdRoleBtn")?.addEventListener("click", () => {
      const jd = state.profileSummary?.active_jd;
      if (jd && typeof onRoleApply === "function") {
        onRoleApply(jd);
        renderProfileSummaryIndicators(state.profileSummary);
        showToast(`Role set to ${jd.title}.`, 1800);
      }
    });
    // Keep the "Use JD role" chip honest as the role headline is edited by hand.
    el("setupRoleInput")?.addEventListener("input", () => {
      local.roleEditedByHand = true;
      if (state.profileSummary) renderProfileSummaryIndicators(state.profileSummary);
    });

    el("closeProfileBtn")?.addEventListener("click", () => setModalState("none"));
    el("doneProfileBtn")?.addEventListener("click", () => setModalState("none"));
    el("profileModal")?.addEventListener("click", (ev) => {
      if (ev.target === el("profileModal")) {
        setModalState("none");
      }
    });

    for (const tabBtn of document.querySelectorAll("[data-profile-tab]")) {
      tabBtn.addEventListener("click", () => {
        setProfileTab(tabBtn.getAttribute("data-profile-tab"));
      });
    }

    // Resume upload & dropzone
    const fileInput = el("resumeFileInput");
    el("resumeChooseFileBtn")?.addEventListener("click", () => fileInput?.click());
    el("resumeReplaceBtn")?.addEventListener("click", () => fileInput?.click());
    fileInput?.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (file) {
        uploadResumeFile(file);
        fileInput.value = "";
      }
    });

    const dropzone = el("resumeDropzone");
    if (dropzone) {
      dropzone.addEventListener("dragover", (ev) => {
        ev.preventDefault();
        dropzone.classList.add("is-dragover");
      });
      dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragover"));
      dropzone.addEventListener("drop", (ev) => {
        ev.preventDefault();
        dropzone.classList.remove("is-dragover");
        const file = ev.dataTransfer?.files?.[0];
        if (file) uploadResumeFile(file);
      });
    }

    el("resumeTogglePasteBtn")?.addEventListener("click", () => {
      local.pasteResumeOpen = !local.pasteResumeOpen;
      renderResumeTab();
    });
    el("resumeSubmitPasteBtn")?.addEventListener("click", submitPastedResume);
    el("resumeDeleteBtn")?.addEventListener("click", deleteResume);
    el("installRampartQuickBtn")?.addEventListener("click", () => installPack("rampart_pii"));

    el("prefKeepLocation")?.addEventListener("change", () => saveRedactionPrefs());
    el("prefHideOrgs")?.addEventListener("change", () => saveRedactionPrefs());
    el("prefApplyCustomBtn")?.addEventListener("click", () => saveRedactionPrefs());
    el("prefCustomTerms")?.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        saveRedactionPrefs();
      }
    });

    el("resumePiiChips")?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-toggle-span-id]");
      if (btn) {
        toggleSpanRedaction(btn.getAttribute("data-toggle-span-id"));
      }
    });

    // Job Listing controls
    el("jdImportBtn")?.addEventListener("click", importJobListing);
    el("jdUrlInput")?.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        importJobListing();
      }
    });
    el("jdTogglePasteBtn")?.addEventListener("click", () => {
      local.pasteJdOpen = !local.pasteJdOpen;
      renderJdTab();
    });
    el("jdListContainer")?.addEventListener("click", (ev) => {
      const selBtn = ev.target.closest("[data-jd-select]");
      if (selBtn) {
        selectJobListing(selBtn.getAttribute("data-jd-select"));
        return;
      }
      const applyBtn = ev.target.closest("[data-jd-apply-role]");
      if (applyBtn) {
        const id = applyBtn.getAttribute("data-jd-apply-role");
        const jd = (local.profile?.job_listings || []).find((j) => j.id === id);
        if (jd && typeof onRoleApply === "function") {
          onRoleApply(jd);
          renderProfileSummaryIndicators();
          showToast(`Role set to ${jd.title}.`, 1800);
        }
        return;
      }
      const togBtn = ev.target.closest("[data-jd-toggle]");
      if (togBtn) {
        const id = togBtn.getAttribute("data-jd-toggle");
        local.expandedJdId = local.expandedJdId === id ? null : id;
        renderJdTab();
        return;
      }
      const delBtn = ev.target.closest("[data-jd-delete]");
      if (delBtn) {
        deleteJobListing(delBtn.getAttribute("data-jd-delete"));
      }
    });

    // Question Banks filters
    el("bankCategorySelect")?.addEventListener("change", (ev) => {
      local.bankCategory = ev.target.value;
      loadBanks();
    });
    el("bankLevelSelect")?.addEventListener("change", (ev) => {
      local.bankLevel = ev.target.value;
      loadBanks();
    });
    el("bankMatchJdCheckbox")?.addEventListener("change", (ev) => {
      local.bankMatchJd = Boolean(ev.target.checked);
      loadBanks();
    });
    el("banksListContainer")?.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-retry-banks]")) {
        loadBanks();
        return;
      }
      const practiceBtn = ev.target.closest("[data-bank-practice]");
      if (practiceBtn && typeof onPracticeQuestion === "function") {
        const payload = {
          id: practiceBtn.getAttribute("data-bank-practice"),
          category: practiceBtn.getAttribute("data-bank-category") || "",
          title: practiceBtn.getAttribute("data-bank-title") || "",
        };
        practiceBtn.disabled = true;
        Promise.resolve(onPracticeQuestion(payload))
          .catch(() => false)
          .then(() => {
            if (practiceBtn.isConnected) practiceBtn.disabled = false;
          });
      }
    });

    // Settings Add-Ons listeners
    el("addonsModulesList")?.addEventListener("change", (ev) => {
      const chk = ev.target.closest("[data-addon-module]");
      if (chk) {
        toggleModule(chk.getAttribute("data-addon-module"), Boolean(chk.checked));
      }
    });
    el("addonsPacksList")?.addEventListener("click", (ev) => {
      const inst = ev.target.closest("[data-addon-install]");
      if (inst) {
        installPack(inst.getAttribute("data-addon-install"));
        return;
      }
      const cancelBtn = ev.target.closest("[data-addon-cancel]");
      if (cancelBtn) {
        cancelPack(cancelBtn.getAttribute("data-addon-cancel"));
        return;
      }
      const uninst = ev.target.closest("[data-addon-uninstall]");
      if (uninst) {
        uninstallPack(uninst.getAttribute("data-addon-uninstall"));
      }
    });
  }

  bindEvents();

  return {
    applyStatusSnapshot,
    syncPackOpsFromAddons,
    // Re-evaluate the Home target-job line (e.g. after switching Home to practice mode).
    syncHomeJdTarget: () => {
      if (state.profileSummary) renderProfileSummaryIndicators(state.profileSummary);
    },
    openProfileModal,
    refreshProfile,
    refreshAddons,
    renderAddonsSection,
    installPack,
    cancelPack,
    getStateSnapshot,
    getProfile: () => local.profile,
    getAddons: () => local.addons,
  };
}
