/* Zero-dependency explicit client Finite State Machine primitives and coordinators.
 * Pure ES module — safe to import in both browser runtime and Node unit tests.
 */

export function formatApiDetail(detail, fallback = "Request failed") {
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (detail && typeof detail === "object") {
    const msg = detail.message || detail.remediation;
    if (typeof msg === "string" && msg.trim()) return msg.trim();
  }
  return fallback;
}

export function createMachineRegistry({ maxTransitions = 40 } = {}) {
  const machines = new Map();
  const activeOps = new Map();
  const recentTransitions = [];

  function recordTransition(entry) {
    recentTransitions.push({
      ts: Date.now(),
      ...entry,
    });
    if (recentTransitions.length > maxTransitions) {
      recentTransitions.splice(0, recentTransitions.length - maxTransitions);
    }
  }

  function register(machine) {
    machines.set(machine.name, machine);
  }

  function unregister(name) {
    machines.delete(name);
  }

  function beginOp(opId, meta = {}) {
    activeOps.set(opId, {
      id: opId,
      startedAt: Date.now(),
      ...meta,
    });
  }

  function endOp(opId) {
    activeOps.delete(opId);
  }

  function hasOp(opId) {
    return activeOps.has(opId);
  }

  function listOps() {
    return Array.from(activeOps.values()).map((op) => ({ ...op }));
  }

  function getRecentTransitions() {
    return recentTransitions.map((t) => ({ ...t }));
  }

  function snapshot() {
    const states = {};
    for (const [name, m] of machines.entries()) {
      states[name] = {
        state: m.state,
        allowed: m.allowedNext(),
      };
    }
    return {
      machines: states,
      activeOps: listOps(),
      recentTransitions: getRecentTransitions(),
    };
  }

  return {
    register,
    unregister,
    recordTransition,
    beginOp,
    endOp,
    hasOp,
    listOps,
    getRecentTransitions,
    snapshot,
  };
}

export function createMachine({
  name,
  initial,
  transitions,
  onTransition = null,
  registry = null,
  strict = false,
}) {
  let current = initial;
  const listeners = new Set();
  if (typeof onTransition === "function") {
    listeners.add(onTransition);
  }

  const normalized = {};
  for (const [k, v] of Object.entries(transitions || {})) {
    normalized[k] = v instanceof Set ? v : new Set(v || []);
  }

  const machine = {
    name,
    get state() {
      return current;
    },
    allowedNext() {
      return Array.from(normalized[current] || []);
    },
    can(next) {
      if (next === current) return true;
      return Boolean(normalized[current]?.has(next));
    },
    transition(next, { event = null, reason = null } = {}) {
      if (next === current) return false;
      const allowed = normalized[current];
      if (!allowed || !allowed.has(next)) {
        const msg = `[${name}] Illegal transition: ${current} -> ${next}${event ? ` (${event})` : ""}`;
        if (strict) {
          throw new Error(msg);
        }
        if (typeof console !== "undefined" && console.warn) {
          console.warn(msg);
        }
        return false;
      }
      const prev = current;
      current = next;
      registry?.recordTransition({
        machine: name,
        from: prev,
        to: next,
        event: event || `${prev}->${next}`,
        reason,
      });
      for (const fn of listeners) {
        try {
          fn(next, prev, { event, reason });
        } catch (err) {
          if (typeof console !== "undefined" && console.error) {
            console.error(`[${name}] Transition listener error:`, err);
          }
        }
      }
      return true;
    },
    reset(target = initial, { event = "reset", reason = null } = {}) {
      if (target === current) return false;
      const prev = current;
      current = target;
      registry?.recordTransition({
        machine: name,
        from: prev,
        to: target,
        event,
        reason,
      });
      for (const fn of listeners) {
        try {
          fn(target, prev, { event, reason });
        } catch (err) {
          if (typeof console !== "undefined" && console.error) {
            console.error(`[${name}] Reset listener error:`, err);
          }
        }
      }
      return true;
    },
    subscribe(fn) {
      if (typeof fn === "function") listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };

  registry?.register(machine);
  return machine;
}

/* ---------------- Canonical App Screen / Route Derivation ---------------- */

export function deriveAppRoute({
  practiceActive = false,
  captureState = "idle",
  forceWorkspace = false,
  hasCopilotContent = false,
}) {
  if (practiceActive) return "practice";
  const isCapturing = captureState !== "idle";
  if (hasCopilotContent || isCapturing || forceWorkspace) {
    return "copilot_workspace";
  }
  return "launchpad";
}

/* ---------------- Modal Open-Request Token Coordinator ---------------- */

export function createModalTokenController({ initial = "none", onChange = null, registry = null } = {}) {
  let active = initial;
  let token = 0;

  function emit(next, prev, event) {
    if (next !== prev) {
      registry?.recordTransition({
        machine: "modal",
        from: prev,
        to: next,
        event,
        reason: null,
      });
    }
    if (typeof onChange === "function") {
      onChange(next, prev);
    }
  }

  return {
    get state() {
      return active;
    },
    get currentToken() {
      return token;
    },
    beginRequest() {
      token += 1;
      return token;
    },
    isCurrent(reqToken) {
      return reqToken === token;
    },
    set(next, { reqToken = null, event = "set_modal" } = {}) {
      if (reqToken !== null && reqToken !== token) {
        return false;
      }
      if (reqToken === null) {
        // Explicit synchronous open/close invalidates any in-flight async modal open
        token += 1;
      }
      const prev = active;
      active = next;
      emit(next, prev, event);
      return true;
    },
    closeIf(expectedModal, expectedToken = null) {
      if (active !== expectedModal) return false;
      if (expectedToken !== null && expectedToken !== token) return false;
      token += 1;
      const prev = active;
      active = "none";
      emit("none", prev, `close_if_${expectedModal}`);
      return true;
    },
  };
}

/* ---------------- Practice Studio Client FSM Factories ---------------- */

export function createPracticeMicMachine(registry = null, onTransition = null) {
  return createMachine({
    name: "practice_mic",
    initial: "idle",
    registry,
    onTransition,
    transitions: {
      idle: ["requesting_mic"],
      requesting_mic: ["recording", "idle"],
      recording: ["transcribing", "idle"],
      transcribing: ["idle"],
    },
  });
}

export function createPracticeOpMachine(registry = null, onTransition = null) {
  const ops = [
    "starting",
    "opening",
    "submitting_turn",
    "requesting_hint",
    "advancing",
    "debriefing",
    "submitting_retry",
    "exiting",
  ];
  const transitions = { idle: ops };
  for (const op of ops) {
    // exiting is allowed to preempt or wait after other ops settle; every op returns to idle
    transitions[op] = op === "exiting" ? ["idle"] : ["idle", "exiting"];
  }
  return createMachine({
    name: "practice_op",
    initial: "idle",
    registry,
    onTransition,
    transitions,
  });
}

export function createRetryTimerMachine(registry = null, onTransition = null) {
  return createMachine({
    name: "practice_retry_timer",
    initial: "idle",
    registry,
    onTransition,
    transitions: {
      idle: ["running"],
      running: ["warning", "expired", "idle"],
      warning: ["expired", "idle"],
      expired: ["running", "idle"],
    },
  });
}

export function createWorkspaceSaveMachine(registry = null, onTransition = null) {
  return createMachine({
    name: "practice_workspace_save",
    initial: "saved",
    registry,
    onTransition,
    transitions: {
      saved: ["dirty", "saving"],
      dirty: ["saving", "saved"],
      saving: ["saved", "dirty", "error"],
      error: ["dirty", "saving", "saved"],
    },
  });
}

/* ---------------- Epoch-Guarded Resume Redaction Queue ---------------- */

export function createRedactionQueueCoordinator() {
  let epoch = 0;
  let queue = [];
  let draining = false;

  function invalidate() {
    epoch += 1;
    queue = [];
    return epoch;
  }

  function enqueue(op) {
    if (op.type === "prefs") {
      const last = queue[queue.length - 1];
      if (last && last.type === "prefs" && last.epoch === epoch) {
        last.payload = op.payload;
        return epoch;
      }
    }
    queue.push({ ...op, epoch });
    return epoch;
  }

  async function drain(executor) {
    if (draining) return;
    draining = true;
    try {
      while (queue.length > 0) {
        const op = queue.shift();
        if (op.epoch !== epoch) continue;
        await executor(op, () => op.epoch === epoch, () => queue.length === 0);
      }
    } finally {
      draining = false;
    }
  }

  return {
    get epoch() {
      return epoch;
    },
    get pendingCount() {
      return queue.length;
    },
    get busy() {
      return draining;
    },
    invalidate,
    enqueue,
    drain,
  };
}

/* ---------------- Shared Client Pack Operation Tracker ---------------- */

export function createPackOpTracker(registry = null) {
  const ops = new Map(); // packId -> "installing" | "uninstalling"
  const listeners = new Set();

  function notify(packId, op) {
    for (const fn of listeners) {
      try {
        fn(packId, op);
      } catch {
        /* ignore */
      }
    }
  }

  return {
    get(packId) {
      return ops.get(packId) || "idle";
    },
    isBusy(packId) {
      return ops.has(packId);
    },
    isInstalling(packId) {
      return ops.get(packId) === "installing";
    },
    isUninstalling(packId) {
      return ops.get(packId) === "uninstalling";
    },
    begin(packId, op) {
      if (ops.has(packId)) return false;
      ops.set(packId, op);
      registry?.beginOp(`pack:${packId}`, { packId, op });
      notify(packId, op);
      return true;
    },
    end(packId) {
      if (!ops.has(packId)) return;
      ops.delete(packId);
      registry?.endOp(`pack:${packId}`);
      notify(packId, "idle");
    },
    snapshot() {
      const out = {};
      for (const [k, v] of ops.entries()) out[k] = v;
      return out;
    },
    subscribe(fn) {
      if (typeof fn === "function") listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

/* ---------------- Canonical Unified Allowed Actions Derivation (AX) ---------------- */

export function deriveAppAllowedActions({
  route = "launchpad",
  modalState = "none",
  captureState = "idle",
  wsState = "disconnected",
  serverAllowedEvents = [],
  practice = null,
} = {}) {
  const actions = [];
  const add = (a) => {
    if (a && !actions.includes(a)) actions.push(a);
  };

  if (modalState && modalState !== "none") {
    add("close_modal");
    if (modalState === "settings") add("save_settings");
    if (modalState === "profile") add("manage_profile");
    return actions;
  }

  add("open_settings");
  add("open_profile");

  if (route === "launchpad") {
    const practiceEnabled = practice?.practiceEnabled !== false;
    const isPracticeHome = practiceEnabled && practice?.homeMode === "practice";
    const practiceBusy = practice?.opState && practice.opState !== "idle";
    if (captureState === "idle" && !practiceBusy) {
      if (isPracticeHome) {
        add("start_practice");
      } else {
        add("start_copilot_capture");
      }
    }
    if (practiceEnabled) {
      if (isPracticeHome) {
        add("switch_home_copilot");
      } else {
        add("switch_home_practice");
      }
    }
    return actions;
  }

  if (route === "copilot_workspace") {
    add("switch_right_tab");
    if (captureState === "idle") {
      add("go_home");
      if (wsState === "connected") add("start_copilot_capture");
    } else if (captureState === "streaming") {
      add("stop_copilot_capture");
    }
    if (Array.isArray(serverAllowedEvents)) {
      for (const evt of serverAllowedEvents) {
        if (typeof evt === "string" && evt.trim()) add(evt.trim());
      }
    }
    return actions;
  }

  if (route === "practice") {
    const opState = practice?.opState || "idle";
    const phase = practice?.phase || null;
    if (opState !== "exiting") {
      add("exit_practice");
    }
    if (phase === "active" && opState === "idle") {
      add("submit_practice_turn");
      add("request_practice_hint");
      add("advance_practice_question");
      add("finish_practice_debrief");
      if (practice?.micState === "idle") add("start_practice_mic");
      if (practice?.micState === "recording") add("stop_practice_mic");
    } else if (phase === "awaiting_debrief" && opState === "idle") {
      add("finish_practice_debrief");
    }
    if (practice?.retryTimerState === "running" || practice?.retryTimerState === "warning") {
      add("cancel_practice_retry");
    }
    if (practice?.workspaceSaveState === "error") {
      add("retry_workspace_save");
    }
    if (phase === "completed" && opState === "idle") {
      add("restart_practice");
    }
    const studioActions = practice?.mountedStudio?.allowedActions;
    if (Array.isArray(studioActions)) {
      for (const a of studioActions) {
        if (typeof a === "string" && a.trim()) add(`studio:${a.trim()}`);
      }
    }
    return actions;
  }

  return actions;
}
