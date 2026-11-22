/* Mockery release site: theme switch, tabs, copy-to-clipboard. No dependencies. */
(() => {
  "use strict";

  const root = document.documentElement;
  const THEME_KEY = "mockery-site-theme";
  const lightQuery = window.matchMedia("(prefers-color-scheme: light)");
  const SYSTEM_LIGHT_MEDIA = "(prefers-color-scheme: light)";

  // ── Theme ────────────────────────────────────────────────────────────────
  // No stored choice means "follow the system". Picking the theme the system
  // already uses clears the override, so the page keeps following it.

  const systemTheme = () => (lightQuery.matches ? "light" : "dark");

  function explicitTheme() {
    const value = root.getAttribute("data-theme");
    return value === "light" || value === "dark" ? value : null;
  }

  const activeTheme = () => explicitTheme() || systemTheme();

  function shotUrl(picture, theme) {
    const source = picture.querySelector("source[data-theme-source]");
    const img = picture.querySelector("img");
    return theme === "light" ? source.getAttribute("srcset") : img.getAttribute("src");
  }

  function isShown(el) {
    return el.getClientRects().length > 0;
  }

  function preload(url) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = img.onerror = resolve;
      img.src = url;
    });
  }

  function preloadShownShots(theme) {
    const urls = [...document.querySelectorAll("picture")].filter(isShown).map((p) => shotUrl(p, theme));
    return Promise.all([...new Set(urls)].map(preload));
  }

  function syncThemeUi() {
    const theme = activeTheme();
    const explicit = explicitTheme();
    // Real screenshots follow the page: an explicit choice overrides the media query on each <source>.
    const media = explicit ? (explicit === "light" ? "all" : "not all") : SYSTEM_LIGHT_MEDIA;
    document.querySelectorAll("source[data-theme-source]").forEach((source) => {
      if (source.media !== media) source.media = media;
    });
    document.querySelectorAll("[data-theme-choice]").forEach((btn) => {
      btn.setAttribute("aria-pressed", String(btn.dataset.themeChoice === theme));
    });
    document.querySelectorAll("a[data-shot-link]").forEach((link) => {
      const picture = link.querySelector("picture");
      if (picture) link.setAttribute("href", shotUrl(picture, theme));
    });
  }

  let themeRequest = 0;

  async function chooseTheme(theme) {
    if (theme === activeTheme()) return;
    const request = ++themeRequest;
    // Fetch the other theme's screenshots first so page and screens change together.
    await Promise.race([preloadShownShots(theme), new Promise((r) => setTimeout(r, 1200))]);
    if (request !== themeRequest) return;
    const followSystem = theme === systemTheme();
    if (followSystem) root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (followSystem) localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, theme);
    } catch (_) {
      /* Private mode: the choice simply lasts for this visit. */
    }
    syncThemeUi();
  }

  document.querySelectorAll("[data-theme-choice]").forEach((btn) => {
    btn.addEventListener("click", () => chooseTheme(btn.dataset.themeChoice));
  });
  lightQuery.addEventListener("change", syncThemeUi);
  syncThemeUi();

  const otherTheme = () => (activeTheme() === "light" ? "dark" : "light");
  const warmOtherTheme = () => preloadShownShots(otherTheme());
  if ("requestIdleCallback" in window) window.addEventListener("load", () => requestIdleCallback(warmOtherTheme));
  else window.addEventListener("load", () => setTimeout(warmOtherTheme, 600));

  // ── Tabs (hero mode switch, studio showcase, per-panel views) ────────────

  function initTablist(tablist) {
    const tabs = [...tablist.querySelectorAll(':scope > [role="tab"]')];
    const panelOf = (tab) => document.getElementById(tab.getAttribute("aria-controls"));
    const note = tablist.parentElement.querySelector("[data-stage-note]");

    function select(tab, { focus = false } = {}) {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        panelOf(t).classList.toggle("is-active", on);
      });
      if (note) note.textContent = panelOf(tab).dataset.note || "";
      if (focus) {
        tab.focus();
        tab.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    }

    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (ev) => {
        const step = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
        let next = null;
        if (step) next = tabs[(i + step + tabs.length) % tabs.length];
        else if (ev.key === "Home") next = tabs[0];
        else if (ev.key === "End") next = tabs[tabs.length - 1];
        if (!next) return;
        ev.preventDefault();
        select(next, { focus: true });
      });
    });
  }

  document.querySelectorAll('[role="tablist"]').forEach(initTablist);

  // ── Copy to clipboard ────────────────────────────────────────────────────

  const live = document.querySelector("[data-live]");

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_) {
        /* Fall through to the selection-based copy below. */
      }
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.cssText = "position:fixed;top:0;left:0;opacity:0;";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (_) {
      ok = false;
    }
    area.remove();
    return ok;
  }

  document.querySelectorAll("button[data-copy]").forEach((btn) => {
    const idleLabel = btn.textContent;
    let resetTimer = 0;
    btn.addEventListener("click", async () => {
      const target = document.querySelector(btn.dataset.copy);
      const text = target.querySelector("[data-copy-text]").textContent.trim();
      const ok = await copyText(text);
      btn.textContent = ok ? "Copied" : "Press ⌘C";
      btn.dataset.copyState = ok ? "copied" : "failed";
      if (!ok) {
        const range = document.createRange();
        range.selectNodeContents(target.querySelector("[data-copy-text]"));
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
      if (live) live.textContent = ok ? "Command copied to clipboard." : "Command selected. Press Command-C to copy.";
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => {
        btn.textContent = idleLabel;
        delete btn.dataset.copyState;
        if (live) live.textContent = "";
      }, 1800);
    });
  });
})();
