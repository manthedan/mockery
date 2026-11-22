"""Headless Chrome check for the release site in site/.

    uv run python docs/marketing/verify_site.py --out-dir /tmp/mockery-site-shots

Serves site/ on 127.0.0.1, then for Desktop (1440x900) and Mobile (390x844),
in Dark and Light, walks every tab and view and asserts:

  * no console errors, uncaught exceptions or HTTP errors
  * no broken images, and every screenshot matches the active theme
  * no horizontal overflow
  * the theme switch and the copy buttons work
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import shutil
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
sys.path.insert(0, str(REPO / "scripts"))

import verify_ui as V  # noqa: E402

VIEWPORTS = {"desktop": (1440, 900, False), "mobile": (390, 844, True)}
INSTALL_CMD = "curl -fsSL https://raw.githubusercontent.com/manthedan/mockery/main/install.sh | sh"

WALK_TABS_JS = """(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const click = async (el) => { el.click(); await sleep(60); };
  const outer = [...document.querySelectorAll('.tabs [role="tab"]')];
  for (const tab of outer) {
    await click(tab);
    const panel = document.getElementById(tab.getAttribute('aria-controls'));
    for (const view of panel.querySelectorAll('.seg [role="tab"]')) await click(view);
    const first = panel.querySelector('.seg [role="tab"]');
    if (first) await click(first);
  }
  await click(outer[0]);
  for (const tab of document.querySelectorAll('.hero [role="tab"]')) await click(tab);
  await click(document.querySelector('.hero [role="tab"]'));
  return outer.length;
})()"""

LOAD_ALL_IMAGES_JS = """(async () => {
  const imgs = [...document.querySelectorAll('img')];
  imgs.forEach((img) => { img.loading = 'eager'; });
  await Promise.all(imgs.map((img) => img.decode().catch(() => null)));
  return imgs.length;
})()"""

AUDIT_JS = """((theme) => {
  const imgs = [...document.querySelectorAll('img')];
  const broken = imgs.filter((i) => !(i.complete && i.naturalWidth > 0)).map((i) => i.currentSrc || i.src);
  const shots = [...document.querySelectorAll('picture img')];
  const wrongTheme = shots.map((i) => i.currentSrc).filter((src) => !src.endsWith('_' + theme + '.webp'));
  const doc = document.documentElement;
  const overflowing = [...document.querySelectorAll('body *')]
    .filter((el) => !el.closest('.tabs-scroll') && el.getClientRects().length)
    .filter((el) => el.getBoundingClientRect().right > window.innerWidth + 1)
    .map((el) => el.tagName + '.' + el.className).slice(0, 6);
  return {
    images: imgs.length,
    broken,
    wrongTheme,
    scrollWidth: doc.scrollWidth,
    innerWidth: window.innerWidth,
    overflowX: doc.scrollWidth > window.innerWidth,
    overflowing,
    bg: getComputedStyle(document.body).backgroundColor,
    pressed: document.querySelector('[data-theme-choice][aria-pressed="true"]')?.dataset.themeChoice,
  };
})"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def serve_site() -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


async def run(out_dir: Path | None) -> int:
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    server, port = serve_site()
    origin = f"http://127.0.0.1:{port}"
    chrome_proc, chrome_profile, ws_url = await V.launch_chrome()
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            failures.append(name)

    try:
        async with V.CDPClient(ws_url, out_dir) as cdp:
            await cdp.send(
                "Browser.grantPermissions",
                {"origin": origin, "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"]},
            )
            for vp_name, (width, height, mobile) in VIEWPORTS.items():
                for theme in ("dark", "light"):
                    print(f"\n{vp_name} {width}x{height} · {theme}")
                    await cdp.send(
                        "Emulation.setDeviceMetricsOverride",
                        {"width": width, "height": height, "deviceScaleFactor": 2, "mobile": mobile},
                    )
                    await cdp.send(
                        "Emulation.setEmulatedMedia", {"features": [{"name": "prefers-color-scheme", "value": theme}]}
                    )
                    await cdp.navigate(f"{origin}/index.html")
                    await cdp.eval("localStorage.clear()")
                    await cdp.navigate(f"{origin}/index.html")
                    await asyncio.sleep(0.5)

                    # 1. System theme is honoured with no stored choice.
                    await cdp.eval(LOAD_ALL_IMAGES_JS)
                    audit = await cdp.eval(f"{AUDIT_JS}({theme!r})")
                    check(
                        "follows prefers-color-scheme",
                        audit["pressed"] == theme and not audit["wrongTheme"],
                        audit["bg"],
                    )

                    # 2. Every tab and view renders its real screenshot.
                    tabs = await cdp.eval(WALK_TABS_JS)
                    audit = await cdp.eval(f"{AUDIT_JS}({theme!r})")
                    check("zero broken images", not audit["broken"], f"{audit['images']} images, {tabs} showcase tabs")
                    check(
                        "zero horizontal overflow",
                        not audit["overflowX"] and not audit["overflowing"],
                        f"scrollWidth {audit['scrollWidth']} / viewport {audit['innerWidth']} {audit['overflowing']}",
                    )

                    if out_dir:
                        await cdp.screenshot(f"site_{vp_name}_{theme}_fold.png")
                        metrics = await cdp.send("Page.getLayoutMetrics")
                        full_h = int(metrics["cssContentSize"]["height"])
                        await cdp.send(
                            "Emulation.setDeviceMetricsOverride",
                            {"width": width, "height": full_h, "deviceScaleFactor": 1, "mobile": mobile},
                        )
                        await asyncio.sleep(0.4)
                        await cdp.screenshot(f"site_{vp_name}_{theme}_full.png")
                        await cdp.send(
                            "Emulation.setDeviceMetricsOverride",
                            {"width": width, "height": height, "deviceScaleFactor": 2, "mobile": mobile},
                        )

                    # 3. Theme switch swaps page and screenshots, then switches back.
                    other = "light" if theme == "dark" else "dark"
                    await cdp.eval(f"document.querySelector('[data-theme-choice=\"{other}\"]').click()")
                    await asyncio.sleep(1.0)
                    await cdp.eval(LOAD_ALL_IMAGES_JS)
                    swapped = await cdp.eval(f"{AUDIT_JS}({other!r})")
                    stored = await cdp.eval("localStorage.getItem('mockery-site-theme')")
                    check(
                        f"theme switch → {other}",
                        swapped["pressed"] == other
                        and not swapped["wrongTheme"]
                        and not swapped["broken"]
                        and stored == other
                        and swapped["bg"] != audit["bg"],
                        f"{audit['bg']} → {swapped['bg']}",
                    )
                    await cdp.eval(f"document.querySelector('[data-theme-choice=\"{theme}\"]').click()")
                    await asyncio.sleep(1.0)
                    back = await cdp.eval(f"{AUDIT_JS}({theme!r})")
                    stored = await cdp.eval("localStorage.getItem('mockery-site-theme')")
                    check(
                        f"theme switch → {theme} (back to system)",
                        back["pressed"] == theme and not back["wrongTheme"] and stored is None,
                    )

                    # 4. Copy buttons put the exact command on the clipboard.
                    await cdp.send("Page.bringToFront")
                    for selector in ('[data-copy="#heroCmd"]', '[data-copy="#installCmd"]'):
                        await cdp.eval("navigator.clipboard.writeText('')")
                        await cdp.eval(f"document.querySelector('{selector}').click()")
                        await asyncio.sleep(0.3)
                        label = await cdp.eval(f"document.querySelector('{selector}').textContent")
                        clip = await cdp.eval("navigator.clipboard.readText()")
                        check(f"copy {selector}", clip == INSTALL_CMD and label == "Copied", repr(label))

                    # 5. Keyboard: arrow keys move between showcase tabs.
                    await cdp.eval("document.getElementById('tabBehavioral').focus()")
                    await cdp.press_key("ArrowRight", "ArrowRight")
                    await asyncio.sleep(0.15)
                    active = await cdp.eval(
                        "document.querySelector('.tabs [aria-selected=\"true\"]').id + '|' + [...document.querySelectorAll('.panel.is-active')].map(p => p.id).join(',')"
                    )
                    check("arrow keys change showcase tab", active == "tabWhiteboard|panelWhiteboard", active)

                    check("zero console errors", not cdp.console_errors, cdp.console_errors)
                    check("zero page exceptions", not cdp.exceptions, cdp.exceptions)
                    check("zero HTTP errors", not cdp.http_errors, cdp.http_errors)
    finally:
        chrome_proc.terminate()
        server.shutdown()
        shutil.rmtree(chrome_profile, ignore_errors=True)

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Mockery release site in headless Chrome.")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Optional directory for above-the-fold and full-page screenshots."
    )
    return asyncio.run(run(parser.parse_args().out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
