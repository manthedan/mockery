#!/usr/bin/env python3
"""
Build a double-clickable macOS `Mockery.app` (and optional `.dmg`) for Apple Silicon.

Features:
  - Compiles `macos/MockeryLauncher.swift` into a native AppKit menu bar binary when `swiftc` is available
    (falls back to a zero-dependency shell launcher if Xcode CLT is not installed).
  - Generates a crisp Ink & Paper `.icns` app icon (no external image libraries required).
  - Writes a clean `Info.plist` and ad-hoc signs the bundle for local macOS Gatekeeper trust.

Usage:
  uv run python scripts/build_macos_app.py              # builds dist/Mockery.app
  uv run python scripts/build_macos_app.py --install    # installs into /Applications/Mockery.app
  uv run python scripts/build_macos_app.py --dmg        # also packages dist/Mockery-macOS-arm64.dmg
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import tempfile
import tomllib
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWIFT_SRC = ROOT / "macos" / "MockeryLauncher.swift"
DIST_DIR = ROOT / "dist"


def _project_version() -> str:
    try:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        version = str((data.get("project") or {}).get("version") or "").strip()
        if version:
            return version
    except Exception:
        pass
    return "0.2.4"


def _write_png_rgba(path: Path, width: int, height: int, rgba: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        raw.extend(rgba[y * stride : (y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _render_ink_and_paper_icon_png(size: int = 512) -> bytes:
    """Render Mockery's Ink & Paper rounded-square icon (warm charcoal #1C1B18 + parchment #FBFAF7 center dot + subtle ring)."""
    bg = (0x1C, 0x1B, 0x18)
    fg = (0xFB, 0xFA, 0xF7)
    ring = (0xD9, 0x77, 0x57)  # subtle editorial terracotta accent ring

    half = size / 2.0
    corner_r = size * 0.22
    inner_half = half - corner_r
    dot_r = size * 0.165
    ring_inner = size * 0.27
    ring_outer = size * 0.30

    out = bytearray(size * size * 4)
    idx = 0
    for y in range(size):
        dy = y + 0.5 - half
        ay = abs(dy)
        for x in range(size):
            dx = x + 0.5 - half
            ax = abs(dx)
            # Rounded-rect SDF for macOS squircle silhouette
            qx = max(ax - inner_half, 0.0)
            qy = max(ay - inner_half, 0.0)
            dist_rect = (qx * qx + qy * qy) ** 0.5 - corner_r
            alpha_shape = max(0.0, min(1.0, 0.5 - dist_rect))
            if alpha_shape <= 0.0:
                idx += 4
                continue

            r_dist = (dx * dx + dy * dy) ** 0.5
            # Center parchment dot
            dot_mix = max(0.0, min(1.0, dot_r - r_dist + 0.6))
            # Subtle warm ring around center
            ring_mix = max(
                0.0,
                min(1.0, min(r_dist - ring_inner + 0.6, ring_outer - r_dist + 0.6)),
            ) * 0.85

            cr = bg[0] * (1.0 - ring_mix) + ring[0] * ring_mix
            cg = bg[1] * (1.0 - ring_mix) + ring[1] * ring_mix
            cb = bg[2] * (1.0 - ring_mix) + ring[2] * ring_mix

            cr = cr * (1.0 - dot_mix) + fg[0] * dot_mix
            cg = cg * (1.0 - dot_mix) + fg[1] * dot_mix
            cb = cb * (1.0 - dot_mix) + fg[2] * dot_mix

            out[idx] = int(round(cr))
            out[idx + 1] = int(round(cg))
            out[idx + 2] = int(round(cb))
            out[idx + 3] = int(round(alpha_shape * 255))
            idx += 4
    return bytes(out)


def build_icns(dest_icns: Path) -> bool:
    if not (shutil.which("sips") and shutil.which("iconutil")):
        return False
    with tempfile.TemporaryDirectory(prefix="mockery_icon_") as td:
        td_path = Path(td)
        base_png = td_path / "base_512.png"
        _write_png_rgba(base_png, 512, 512, _render_ink_and_paper_icon_png(512))

        iconset = td_path / "Mockery.iconset"
        iconset.mkdir()
        sizes = [
            (16, "icon_16x16.png"),
            (32, "icon_16x16@2x.png"),
            (32, "icon_32x32.png"),
            (64, "icon_32x32@2x.png"),
            (128, "icon_128x128.png"),
            (256, "icon_128x128@2x.png"),
            (256, "icon_256x256.png"),
            (512, "icon_256x256@2x.png"),
            (512, "icon_512x512.png"),
        ]
        for px, fname in sizes:
            subprocess.run(
                ["sips", "-z", str(px), str(px), str(base_png), "--out", str(iconset / fname)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(dest_icns)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return dest_icns.exists()


SHELL_FALLBACK_LAUNCHER = """#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
DATA_DIR="$HOME/Library/Application Support/Mockery"
LOG_DIR="$HOME/Library/Logs/Mockery"
mkdir -p "$DATA_DIR" "$LOG_DIR"
export MOCKERY_HOME="$DATA_DIR"

if curl -fsS http://127.0.0.1:8765/api/machines >/dev/null 2>&1; then
  open "http://127.0.0.1:8765"
  exit 0
fi

if command -v mockery >/dev/null 2>&1; then
  nohup mockery --host 127.0.0.1 --port 8765 --data-dir "$DATA_DIR" >>"$LOG_DIR/server.log" 2>&1 &
elif command -v uv >/dev/null 2>&1; then
  nohup uv tool run --from git+https://github.com/manthedan/mockery.git mockery --host 127.0.0.1 --port 8765 --data-dir "$DATA_DIR" >>"$LOG_DIR/server.log" 2>&1 &
else
  osascript -e 'display dialog "Mockery is not installed yet. Run the one-line installer in Terminal from github.com/manthedan/mockery." buttons {"OK"} default button 1'
  exit 1
fi
"""


def _has_real_swiftc() -> bool:
    """Return True only when a working Swift compiler is installed (not the macOS CLT shim)."""
    swiftc = shutil.which("swiftc")
    if not swiftc:
        return False
    if swiftc != "/usr/bin/swiftc":
        return True
    res = subprocess.run(
        ["xcode-select", "-p"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def build_app(*, force_shell: bool = False) -> Path:
    app_dir = DIST_DIR / "Mockery.app"
    contents = app_dir / "Contents"
    macos_dir = contents / "MacOS"
    res_dir = contents / "Resources"

    if app_dir.exists():
        shutil.rmtree(app_dir)
    macos_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    exe_path = macos_dir / "Mockery"
    used_swift = False
    if not force_shell and SWIFT_SRC.exists() and _has_real_swiftc():
        print("Compiling native Swift menu bar launcher...")
        subprocess.run(
            [
                "swiftc",
                "-O",
                "-framework",
                "AppKit",
                str(SWIFT_SRC),
                "-o",
                str(exe_path),
            ],
            check=True,
        )
        used_swift = True
    else:
        print("Using zero-dependency shell launcher fallback...")
        exe_path.write_text(SHELL_FALLBACK_LAUNCHER, encoding="utf-8")
        os.chmod(exe_path, 0o755)

    has_icon = build_icns(res_dir / "AppIcon.icns")

    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>Mockery</string>
  <key>CFBundleExecutable</key>
  <string>Mockery</string>
  <key>CFBundleIdentifier</key>
  <string>com.manthedan.mockery</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Mockery</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>{_project_version()}</string>
  <key>CFBundleVersion</key>
  <string>6</string>
  {"<key>CFBundleIconFile</key><string>AppIcon</string>" if has_icon else ""}
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>LSUIElement</key>
  <{"true" if used_swift else "false"}/>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Mockery uses your microphone for local offline mock interview transcription.</string>
</dict>
</plist>
"""
    (contents / "Info.plist").write_text(info_plist, encoding="utf-8")

    if shutil.which("codesign"):
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(app_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(f"Built {app_dir} ({'native Swift menu bar' if used_swift else 'shell launcher'})")
    return app_dir


def build_dmg(app_dir: Path) -> Path:
    dmg_path = DIST_DIR / "Mockery-macOS-arm64.dmg"
    if dmg_path.exists():
        dmg_path.unlink()
    with tempfile.TemporaryDirectory(prefix="mockery_dmg_") as td:
        stage = Path(td)
        shutil.copytree(app_dir, stage / "Mockery.app", symlinks=True)
        os.symlink("/Applications", stage / "Applications")
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                "Mockery",
                "-srcfolder",
                str(stage),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            check=True,
        )
    print(f"Created {dmg_path}")
    return dmg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mockery.app for macOS Apple Silicon")
    parser.add_argument("--install", action="store_true", help="Copy built Mockery.app into /Applications")
    parser.add_argument("--dmg", action="store_true", help="Also build dist/Mockery-macOS-arm64.dmg")
    parser.add_argument("--shell-fallback", action="store_true", help="Force shell launcher instead of Swift")
    args = parser.parse_args()

    app_dir = build_app(force_shell=args.shell_fallback)
    if args.dmg:
        build_dmg(app_dir)
    if args.install:
        dest = Path("/Applications/Mockery.app")
        staging = dest.parent / ".Mockery.app.staging"
        backup = dest.parent / ".Mockery.app.backup"
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(app_dir, staging, symlinks=True)
        staged_exe = staging / "Contents" / "MacOS" / "Mockery"
        if not staged_exe.is_file() or not os.access(staged_exe, os.X_OK):
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"Staged app bundle missing executable launcher: {staged_exe}")
        try:
            if dest.exists():
                dest.rename(backup)
            staging.rename(dest)
        except Exception:
            if not dest.exists() and backup.exists():
                backup.rename(dest)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        print(f"Installed to {dest}")


if __name__ == "__main__":
    main()
