#!/usr/bin/env bash
# Mockery one-line installer for Apple Silicon macOS
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/manthedan/mockery/main/install.sh | sh
set -euo pipefail

REPO_URL="${MOCKERY_REPO_URL:-https://github.com/manthedan/mockery.git}"
APP_DEST="/Applications/Mockery.app"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "  • %s\n" "$*"; }
warn() { printf "\033[33m  ! %s\033[0m\n" "$*"; }

echo ""
bold "Installing Mockery — Local Apple Silicon Mock Interview Studio"
echo ""

# 1. Platform check
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [ "$OS_NAME" != "Darwin" ]; then
  warn "Mockery's local Parakeet MLX speech engine requires macOS on Apple Silicon (detected: $OS_NAME $ARCH_NAME)."
  exit 1
fi
if [ "$ARCH_NAME" != "arm64" ]; then
  warn "Apple Silicon (M1/M2/M3/M4) is required for local MLX speech recognition (detected: $ARCH_NAME)."
  exit 1
fi

# 2. Ensure uv is installed
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (fast Python runtime manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 3. Install Mockery CLI + isolated Python 3.12 runtime
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  info "Installing Mockery from local checkout ($SCRIPT_DIR)..."
  uv tool install --force --python 3.12 "$SCRIPT_DIR"
else
  info "Installing Mockery from $REPO_URL..."
  uv tool install --force --python 3.12 "git+${REPO_URL}"
fi

# 4. Install double-clickable /Applications/Mockery.app
APP_BUILT=0
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/scripts/build_macos_app.py" ]; then
  info "Building and installing $APP_DEST..."
  if uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/scripts/build_macos_app.py" --install; then
    APP_BUILT=1
  fi
else
  info "Creating $APP_DEST..."
  TMP_BUILD="$(mktemp -d)"
  trap 'rm -rf "$TMP_BUILD"' EXIT
  # Only run git clone when real git is present (avoid triggering macOS Xcode CLT stub popup)
  if command -v git >/dev/null 2>&1 && ( [ -x /opt/homebrew/bin/git ] || [ -x /usr/local/bin/git ] || xcode-select -p >/dev/null 2>&1 ); then
    if git clone --depth 1 "$REPO_URL" "$TMP_BUILD/mockery" >/dev/null 2>&1; then
      UV_PY="$(uv python find 3.12 2>/dev/null || true)"
      if [ -n "$UV_PY" ] && "$UV_PY" "$TMP_BUILD/mockery/scripts/build_macos_app.py" --install; then
        APP_BUILT=1
      elif uv run --python 3.12 python "$TMP_BUILD/mockery/scripts/build_macos_app.py" --install; then
        APP_BUILT=1
      else
        warn "Could not compile native menu bar app; falling back to shell launcher..."
      fi
    fi
  fi
  # Zero-dependency fallback when git/Swift build did not produce the .app
  if [ "$APP_BUILT" -ne 1 ]; then
    STAGING_APP="/Applications/.Mockery.app.staging"
    rm -rf "$STAGING_APP"
    mkdir -p "$STAGING_APP/Contents/MacOS" "$STAGING_APP/Contents/Resources"
    cat >"$STAGING_APP/Contents/MacOS/Mockery" <<'LAUNCHER'
#!/bin/bash
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
DATA_DIR="$HOME/Library/Application Support/Mockery"
LOG_DIR="$HOME/Library/Logs/Mockery"
mkdir -p "$DATA_DIR" "$LOG_DIR"
export MOCKERY_HOME="$DATA_DIR"
if curl -fsS http://127.0.0.1:8765/api/machines >/dev/null 2>&1; then
  open "http://127.0.0.1:8765"
  exit 0
fi
nohup mockery --host 127.0.0.1 --port 8765 --data-dir "$DATA_DIR" >>"$LOG_DIR/server.log" 2>&1 &
LAUNCHER
    chmod +x "$STAGING_APP/Contents/MacOS/Mockery"
    cat >"$STAGING_APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key>
  <string>Mockery</string>
  <key>CFBundleExecutable</key>
  <string>Mockery</string>
  <key>CFBundleIdentifier</key>
  <string>com.manthedan.mockery</string>
  <key>CFBundleName</key>
  <string>Mockery</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.2.3</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Mockery uses your microphone for local offline mock interview transcription.</string>
</dict>
</plist>
PLIST
    rm -rf "$APP_DEST"
    mv "$STAGING_APP" "$APP_DEST"
  fi
fi

if [ ! -x "$APP_DEST/Contents/MacOS/Mockery" ]; then
  warn "Failed to install a valid $APP_DEST launcher."
  exit 1
fi

echo ""
bold "Mockery is installed!"
echo ""
info "Launch via Spotlight / Applications -> Mockery.app"
info "Or from Terminal: mockery"
info "Saved interviews live in: ~/Library/Application Support/Mockery/sessions"
echo ""
