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
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/scripts/build_macos_app.py" ]; then
  info "Building and installing $APP_DEST..."
  uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/scripts/build_macos_app.py" --install
else
  info "Creating $APP_DEST..."
  TMP_BUILD="$(mktemp -d)"
  trap 'rm -rf "$TMP_BUILD"' EXIT
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 "$REPO_URL" "$TMP_BUILD/mockery" >/dev/null 2>&1
    python3 "$TMP_BUILD/mockery/scripts/build_macos_app.py" --install || true
  fi
  # Zero-dependency fallback if python3/git didn't build the .app
  if [ ! -d "$APP_DEST" ]; then
    mkdir -p "$APP_DEST/Contents/MacOS" "$APP_DEST/Contents/Resources"
    cat >"$APP_DEST/Contents/MacOS/Mockery" <<'LAUNCHER'
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
    chmod +x "$APP_DEST/Contents/MacOS/Mockery"
    cat >"$APP_DEST/Contents/Info.plist" <<'PLIST'
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
  <string>0.2.0</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Mockery uses your microphone for local offline mock interview transcription.</string>
</dict>
</plist>
PLIST
  fi
fi

echo ""
bold "Mockery is installed!"
echo ""
info "Launch via Spotlight / Applications -> Mockery.app"
info "Or from Terminal: mockery"
info "Saved interviews live in: ~/Library/Application Support/Mockery/sessions"
echo ""
