#!/usr/bin/env bash
# Remove this app's per-user installation, preserving settings and model caches.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo "This uninstaller is for Ubuntu Linux." >&2
  exit 1
fi
if [[ $# != 0 ]]; then
  echo "Usage: $0" >&2
  exit 1
fi

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
if [[ "$DATA_HOME" != /* || "$DATA_HOME" == *$'\n'* || "$DATA_HOME" == *$'\r'* ]]; then
  echo "XDG_DATA_HOME must be an absolute path without line breaks." >&2
  exit 1
fi
APP_DIR="$DATA_HOME/video-transcriber"
APP_ID="com.videotranscriber.desktop"
ICON_ROOT="$DATA_HOME/icons/hicolor"
if [[ -L "$APP_DIR" ]] || [[ ! -f "$APP_DIR/.video-transcriber-install" || "$(< "$APP_DIR/.video-transcriber-install")" != "$APP_ID" ]]; then
  echo "No recognized Video Transcriber installation at: $APP_DIR" >&2
  exit 1
fi

rm -f -- "$DATA_HOME/applications/$APP_ID.desktop" "$ICON_ROOT/scalable/apps/$APP_ID.svg"
for icon in "$ICON_ROOT"/*/apps/"$APP_ID.png"; do
  [[ ! -f "$icon" ]] || rm -f -- "$icon"
done
rm -rf -- "$APP_DIR"
if command -v update-desktop-database >/dev/null 2>&1 && [[ -d "$DATA_HOME/applications" ]]; then
  update-desktop-database "$DATA_HOME/applications"
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1 && [[ -d "$ICON_ROOT" ]]; then
  gtk-update-icon-cache --force --ignore-theme-index "$ICON_ROOT" || true
fi
echo "Video Transcriber removed. Transcriptions, settings, and downloaded model caches were preserved."
