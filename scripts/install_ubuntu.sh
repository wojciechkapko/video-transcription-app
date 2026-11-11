#!/usr/bin/env bash
# Install the built app and its own Python environment for the current user.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo "This installer is for Ubuntu Linux." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT/src-tauri/target/release/tauri-transcribe"
case "${1:-}" in
  --binary)
    if [[ $# != 2 ]]; then
      echo "Usage: $0 [--binary /path/to/built/tauri-transcribe]" >&2
      exit 1
    fi
    BINARY="$2"
    ;;
  --help|-h)
    echo "Usage: $0 [--binary /path/to/built/tauri-transcribe]"
    echo "Install for the current user after: npm run tauri -- build --no-bundle"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1. Use --help for usage." >&2; exit 1 ;;
esac

for dependency in python3 install mktemp ldd; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    echo "Missing command: $dependency. See README.md for Ubuntu prerequisites." >&2
    exit 1
  fi
done
if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
  echo "Python venv/pip support is missing. Install Ubuntu package python3-venv." >&2
  exit 1
fi
if [[ ! -x "$BINARY" ]]; then
  echo "Built executable not found: $BINARY" >&2
  echo "Run npm run tauri -- build --no-bundle first." >&2
  exit 1
fi
for resource in src-tauri/python/transcribe_worker.py src/assets/app-icon.svg src-tauri/icons/icon.png; do
  if [[ ! -f "$ROOT/$resource" ]]; then
    echo "Required resource is missing: $resource" >&2
    exit 1
  fi
done
if ! ldd "$BINARY" > /dev/null; then
  echo "Cannot inspect the app's native libraries. Build it for this Ubuntu system first." >&2
  exit 1
fi
if ldd "$BINARY" | grep -F 'not found'; then
  echo "Native runtime libraries are missing. See README.md for Ubuntu prerequisites." >&2
  exit 1
fi

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
if [[ "$DATA_HOME" != /* || "$DATA_HOME" == *$'\n'* || "$DATA_HOME" == *$'\r'* ]]; then
  echo "XDG_DATA_HOME must be an absolute path without line breaks." >&2
  exit 1
fi
APP_DIR="$DATA_HOME/video-transcriber"
APPLICATIONS_DIR="$DATA_HOME/applications"
ICON_ROOT="$DATA_HOME/icons/hicolor"
APP_ID="com.videotranscriber.desktop"
DESKTOP_FILE="$APPLICATIONS_DIR/$APP_ID.desktop"
PNG_SIZE="$(python3 - "$ROOT/src-tauri/icons/icon.png" <<'PYICON'
import struct
import sys
with open(sys.argv[1], "rb") as icon:
    header = icon.read(24)
if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
    raise SystemExit("App icon is not a valid PNG")
width, height = struct.unpack(">II", header[16:24])
if width != height or width < 16:
    raise SystemExit("App icon must be square and at least 16 pixels")
print(f"{width}x{height}")
PYICON
)"

if [[ -L "$APP_DIR" ]] || [[ -e "$APP_DIR" && (! -f "$APP_DIR/.video-transcriber-install" || "$(< "$APP_DIR/.video-transcriber-install")" != "$APP_ID") ]]; then
  echo "Refusing to overwrite an unrecognized directory: $APP_DIR" >&2
  exit 1
fi
mkdir -p "$APP_DIR/bin" "$APP_DIR/python" "$APPLICATIONS_DIR" \
  "$ICON_ROOT/scalable/apps" "$ICON_ROOT/$PNG_SIZE/apps"
printf '%s\n' "$APP_ID" > "$APP_DIR/.video-transcriber-install"

# Keep the environment at its final path: venv scripts contain absolute paths.
echo "==> preparing the installed Python environment"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/python" -m pip install 'faster-whisper==1.2.1'
"$APP_DIR/venv/bin/python" -c 'import faster_whisper, ctranslate2, onnxruntime, av; print("Python transcription dependencies OK")'

STAGING_DIR="$(mktemp -d "$APP_DIR/.install.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT
install -m 0755 "$BINARY" "$STAGING_DIR/video-transcriber"
install -m 0644 "$ROOT/src-tauri/python/transcribe_worker.py" "$STAGING_DIR/transcribe_worker.py"
cat > "$STAGING_DIR/run-video-transcriber" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_TRANSCRIBE="$APP_DIR/venv/bin/python"
export TRANSCRIBE_WORKER_PATH="$APP_DIR/python/transcribe_worker.py"
if [[ ! -x "$PYTHON_TRANSCRIBE" || ! -f "$TRANSCRIBE_WORKER_PATH" || ! -x "$APP_DIR/bin/video-transcriber" ]]; then
  echo "Video Transcriber installation is incomplete. Run its Ubuntu installer again." >&2
  exit 1
fi
if [[ "${1:-}" == --check ]]; then
  "$PYTHON_TRANSCRIBE" -c 'import faster_whisper, ctranslate2, onnxruntime, av; print("Python transcription dependencies OK")'
  "$PYTHON_TRANSCRIBE" "$TRANSCRIBE_WORKER_PATH" --probe
  exit
fi
exec "$APP_DIR/bin/video-transcriber" "$@"
LAUNCHER
chmod 0755 "$STAGING_DIR/run-video-transcriber"
bash -n "$STAGING_DIR/run-video-transcriber"
# Rename into place so an already-running app can finish using its old executable.
mv -f "$STAGING_DIR/video-transcriber" "$APP_DIR/bin/video-transcriber"
mv -f "$STAGING_DIR/transcribe_worker.py" "$APP_DIR/python/transcribe_worker.py"
mv -f "$STAGING_DIR/run-video-transcriber" "$APP_DIR/run-video-transcriber"
"$APP_DIR/run-video-transcriber" --check

# Desktop Exec is its own quoting format, with an additional string escape layer.
python3 - "$APP_DIR/run-video-transcriber" "$APP_ID" "$STAGING_DIR/app.desktop" <<'PYDESKTOP'
from pathlib import Path
import sys
launcher, app_id, output = sys.argv[1:]
def quote_exec(value):
    value = value.replace("%", "%%")
    value = "".join("\\" + c if c in '\\"`$' else c for c in value)
    value = value.replace("\\", "\\\\")
    return '"' + value + '"'
# GIO checks the executable before expanding %% field escapes. Use env for
# percent-containing paths so that check sees an ordinary executable name.
exec_prefix = "/usr/bin/env " if "%" in launcher else ""
Path(output).write_text(
    "[Desktop Entry]\n"
    "Version=1.0\n"
    "Type=Application\n"
    "Name=Video Transcriber\n"
    "Comment=Transcribe video and audio to text and subtitles\n"
    f"Exec={exec_prefix}{quote_exec(launcher)}\n"
    f"Icon={app_id}\n"
    "Terminal=false\n"
    "Categories=AudioVideo;Audio;\n"
    "Keywords=video;audio;transcription;whisper;subtitles;\n"
    "StartupWMClass=video-transcriber\n",
    encoding="utf-8",
)
PYDESKTOP
if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$STAGING_DIR/app.desktop"
fi
install -m 0644 "$ROOT/src/assets/app-icon.svg" "$ICON_ROOT/scalable/apps/$APP_ID.svg"
install -m 0644 "$ROOT/src-tauri/icons/icon.png" "$ICON_ROOT/$PNG_SIZE/apps/$APP_ID.png"
install -m 0644 "$STAGING_DIR/app.desktop" "$DESKTOP_FILE"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPLICATIONS_DIR"
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache --force --ignore-theme-index "$ICON_ROOT" || \
    echo "Icon cache could not be refreshed; log out and back in if the icon is not visible." >&2
fi
printf '\nInstalled. Find Video Transcriber in Ubuntu Applications.\n'
printf 'Launch from a terminal: %q\n' "$APP_DIR/run-video-transcriber"
echo "The first transcription downloads the ~3 GB Whisper model."
