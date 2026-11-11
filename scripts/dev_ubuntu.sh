#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo "This launcher is for Ubuntu Linux. See README.md for Windows development." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_TRANSCRIBE:-$ROOT/.venv/bin/python}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Transcription Python is unavailable: $PYTHON" >&2
  echo "Run npm run setup:ubuntu, or set PYTHON_TRANSCRIBE to a Python with faster-whisper installed." >&2
  exit 1
fi
# Resolve PATH names and relative paths before changing to the repository root.
# Keep the venv entry point; resolving its symlink would bypass the venv.
PYTHON="$(command -v "$PYTHON")"
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"

if ! "$PYTHON" -c 'import faster_whisper, ctranslate2, onnxruntime, av'; then
  echo "Transcription dependencies are missing or broken in: $PYTHON" >&2
  echo "Run npm run setup:ubuntu, or install faster-whisper in your PYTHON_TRANSCRIBE environment." >&2
  exit 1
fi

for dependency in npm cargo rustc; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    echo "Missing command: $dependency. See the Ubuntu prerequisites in README.md." >&2
    exit 1
  fi
done

if [[ ! -x "$ROOT/node_modules/.bin/tauri" ]]; then
  echo "The Tauri CLI is missing. Run npm run setup:ubuntu (or npm ci)." >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No desktop display detected. Launch from an Ubuntu desktop session." >&2
  exit 1
fi

cd "$ROOT"
export PYTHON_TRANSCRIBE="$PYTHON"
exec npm run tauri -- dev "$@"
