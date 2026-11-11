#!/usr/bin/env bash
# Prepare repository-local dependencies. Install OS packages and Rust yourself.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo "This setup script is for Ubuntu Linux. See README.md for Windows setup." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
missing=0
for dependency in node npm python3 cargo rustc cc pkg-config; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    echo "Missing command: $dependency" >&2
    missing=1
  fi
done

if command -v python3 >/dev/null 2>&1 && ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
  echo "Python venv/pip support is missing (Ubuntu package: python3-venv)." >&2
  missing=1
fi

if command -v pkg-config >/dev/null 2>&1; then
  for library in webkit2gtk-4.1 gtk+-3.0 openssl ayatana-appindicator3-0.1 librsvg-2.0; do
    if ! pkg-config --exists "$library"; then
      echo "Missing development library: $library" >&2
      missing=1
    fi
  done
fi

if (( missing )); then
  cat >&2 <<'EOF'
Install the Ubuntu prerequisites, then rerun this script:
  sudo apt update
  sudo apt install build-essential pkg-config libwebkit2gtk-4.1-dev libssl-dev \
    libayatana-appindicator3-dev librsvg2-dev libxdo-dev curl wget file \
    python3-venv xdg-utils
Install Node.js (with npm) and Rust using the instructions in README.md.
After installing Rust, load its environment: source "$HOME/.cargo/env"
No system packages or global toolchains are installed by this script.
EOF
  exit 1
fi

cd "$ROOT"
echo "==> preparing .venv"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install faster-whisper

echo "==> checking transcription dependencies"
"$ROOT/.venv/bin/python" -c 'import faster_whisper, ctranslate2, onnxruntime, av; print("Python transcription dependencies OK")'

echo "==> installing locked Node dependencies"
npm ci

echo "==> ready; launch with: npm run dev:ubuntu"
echo "Choose CPU for Ubuntu development. The first transcription downloads the ~3 GB model."
