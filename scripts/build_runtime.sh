#!/usr/bin/env bash
# Regenerate the bundled portable Python runtime (CPU-only faster-whisper).
#
# Produces src-tauri/runtime/ (gitignored). Tauri bundles it via
# `bundle.resources: ["runtime/**/*"]`, and lib.rs resolves
# `resources/runtime/python.exe` at runtime — so the shipped .exe is
# self-contained: no Python install required on the target machine.
#
# Requirements: `curl`, `unzip`, and a `python` on PATH whose minor version
# MATCHES PY_VER (so the pip-installed wheels' cpXY tags line up with the
# embeddable interpreter). The bundled runtime ships no NVIDIA DLLs — it is
# CPU-only as built; GPU users either use the in-app "Install GPU" flow
# (worker --install-cuda) or point at their own env via PYTHON_TRANSCRIBE.
set -euo pipefail

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *)
    echo "This script builds the Windows Python runtime and must run on Windows (Git Bash/MSYS2)." >&2
    echo "On Ubuntu, use: npm run setup:ubuntu" >&2
    exit 1
    ;;
esac

PY_VER="3.11.9"
PY_VER_SHORT="311"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/src-tauri/runtime"
PIP_PYTHON="${PYTHON_TRANSCRIBE:-python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The build interpreter must match the embeddable minor version (cpXY wheels).
"$PIP_PYTHON" -c "import sys; v=sys.version_info[:2]; assert v==(3,11), f'need Python 3.11 build interpreter, got {v}'"

echo "==> cleaning old runtime"
rm -rf "$DEST"
mkdir -p "$DEST"

echo "==> downloading Python ${PY_VER} embeddable package"
curl -sSL -o "$TMP/python.zip" \
  "https://www.python.org/ftp/python/${PY_VER}/python-${PY_VER}-embed-amd64.zip"
unzip -qo "$TMP/python.zip" -d "$DEST"

echo "==> enabling site-packages in the embeddable interpreter"
cat > "$DEST/python${PY_VER_SHORT}._pth" <<'EOF'
python311.zip
.
Lib\site-packages

import site
EOF

echo "==> installing faster-whisper (CPU wheels) into the runtime"
"$PIP_PYTHON" -m pip install --target "$DEST/Lib/site-packages" \
  --only-binary=:all: --no-cache-dir --upgrade faster-whisper

echo "==> verifying the runtime imports cleanly"
"$DEST/python.exe" -c "import faster_whisper, ctranslate2, onnxruntime, av; print('runtime OK:', faster_whisper.__version__ if hasattr(faster_whisper, '__version__') else 'n/a')"

echo "==> done. runtime at: $DEST"
du -sh "$DEST"
