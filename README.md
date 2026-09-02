# Video Transcriber

A local-first desktop app that transcribes video and audio files to text — entirely on your own machine, nothing leaves it. Built with [Tauri 2](https://tauri.app/) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper `large-v3`).

Drag in a file, pick a language, and get three output formats with word-level timestamps, running on your GPU when available and falling back to CPU otherwise.

## Features

- **Drop a video or audio file** (MP4, MKV, MOV, WebM, AVI, MP3, WAV, FLAC, …) or browse for one.
- **Three outputs at once** — plain text (`.txt`), SubRip subtitles (`.srt`), and WebVTT (`.vtt`) — written next to the source file.
- **Whisper `large-v3`** with beam search, word-level timestamps, and silence filtered out (VAD), with repetition-collapse guards.
- **Language auto-detection** — or pin one of 8 languages (English, Polish, German, Spanish, French, Italian, Dutch, Portuguese).
- **Device selection** — *Auto (GPU → CPU)*, *CPU*, or *GPU (CUDA)*, remembered between launches.
- **On-demand GPU support** — if CUDA isn't present, the app offers a one-click *Install GPU support* that downloads the NVIDIA libraries (~1.1 GB) and unlocks GPU acceleration.
- **First-run model download with live progress** — the ~3 GB `large-v3` model is fetched once and cached.
- **Cancel at any time** — stop a run mid-way with no leftover or corrupt files (outputs are written atomically).
- **Live progress** — segment count, elapsed/remaining time, language, and a progress bar.

## How it works

The app is a thin Tauri shell around a Python worker:

```
┌─────────────────────────────────────────────────────┐
│  Tauri 2 (Rust)  —  src-tauri/src/lib.rs            │
│    • spawns the Python worker                        │
│    • streams worker stdout as progress events        │
├─────────────────────────────────────────────────────┤
│  Python worker  —  src-tauri/python/transcribe_worker.py │
│    • loads faster-whisper (large-v3)                  │
│    • decodes audio → segments                        │
│    • writes .txt / .srt / .vtt atomically            │
└─────────────────────────────────────────────────────┘
```

The worker prints JSON-lines progress to stdout (`loading_model`, `transcribing`, `progress`, `done`, `download_progress`, …), which the Rust side relays to the vanilla-JS frontend as Tauri events.

**Model**: the ~3 GB `large-v3` model downloads from Hugging Face on first run (with a progress bar) into `~\.cache\whisper-transcribe\large-v3`, or is reused from the Hugging Face cache if already present.

**GPU (CUDA)**: the shipped bundle is CPU-only for size and portability. If you pick *GPU* and the needed NVIDIA libraries (cuBLAS/cuDNN/cudart) aren't found, the app offers to download and install them into `%LOCALAPPDATA%\whisper-transcribe\cuda`. Alternatively, point `PYTHON_TRANSCRIBE` at an existing Python with `faster-whisper` + CUDA wheels installed.

## Requirements

- **Windows 10/11** (x64). The bundled runtime and CUDA bootstrap are Windows-targeted.
- To run from source: [Node.js](https://nodejs.org/) 18+, the [Rust toolchain](https://rustup.rs/), and the [Tauri pre-requisites](https://tauri.app/start/prerequisites/) (WebView2, MSVC build tools).
- To build the portable runtime: `curl`, `unzip`, and a Python 3.11 on `PATH`.

An NVIDIA GPU is optional — everything runs on CPU, just slower.

## Development

```bash
npm install
npm run tauri dev
```

`tauri dev` runs the app directly against the source; in debug builds the Python interpreter is resolved from `%LOCALAPPDATA%\hermes\hermes-agent\venv` (and can be overridden, see below).

## Building a standalone `.exe`

The shipped `.exe` bundles a portable, self-contained Python runtime so end users don't install Python.

1. **Build the portable runtime** (downloads the Python 3.11 embeddable package + `faster-whisper` CPU wheels into `src-tauri/runtime/`, which is git-ignored):

   ```bash
   ./scripts/build_runtime.sh
   ```

2. **Compile the app**:

   ```bash
   npm run tauri build            # produce installers (NSIS/MSI)
   npm run tauri build -- --no-bundle   # just the .exe
   ```

## Project structure

```
src/                       # vanilla JS frontend (no bundler)
  index.html  main.js  styles.css
src-tauri/
  src/lib.rs               # Tauri commands: transcribe, cancel, check_gpu,
                           #   install_cuda, settings, read/open
  python/transcribe_worker.py   # the worker (see --help / modes below)
  runtime/                 # portable Python runtime (generated, git-ignored)
  tauri.conf.json          # bundles python/ + runtime/ as resources
scripts/build_runtime.sh   # builds the portable runtime
```

## Worker CLI

The worker also exposes standalone modes (useful for testing):

```bash
python transcribe_worker.py <input> <output_prefix> [language]   # transcribe
python transcribe_worker.py --probe        # print device availability (JSON)
python transcribe_worker.py --install-cuda # download + install CUDA libs
```

## Configuration

Settings (device + language) are saved to `%LOCALAPPDATA%\whisper-transcribe\settings.json`.

Environment variables:

| Variable | Purpose |
| --- | --- |
| `PYTHON_TRANSCRIBE` | Path to a Python with `faster-whisper` installed (overrides the bundled/dev interpreter). Use this to point at a CUDA-equipped Python. |
| `TRANSCRIBE_DEVICE` | Force `cpu` or `cuda` (the UI's device selector sets this per run). |
| `TRANSCRIBE_COMPUTE_TYPE` | Override the model precision (`int8_float16` on GPU, `int8` on CPU by default). |
| `TRANSCRIBE_WORKER_PATH` | Override the location of `transcribe_worker.py`. |
| `WHISPER_MODEL_DIR` | Where to look for / download the model. |
| `HF_HOME` / `HF_TOKEN` | Hugging Face cache location / access token (for gated repos or higher rate limits). |

## Notes

- The GPU‑enabling download installs NVIDIA libraries into your user profile (`%LOCALAPPDATA%\whisper-transcribe\cuda`), not system-wide, and is opt-in.
- Output files are written to `.txt.part`/`.srt.part`/`.vtt.part` temp files and atomically renamed on success, so a cancelled or crashed run never leaves truncated results.