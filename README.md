# Video Transcriber

A local-first desktop app that transcribes video and audio files on your own machine. Media stays local; setup and the first model download need internet access. Built with [Tauri 2](https://tauri.app/) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper `large-v3`).

Drag in a file, pick a language, and get three output formats with word-level timestamps. Windows supports the bundled Python runtime and optional CUDA libraries. Ubuntu supports a per-user desktop installation with its own Python environment and CPU transcription.

## Features

- **Drop a video or audio file** (MP4, MKV, MOV, WebM, AVI, MP3, WAV, FLAC, …) or browse for one.
- **Three outputs at once** — plain text (`.txt`), SubRip subtitles (`.srt`), and WebVTT (`.vtt`) — written next to the source file.
- **Whisper `large-v3`** with beam search, word-level timestamps, and silence filtered out (VAD), with repetition-collapse guards.
- **Language auto-detection** — or pin one of 8 languages (English, Polish, German, Spanish, French, Italian, Dutch, Portuguese).
- **Device selection** — *Auto (GPU → CPU)*, *CPU*, or *GPU (CUDA)*, remembered between launches.
- **On-demand GPU support on Windows** — the *Install GPU support* action downloads NVIDIA libraries (~1.1 GB). Linux does not offer this installer.
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

**Model**: the ~3 GB `large-v3` model downloads from Hugging Face on the first transcription (with a progress bar) into `~/.cache/whisper-transcribe/large-v3`, or is reused from the Hugging Face cache if already present. Allow additional disk space for Python packages and Rust build artifacts.

**GPU (CUDA)**: the Windows runtime is CPU-only for size and portability. On Windows, the app can download missing NVIDIA libraries (cuBLAS/cuDNN/cudart) into `%LOCALAPPDATA%\whisper-transcribe\cuda`. Alternatively, point `PYTHON_TRANSCRIBE` at an existing Python environment. Linux CUDA setup is manual and outside the CPU development setup below.

## Requirements

- **Windows 10/11** (x64) for the bundled runtime and Windows installers.
- **Ubuntu 24.04 desktop** for a per-user installation or development from source. The Ubuntu installer creates a dedicated Python virtual environment; system Python and native desktop libraries must remain installed.
- Source builds need [Node.js](https://nodejs.org/) with npm, the [Rust toolchain](https://rustup.rs/), and the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) for the target platform.
- Building the Windows portable runtime also needs Git Bash/MSYS2, `curl`, `unzip`, and Python 3.11 on `PATH`.

An NVIDIA GPU is optional — everything runs on CPU, just slower.

## Development

### Ubuntu

Install the OS prerequisites:

```bash
sudo apt update
sudo apt install build-essential pkg-config libwebkit2gtk-4.1-dev \
  libssl-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev \
  curl wget file python3-venv xdg-utils
```

Install Node.js with npm if needed, then install Rust following [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

From the repository directory:

```bash
npm run setup:ubuntu
npm run dev:ubuntu
```

The setup script checks prerequisites, creates or reuses `.venv`, installs `faster-whisper`, verifies its imports, and runs `npm ci`. It does not install system packages or Rust. Re-running it is supported. Python 3.12 (Ubuntu 24.04's default) is suitable; PyAV provides the decoding libraries, so this setup does not require the `ffmpeg` command.

The launcher uses `.venv/bin/python` unless `PYTHON_TRANSCRIBE` is set, checks dependencies, and starts Tauri from the repository root. It forwards extra arguments, for example `npm run dev:ubuntu -- --no-watch`. Run it in a desktop session and select **CPU** for the initial transcription. CPU speed depends on the hardware; full `large-v3` transcription can be slow.

Tauri automatically merges `src-tauri/tauri.linux.conf.json` on Linux, so it bundles only the worker script and does not require the Windows `runtime/` directory. Plain `npm run tauri -- dev` also finds the repository `.venv` in debug builds, then falls back to `python3` on `PATH`. The desktop installer below supplies the release build with its own Python environment. A manually launched release executable instead needs `PYTHON_TRANSCRIBE` or `python3` on `PATH` with the dependencies installed.

For a dependency sanity check without starting the UI or downloading the model:

```bash
.venv/bin/python -c 'import faster_whisper, ctranslate2, onnxruntime, av'
.venv/bin/python src-tauri/python/transcribe_worker.py --probe
```

Run the import check as well as `--probe`: device probing can report a CPU fallback even when a Python dependency is missing. A successful probe does not verify model loading or transcription.

### Install on Ubuntu

After the Ubuntu setup above, build the release executable and install it for your user:

```bash
npm run tauri -- build --no-bundle
npm run install:ubuntu
```

Open Ubuntu **Applications**, search for **Video Transcriber**, and launch it using its icon. You can right-click the running icon and choose **Pin to Dash** to keep it on the dock. No terminal, Node.js, Rust, or source checkout is needed to launch the installed app.

The installer copies the executable and worker into `$XDG_DATA_HOME/video-transcriber` (default `~/.local/share/video-transcriber`), creates a separate `venv` there with `faster-whisper==1.2.1`, and registers the icon and application launcher. It does not require sudo or install system packages. Package installation needs internet; the ~3 GB model downloads on the first transcription. Existing settings and downloaded models are reused.

To update, rebuild and rerun `npm run install:ubuntu`. An already-running app keeps its current executable until restarted. For a custom Cargo output directory, use `npm run install:ubuntu -- --binary /absolute/path/to/release/tauri-transcribe`.

To check the installed Python dependencies and device probe without opening a window or downloading the model:

```bash
"${XDG_DATA_HOME:-$HOME/.local/share}/video-transcriber/run-video-transcriber" --check
```

Run that same launcher without `--check` to open the app from a terminal. To remove the installed app and its Python environment, run `npm run uninstall:ubuntu`; your transcription files, settings, and model caches are preserved.

### Windows

```bash
npm ci
npm run tauri -- dev
```

Prepare the portable runtime below, or set `PYTHON_TRANSCRIBE` to an existing Python with `faster-whisper`. Windows debug builds also retain the `%LOCALAPPDATA%\hermes\hermes-agent\venv` fallback.

## Building a standalone `.exe`

Windows builds bundle a portable, self-contained Python runtime so end users don't install Python. Run these steps on Windows; `build_runtime.sh` refuses to run on Linux before touching the runtime directory.

1. **Build the portable runtime** (downloads the Python 3.11 embeddable package + `faster-whisper` CPU wheels into `src-tauri/runtime/`, which is git-ignored):

   ```bash
   ./scripts/build_runtime.sh
   ```

2. **Compile the app**:

   ```bash
   npm run tauri -- build              # produce installers (NSIS/MSI)
   npm run tauri -- build --no-bundle   # just the .exe
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
  tauri.conf.json          # Windows resources: worker + runtime
  tauri.linux.conf.json    # Linux resources: worker only
scripts/build_runtime.sh   # builds the Windows portable runtime
scripts/setup_ubuntu.sh    # prepares local Python + Node dependencies
scripts/dev_ubuntu.sh      # launches with the local Python environment
scripts/install_ubuntu.sh  # installs the release app and dedicated Python for your user
scripts/uninstall_ubuntu.sh # removes installed app; preserves settings and model caches
.venv/                    # Ubuntu development Python (generated, git-ignored)
```

## Worker CLI

The worker also exposes standalone modes (useful for testing):

```bash
python transcribe_worker.py <input> <output_prefix> [language]   # transcribe
python transcribe_worker.py --probe        # print device availability (JSON)
python transcribe_worker.py --install-cuda # Windows only: install CUDA libs
```

## Configuration

Settings (device + language) are saved to `%LOCALAPPDATA%\whisper-transcribe\settings.json` on Windows and `$XDG_CONFIG_HOME/whisper-transcribe/settings.json` on Linux (default: `~/.config/whisper-transcribe/settings.json`).

Environment variables:

| Variable | Purpose |
| --- | --- |
| `PYTHON_TRANSCRIBE` | Path to a Python with `faster-whisper` installed (overrides the bundled/dev interpreter). For example, `/path/to/project/.venv/bin/python` on Ubuntu. |
| `TRANSCRIBE_DEVICE` | Force `cpu` or `cuda` (the UI's device selector sets this per run). |
| `TRANSCRIBE_COMPUTE_TYPE` | Override the model precision (`int8_float16` on GPU, `int8` on CPU by default). |
| `TRANSCRIBE_WORKER_PATH` | Override the location of `transcribe_worker.py`. |
| `WHISPER_MODEL_DIR` | Where to look for / download the model. |
| `HF_HOME` / `HF_TOKEN` | Hugging Face cache location / access token (for gated repos or higher rate limits). |
| `HF_HUB_DISABLE_XET` | Defaults to `1` in the worker, using standard HTTPS downloads to avoid Xet/CAS transfer failures on restricted networks. Set `0` before launching to opt into Xet. |

## Notes

- On Windows, the optional GPU download installs NVIDIA libraries into your user profile (`%LOCALAPPDATA%\whisper-transcribe\cuda`), not system-wide.
- Output files are written to `.txt.part`/`.srt.part`/`.vtt.part` temp files and atomically renamed on success, so a cancelled or crashed run never leaves truncated results.
