# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A local-first desktop app that transcribes video/audio to text with Whisper `large-v3` (faster-whisper). Tauri 2 shell → Rust backend → spawned Python worker. Everything runs on-device.

- **Frontend:** vanilla JS, no bundler — `src/index.html`, `src/main.js`, `src/styles.css`.
- **Backend (Rust):** `src-tauri/src/lib.rs` — Tauri commands only; it never imports Python, it **spawns** the worker as a subprocess.
- **Worker (Python):** `src-tauri/python/transcribe_worker.py` — does the actual transcription.
- **Portable runtime:** `src-tauri/runtime/` — a self-contained Python 3.11 + `faster-whisper` (CPU). **Generated, git-ignored.**

## Commands

```bash
npm run tauri dev                    # run from source (Windows)
npm run tauri build                  # bundled installers
npm run tauri build -- --no-bundle   # just the .exe
./scripts/build_runtime.sh           # (re)generate src-tauri/runtime/
```

Checks (no unified test suite):

```bash
cargo check            # in src-tauri/
node --check src/main.js
python -m py_compile src-tauri/python/transcribe_worker.py
cargo fmt               # Rust formatting
npx prettier --write src/   # JS/HTML/CSS formatting
```

## Critical invariants (get these wrong and it breaks)

1. **The worker is a subprocess, not a library.** Rust spawns it and streams its stdout, emitting each parsed line as a Tauri event. There is no Python<->Rust FFI.

2. **Worker stdout is a JSON-lines protocol.** Every line printed to stdout **must** be valid JSON (use the `emit()` helper). Non-JSON on stdout is silently dropped by the Rust side. Debug/prints go to **stderr**.

3. **Worker stderr is only surfaced on failure.** Rust drains stderr concurrently and, on non-zero exit, returns its tail in the error message. There is no streaming of stderr.

4. **The model (~3 GB) is downloaded at runtime, never committed.** First run fetches `large-v3` to `~/.cache/whisper-transcribe/large-v3` (or the Hugging Face cache). `scripts/build_runtime.sh` and the model cache are both out of the repo.

5. **`runtime/`, `target/`, and `node_modules/` are git-ignored build artifacts.** Do not edit fixtures into `runtime/`; regenerate it with `scripts/build_runtime.sh` (requires a Python 3.11 on `PATH` — the embeddable interpreter is 3.11, so wheels are `cp311`).

6. **Outputs are written atomically.** The worker writes `<out>.txt.part`/`.srt.part`/`.vtt.part` and `os.replace`s to the final names only on success. Cancel (below) deletes the `.part` files. Never bypass this.

7. **Windows-only, in practice.** The CUDA DLL bootstrap (`os.name == "nt"`), process-tree cancel (`taskkill /T /F`), the embeddable-Python runtime, and `open_in_folder` (explorer) are all Windows-specific. There are `#[cfg(target_os = "windows")]` / `#[cfg(debug_assertions)]` guards — keep them.

## Device / GPU / CUDA

- The shipped bundle is **CPU-only by design** (size/portability). GPU needs either:
  - the in-app **Install GPU** flow (`worker --install-cuda` downloads `nvidia-cublas/cudnn/cuda-runtime-cu12` wheels to `%LOCALAPPDATA%\whisper-transcribe\cuda`), or
  - `PYTHON_TRANSCRIBE` pointing at an existing Python with `faster-whisper` + CUDA wheels.
- `choose_device()` (**don't** simplify it) gates CUDA on the compute libs *actually being loadable* (cublas/cudnn/cudart resolved by `_bootstrap_cuda()`), **not** on `get_cuda_device_count() > 0`. A visible GPU without the libs must fall back to CPU, or model load fails with `cublas64_12.dll not found`.
- `_bootstrap_cuda()` searches both the runtime's `site-packages/nvidia` **and** the per-user CUDA dir. Keep that.

## Cancellation

`RunState` (`Arc<Mutex<…>>` in `lib.rs`) tracks the in-flight worker PID + output prefix. `transcribe` registers on spawn, returns `Err("cancelled")` after `wait()` if a cancel was requested. `cancel_transcribe` sets the flag → `taskkill /T /F` → deletes the `.part` files. The frontend treats the `"cancelled"` string distinctly from real errors.

## Conventions

- Device/language settings persist to `%LOCALAPPDATA%\whisper-transcribe\settings.json` via `get_settings`/`set_settings`.
- Tauri resolves resources via `app.path().resource_dir()` — on Windows that's the exe directory (the build copies `python/` and `runtime/` there). Resolve the worker and bundled `python.exe` from it, with source-tree/`CARGO_MANIFEST_DIR` fallbacks for dev.
- Env vars consumed by the worker: `PYTHON_TRANSCRIBE`, `TRANSCRIBE_DEVICE`, `TRANSCRIBE_COMPUTE_TYPE`, `TRANSCRIBE_WORKER_PATH`, `WHISPER_MODEL_DIR`, `HF_HOME`, `HF_TOKEN` (see README for detail).
- No credentials or user-specific paths in tracked source. (The dev Python is resolved from `%LOCALAPPDATA%`, not hardcoded.)