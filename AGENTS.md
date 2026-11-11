# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A local-first desktop app that transcribes video/audio to text with Whisper `large-v3` (faster-whisper). Tauri 2 shell → Rust backend → spawned Python worker. Everything runs on-device.

- **Frontend:** vanilla JS, no bundler — `src/index.html`, `src/main.js`, `src/styles.css`.
- **Backend (Rust):** `src-tauri/src/lib.rs` — Tauri commands only; it never imports Python, it **spawns** the worker as a subprocess.
- **Worker (Python):** `src-tauri/python/transcribe_worker.py` — does the actual transcription.
- **Windows portable runtime:** `src-tauri/runtime/` — a self-contained Python 3.11 + `faster-whisper` (CPU). **Generated, git-ignored.**
- **Ubuntu development environment:** `.venv/` — external Python + `faster-whisper`, prepared by `scripts/setup_ubuntu.sh`. **Generated, git-ignored; never bundled.**
- **Ubuntu user installation:** `$XDG_DATA_HOME/video-transcriber` (default `~/.local/share/video-transcriber`) — installed executable, worker, and independent `venv`, prepared by `scripts/install_ubuntu.sh`. The launcher sets absolute installed Python/worker paths; runtime must not depend on the checkout.

## Commands

```bash
npm run setup:ubuntu                 # check prerequisites; prepare .venv + npm
npm run dev:ubuntu                   # run from source on Ubuntu desktop
npm run tauri -- dev                 # run from source (Windows or Linux)
npm run tauri -- build               # build; Linux still needs external Python
npm run tauri -- build --no-bundle    # executable without installer
npm run install:ubuntu               # install the built release app for your user
npm run uninstall:ubuntu             # remove app/runtime, preserve settings/models
./scripts/build_runtime.sh           # Windows only: regenerate portable runtime
```

Checks:

```bash
cargo check            # in src-tauri/
node --check src/main.js
python3 -m py_compile src-tauri/python/transcribe_worker.py # python on Windows
.venv/bin/python -m unittest discover -s tests -v # includes Hub API checks; use your worker Python on Windows
for script in scripts/*.sh; do bash -n "$script" || exit; done
.venv/bin/python -c 'import faster_whisper, ctranslate2, onnxruntime, av' # Ubuntu
cargo fmt               # Rust formatting
npx prettier --write src/   # JS/HTML/CSS formatting
```

## Critical invariants (get these wrong and it breaks)

1. **The worker is a subprocess, not a library.** Rust spawns it and streams its stdout, emitting each parsed line as a Tauri event. There is no Python<->Rust FFI.

2. **Worker stdout is a JSON-lines protocol.** Every line printed to stdout **must** be valid JSON (use the `emit()` helper). Non-JSON on stdout is silently dropped by the Rust side. Debug/prints go to **stderr**.

3. **Worker stderr is only surfaced on failure.** Rust drains stderr concurrently and, on non-zero exit, returns its tail in the error message. There is no streaming of stderr.

4. **The model (~3 GB) is downloaded at runtime, never committed.** First transcription fetches `large-v3` to `~/.cache/whisper-transcribe/large-v3` (or reuses the Hugging Face cache). The model cache is outside the repository; the runtime generation script is tracked.

5. **`runtime/`, `.venv/`, `target/`, `node_modules/`, and Tauri generated schemas are git-ignored artifacts.** Do not edit fixtures into `runtime/`; regenerate it on Windows with `scripts/build_runtime.sh` (requires Python 3.11 on `PATH` — the embeddable interpreter is 3.11, so wheels are `cp311`). The script must reject Linux before deleting the existing runtime. Ubuntu setup must not install system packages or global toolchains automatically.

6. **Outputs are written atomically.** The worker writes `<out>.txt.part`/`.srt.part`/`.vtt.part` and `os.replace`s to the final names only on success. Cancel (below) deletes the `.part` files. Never bypass this.

7. **Keep platform boundaries explicit.** Windows uses the embeddable Python runtime, CUDA DLL bootstrap (`os.name == "nt"`), `taskkill /T /F`, and Explorer. Linux uses external Python, a native worker cancellation path, and `xdg-open` for the containing folder. `tauri.linux.conf.json` overrides resources to include only the worker; it does not create a self-contained Linux runtime. Preserve platform and debug-build guards.

## Device / GPU / CUDA

- The Windows runtime is **CPU-only by design** (size/portability). On Windows, GPU needs either:
  - the in-app **Install GPU** flow (`worker --install-cuda` downloads `nvidia-cublas/cudnn/cuda-runtime-cu12` wheels to `%LOCALAPPDATA%\whisper-transcribe\cuda`), or
  - `PYTHON_TRANSCRIBE` pointing at an existing Python with `faster-whisper` + CUDA wheels.
- `choose_device()` (**don't** simplify it) gates CUDA on the compute libs *actually being loadable* (cublas/cudnn/cudart resolved by `_bootstrap_cuda()`), **not** on `get_cuda_device_count() > 0`. A visible GPU without the libs must fall back to CPU, or model load fails with `cublas64_12.dll not found`.
- `_bootstrap_cuda()` searches both the runtime's `site-packages/nvidia` **and** the per-user CUDA dir. Keep that.
- Ubuntu development is CPU-first. Never offer the Windows CUDA wheel installer on Linux; guard both the UI and worker/backend entry points. Linux CUDA setup is manual.
- A successful `--probe` is not a dependency or transcription test: missing imports may produce CPU fallback. Verify `faster_whisper`, `ctranslate2`, `onnxruntime`, and `av` imports separately.

## Cancellation

`RunState` (`Arc<Mutex<…>>` in `lib.rs`) tracks the in-flight worker PID + output prefix. `transcribe` registers on spawn, returns `Err("cancelled")` after `wait()` if a cancel was requested. `cancel_transcribe` sets the flag, terminates the worker (`taskkill /T /F` on Windows, SIGTERM on Linux), then deletes the `.part` files. Transcription cleans partial files again after the worker exits to cover asynchronous termination. The frontend treats the `"cancelled"` string distinctly from real errors.

## Conventions

- Device/language settings persist via `get_settings`/`set_settings`: `%LOCALAPPDATA%\whisper-transcribe\settings.json` on Windows; `$XDG_CONFIG_HOME/whisper-transcribe/settings.json` on Linux, defaulting to `~/.config/whisper-transcribe/settings.json`.
- Tauri resolves the worker via `app.path().resource_dir()`, with source-tree/`CARGO_MANIFEST_DIR` fallbacks for dev. Only Windows resolves a bundled `runtime/python.exe`.
- `PYTHON_TRANSCRIBE` overrides interpreter discovery. Linux debug builds try the repository `.venv/bin/python` before `python3` on `PATH`; release builds need external Python. Windows preserves its bundled/dev interpreter discovery.
- Env vars: Rust consumes `PYTHON_TRANSCRIBE` and `TRANSCRIBE_WORKER_PATH`; the worker consumes `TRANSCRIBE_DEVICE`, `TRANSCRIBE_COMPUTE_TYPE`, `WHISPER_MODEL_DIR`, `HF_HOME`, and `HF_TOKEN` (see README for detail).
- No credentials or user-specific paths in tracked source. Development paths must derive from the repository or platform environment.
