#!/usr/bin/env python
"""Generalized transcription worker for the Tauri app.
Usage: transcribe_worker.py <input_video> <output_prefix> [language]
Prints JSON progress lines to stdout; final line is a JSON summary.

On error: a JSON {"event":"error"} line on stdout, a traceback on stderr,
and a non-zero exit code.
"""
import os, time, glob, ctypes, site, sys, json, traceback

# ---- CUDA DLL bootstrap (Windows) ----
def _cuda_dir():
    """Per-user dir where on-demand NVIDIA libs are installed (writable — the
    bundled runtime under the install dir is not)."""
    localappdata = os.environ.get("LOCALAPPDATA")
    return os.path.join(localappdata, "whisper-transcribe", "cuda") if localappdata else None


def _nvidia_roots():
    """Candidate nvidia/ roots: the runtime's own site-packages plus the
    per-user CUDA dir we install into on demand."""
    _sp = None
    for p in site.getsitepackages():
        if p.endswith("site-packages"):
            _sp = p
            break
    if _sp is None:
        _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    roots = [os.path.join(_sp, "nvidia")]
    cuda_dir = _cuda_dir()
    if cuda_dir:
        roots.append(os.path.join(cuda_dir, "nvidia"))
    return roots


def _bootstrap_cuda():
    """Register every nvidia/*/bin dir on the DLL search path and eagerly load
    the CUDA runtime families, so pip-installed CUDA/cuDNN resolves at import.
    Returns (resolved, missing) family names."""
    # Families ctranslate2 actually loads. `nvrtc` is deliberately omitted (it's
    # only for CUDA JIT compilation, not inference) — listing it would leave a
    # permanently-"missing" warning after every GPU install.
    families = {
        "cublas": "cublas64_*.dll",
        "cublasLt": "cublasLt64_*.dll",
        "cudnn": "cudnn64_*.dll",
        "cudart": "cudart64_*.dll",
    }

    for base in _nvidia_roots():
        for dll_dir in glob.glob(os.path.join(base, "*", "bin")):
            try:
                os.add_dll_directory(dll_dir)
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            except Exception as exc:
                print(json.dumps({"event": "cuda_bootstrap", "path": dll_dir, "error": str(exc)}), flush=True)

    # Match by family prefix, not exact SONAME: NVIDIA wheels bump the trailing
    # digit on upgrade (cudnn64_9 -> cudnn64_10), so hardcoding the version
    # silently breaks the probe while the DLLs are actually present.
    resolved, missing = [], []
    for fam, pattern in families.items():
        loaded = False
        for base in _nvidia_roots():
            for path in glob.glob(os.path.join(base, "*", "bin", pattern)):
                try:
                    ctypes.WinDLL(path)
                    loaded = True
                    break
                except Exception:
                    continue
            if loaded:
                break
        (resolved if loaded else missing).append(fam)

    # Surface only a *partial* CUDA install (some families present, others
    # missing) — a real breakage worth surfacing. A fully-absent nvidia/ dir
    # means a clean CPU-only machine; choose_device() falls back to CPU
    # silently. The forced-CUDA case is surfaced even when nothing resolved.
    explicit_cuda = os.environ.get("TRANSCRIBE_DEVICE", "").strip().lower() == "cuda"
    if missing and (resolved or explicit_cuda):
        print(json.dumps({"event": "cuda_bootstrap",
                          "resolved": resolved, "missing": missing}), flush=True)
    return resolved, missing


# Result of the CUDA bootstrap: which compute-library families actually
# resolved (and are therefore loadable). None on non-Windows, where we don't
# gate device selection on window-specific DLLs.
CUDA_FAMILIES_RESOLVED = _bootstrap_cuda()[0] if os.name == "nt" else None


def emit(obj):
    print(json.dumps(obj), flush=True)


def choose_device():
    """Probe for CUDA and fall back to CPU; explicit override wins."""
    explicit = os.environ.get("TRANSCRIBE_DEVICE", "").strip().lower()
    if explicit:
        return explicit
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            return "cpu"
        # A visible GPU isn't enough: the compute libraries (cuBLAS/cuDNN/
        # cudart) must actually be loadable, or model load fails with
        # "cublas64_*.dll not found". On Windows, gate on the bootstrap result
        # from import time.
        if CUDA_FAMILIES_RESOLVED is not None:
            critical = {"cublas", "cudnn", "cudart"}
            if not critical.issubset(set(CUDA_FAMILIES_RESOLVED)):
                return "cpu"
        return "cuda"
    except Exception:
        return "cpu"


def ts(s, sep):
    """Format seconds as HH:MM:SS,mmm with the given millisecond separator."""
    total_ms = int(round(s * 1000))
    h = total_ms // 3600000
    m = (total_ms // 60000) % 60
    sec = (total_ms // 1000) % 60
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{sec:02d}{sep}{ms:03d}"


def ts_srt(s):
    return ts(s, ",")


def ts_vtt(s):
    return ts(s, ".")


MODEL_REPO = "Systran/faster-whisper-large-v3"
# Files faster-whisper needs (CTranslate2 format); model.bin dominates (~3 GB).
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin",
               "tokenizer.json", "vocabulary.json"]


def _model_dir():
    """Dedicated directory we download the model into when it isn't cached."""
    return os.environ.get("WHISPER_MODEL_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "whisper-transcribe", "large-v3")


def _model_ready(d):
    return os.path.isfile(os.path.join(d, "model.bin"))


def _model_cached_in_hf_hub():
    """True if large-v3 is already in the default HuggingFace cache."""
    try:
        root = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub")
        return os.path.isdir(
            os.path.join(root, "models--Systran--faster-whisper-large-v3"))
    except Exception:
        return False


class _JsonBar:
    """Minimal tqdm-compatible sink relaying download bytes as JSON on stdout.

    `huggingface_hub.hf_hub_download(tqdm_class=_JsonBar)` instantiates one of
    these per file (with `total` = file size) and drives it through either the
    plain-HTTP loop (`.update(n)`) or the Xet path (`.update(n)`,
    `.set_postfix_str(...)`, and reads/mutates `.n`/`.total`). We implement just
    enough of tqdm's surface for both. Tiny metadata files (total < 1 MB) are
    skipped so only the ~3 GB model.bin drives the progress bar.
    """

    def __init__(self, **kwargs):
        self.total = int(kwargs.get("total") or 0)
        self.desc = str(kwargs.get("desc", ""))
        self.n = int(kwargs.get("initial") or 0)
        self._last_emit = -1

    def update(self, n=1):
        self.n += int(n)
        # Throttle: emit at most ~every 0.25 s (chunks arrive faster than that).
        now = time.monotonic()
        if self.total >= 1_000_000 and now - self._last_emit >= 0.25:
            self._last_emit = now
            emit({"event": "download_progress", "file": self.desc,
                  "bytes": self.n, "total": self.total})

    # ---- no-op tqdm surface used by huggingface_hub's Xet reporter ----
    def refresh(self, **kwargs):
        pass

    def close(self):
        pass

    def set_postfix_str(self, postfix, refresh=False):
        pass

    def set_transfer_postfix_str(self, postfix, refresh=False):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _load_model(device, compute_type):
    """Return a loaded WhisperModel, downloading large-v3 (with live progress)
    only when it isn't already anywhere on disk."""
    from faster_whisper import WhisperModel

    loading = {"event": "loading_model", "device": device, "compute_type": compute_type}

    # 1) Already in our dedicated download dir.
    model_dir = _model_dir()
    if _model_ready(model_dir):
        emit(loading)
        return WhisperModel(model_dir, device=device, compute_type=compute_type)

    # 2) Already in the default HuggingFace cache (fast path, no download).
    if _model_cached_in_hf_hub():
        emit(loading)
        return WhisperModel("large-v3", device=device, compute_type=compute_type)

    # 3) Nothing on disk: download with progress, then load from the local dir.
    emit({"event": "downloading_model", "device": device,
          "note": "first run: downloading ~3 GB Whisper large-v3 model"})
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(model_dir, exist_ok=True)
        for name in MODEL_FILES:
            hf_hub_download(MODEL_REPO, name, local_dir=model_dir, tqdm_class=_JsonBar)
    except Exception as e:
        raise RuntimeError(f"model download failed: {e}") from e

    emit(loading)
    return WhisperModel(model_dir, device=device, compute_type=compute_type)


def _main():
    src = sys.argv[1]
    out_prefix = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("", "auto") else None

    device = choose_device()
    compute_type = os.environ.get("TRANSCRIBE_COMPUTE_TYPE", "").strip() or (
        "int8_float16" if device == "cuda" else "int8"
    )

    model = _load_model(device, compute_type)

    emit({"event": "transcribing", "file": os.path.basename(src), "device": device})
    t0 = time.time()
    segments, info = model.transcribe(
        src,
        language=language,
        beam_size=8,
        word_timestamps=True,
        # Silence is skipped rather than fed to the decoder, so there's no dead
        # air to hallucinate on (and it's faster on quiet stretches).
        vad_filter=True,
        # Critical: stop conditioning each token on already-emitted text while
        # decoding greedily — the fixed-point loop behind runaway repetition
        # (e.g. "Yeah Yeah Yeah…"). `no_repeat_ngram_size` + `repetition_penalty`
        # hard-stop any residual n-gram looping.
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        temperature=0.0,
    )
    media_duration = info.duration
    emit({"event": "detected_language", "language": info.language,
          "probability": round(info.language_probability, 3),
          "duration_s": media_duration})

    total_chars = 0
    n = 0
    last_end = 0.0
    # Write to *.part temp files and atomically rename only after full success,
    # so a crash mid-transcribe never leaves truncated files that look complete.
    tmp = {ext: f"{out_prefix}.{ext}.part" for ext in ("txt", "srt", "vtt")}
    with open(tmp["txt"], "w", encoding="utf-8") as f_txt, \
         open(tmp["srt"], "w", encoding="utf-8") as f_srt, \
         open(tmp["vtt"], "w", encoding="utf-8") as f_vtt:
        f_vtt.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            line = seg.text.strip()
            n += 1
            total_chars += len(line)
            last_end = seg.end
            f_txt.write(line + "\n")
            f_srt.write(f"{i}\n{ts_srt(seg.start)} --> {ts_srt(seg.end)}\n{line}\n\n")
            f_vtt.write(f"{ts_vtt(seg.start)} --> {ts_vtt(seg.end)}\n{line}\n\n")
            if n % 25 == 0:
                for f in (f_txt, f_srt, f_vtt):
                    f.flush()
                # media_s is the media timestamp of the latest segment, so the
                # frontend can compute an honest "audio processed" fraction.
                emit({"event": "progress", "segments": n,
                      "elapsed_s": round(time.time() - t0, 1),
                      "media_s": round(last_end, 1),
                      "duration_s": media_duration})

    for ext in ("txt", "srt", "vtt"):
        os.replace(tmp[ext], f"{out_prefix}.{ext}")

    duration = media_duration if media_duration else last_end
    emit({"event": "done", "segments": n, "language": info.language,
          "duration_s": round(duration or 0, 1),
          "elapsed_s": round(time.time() - t0, 1),
          "chars": total_chars,
          "txt": out_prefix + ".txt",
          "srt": out_prefix + ".srt",
          "vtt": out_prefix + ".vtt"})


# ---- on-demand CUDA install (pip-free) ----
# Known-good pin set (matches the dev venv proven to work with the bundled
# ctranslate2 4.8.2). nvrtc is omitted — not needed for inference.
CUDA_WHEELS = [
    ("nvidia-cublas-cu12", "12.9.2.10"),
    ("nvidia-cudnn-cu12", "9.25.1.1"),
    ("nvidia-cuda-runtime-cu12", "12.9.79"),
]


def _resolve_wheel(name, version):
    """Return (url, sha256) for the win_amd64 bdist_wheel from PyPI."""
    import httpx
    resp = httpx.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    for f in data.get("urls", []):
        if f.get("packagetype") == "bdist_wheel" and "win_amd64" in f.get("filename", ""):
            sha = (f.get("digests") or {}).get("sha256")
            if not sha:
                # A missing digest must be a hard failure on a verify-before-
                # extract path, not a silent skip of verification.
                raise RuntimeError(
                    f"PyPI returned no sha256 digest for {name}=={version}; "
                    "refusing to download an unverifiable wheel")
            return f["url"], sha
    raise RuntimeError(f"no win_amd64 wheel for {name}=={version}")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_wheel(name, version, url, sha, dest_dir, progress):
    import httpx
    dest = os.path.join(dest_dir, f"{name}-{version}.whl")
    last = 0.0
    with httpx.stream("GET", url, follow_redirects=True,
                      timeout=httpx.Timeout(30.0, read=120.0)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= 0.25:  # throttle: ~4 events/sec, not ~1/1 MB
                    last = now
                    progress(name, done, total)
    if total:  # final update so the bar reaches the full total
        progress(name, done, total)
    if sha and _sha256(dest) != sha:
        os.remove(dest)
        raise RuntimeError(f"checksum mismatch for {name}-{version}")
    return dest


def _extract_wheel(wheel_path, dest_dir):
    import zipfile
    dest = os.path.realpath(dest_dir)
    with zipfile.ZipFile(wheel_path) as z:
        # Zip-slip guard: reject any member that resolves outside dest_dir
        # (e.g. a `../../` entry). Wheels are sha256-verified against PyPI
        # before we get here, but this keeps a compromised artifact from ever
        # turning an extract into an arbitrary file write.
        for member in z.infolist():
            target = os.path.realpath(os.path.join(dest, member.filename))
            if target != dest and not target.startswith(dest + os.sep):
                raise RuntimeError(
                    f"refusing to extract {member.filename!r}: escapes destination")
        z.extractall(dest)
    os.remove(wheel_path)


def _install_cuda():
    cuda_dir = _cuda_dir()
    if not cuda_dir:
        raise RuntimeError("LOCALAPPDATA not set; cannot choose an install location")
    os.makedirs(cuda_dir, exist_ok=True)

    def progress(name, done, total):
        emit({"event": "cuda_install_progress", "stage": "download",
              "wheel": name, "bytes": done, "total": total})

    for i, (name, version) in enumerate(CUDA_WHEELS):
        emit({"event": "cuda_install_progress", "stage": "resolve",
              "wheel": name, "index": i, "count": len(CUDA_WHEELS)})
        url, sha = _resolve_wheel(name, version)
        emit({"event": "cuda_install_progress", "stage": "download",
              "wheel": name, "bytes": 0, "total": 0})
        wheel_path = _download_wheel(name, version, url, sha, cuda_dir, progress)
        emit({"event": "cuda_install_progress", "stage": "extract", "wheel": name})
        _extract_wheel(wheel_path, cuda_dir)

    emit({"event": "cuda_install_done", "dir": cuda_dir})


def _probe_device():
    """Lightweight CUDA-availability probe (no model load).

    Returns the same device `choose_device()` would pick, so the frontend can
    disable the GPU option when CUDA isn't actually usable.
    """
    try:
        dev = choose_device()
        return {"event": "probe_result", "device": dev, "cuda_available": dev == "cuda"}
    except Exception as e:
        return {"event": "probe_result", "cuda_available": False, "error": str(e)}


def main():
    try:
        _main()
    except Exception as exc:
        try:
            emit({"event": "error", "message": str(exc)})
        except Exception:
            print(str(exc), flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        emit(_probe_device())
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--install-cuda":
        try:
            _install_cuda()
            sys.exit(0)
        except Exception as exc:
            emit({"event": "error", "message": str(exc)})
            traceback.print_exc()
            sys.exit(1)
    main()