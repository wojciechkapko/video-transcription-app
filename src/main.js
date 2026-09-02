const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);

let currentFile = null;
let currentFileName = null;
let outputPaths = { txt: null, srt: null, vtt: null };
let busy = false;
let mediaDuration = 0;          // set from worker events; drives the progress bar
let currentDevice = null;       // "cuda" | "cpu" | null
let runStart = 0;               // performance.now() at transcribe start
let lastFrac = 0;               // last media fraction for ETA
let activeTab = "txt";

const els = {
  dropzone: $("dropzone"),
  dzIcon: $("dz-icon"),
  dzTitle: $("dz-title"),
  dzFile: $("dz-file"),
  dzReady: $("dz-ready"),
  dzReadyName: $("dz-ready-name"),
  browseBtn: $("browse-btn"),
  changeFile: $("change-file"),
  langSelect: $("lang-select"),
  transcribeBtn: $("transcribe-btn"),
  progress: $("progress"),
  progressTitle: $("progress-title"),
  progressPct: $("progress-pct"),
  progressFill: $("progress-fill"),
  progressStatus: $("progress-status"),
  progressLog: $("progress-log"),
  statElapsed: $("stat-elapsed"),
  statEta: $("stat-eta"),
  statSegments: $("stat-segments"),
  statDevice: $("stat-device"),
  results: $("results"),
  resultsFile: $("results-file"),
  statsStrip: $("stats-strip"),
  transcriptView: $("transcript-view"),
  copyActive: $("copy-active"),
  openFolder: $("open-folder"),
  devicePill: $("device-pill"),
  devicePillText: $("device-pill-text"),
  toast: $("toast"),
};

/* ---------- tiny helpers ---------- */

function fmtClock(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "—";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtDuration(sec) {
  if (!Number.isFinite(sec) || sec <= 0) return "—";
  if (sec < 60) return `${Math.round(sec)}s`;
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

let toastTimer = null;
function toast(msg, tone = "") {
  els.toast.innerHTML = msg;
  els.toast.className = "toast show " + tone;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.className = "toast"), 2200);
}

function setDevicePill(state, text) {
  els.devicePill.dataset.state = state;
  els.devicePillText.textContent = text;
}

function deviceLabel(d) {
  if (!d) return "—";
  return d === "cuda" ? "CUDA · GPU" : "CPU";
}

function logLine(msg) {
  els.progressLog.classList.remove("hidden");
  const li = document.createElement("li");
  li.textContent = msg;
  els.progressLog.appendChild(li);
  els.progressLog.scrollTop = els.progressLog.scrollHeight;
}

function setProgress(pct, status) {
  const p = Math.max(2, Math.min(99, pct));
  els.progressFill.style.width = `${p}%`;
  els.progressPct.textContent = `${Math.round(p)}%`;
  if (status != null) els.progressStatus.textContent = status;
}

// Fraction of media processed (latest segment timestamp vs. total duration),
// capped so the bar never looks stuck; 100% is reserved for "done".
function progressPct(ev) {
  const base = 15;
  const mediaS = ev.media_s;
  if (mediaDuration > 0 && mediaS > 0) {
    const frac = Math.min(0.95, mediaS / mediaDuration);
    lastFrac = frac;
    return base + frac * 80;
  }
  return base + Math.min(80, (ev.segments / 200) * 80);
}

function updateEta(ev) {
  if (runStart && ev.elapsed_s > 1 && mediaDuration > 0 && lastFrac > 0.01) {
    const rate = lastFrac / ev.elapsed_s;         // fraction per second
    const remain = (1 - lastFrac) / rate;
    els.statEta.textContent = fmtClock(remain);
  } else {
    els.statEta.textContent = "—";
  }
}

/* ---------- file selection ---------- */

function setFile(path) {
  currentFile = path;
  currentFileName = path.split(/[\\/]/).pop();
  els.dropzone.classList.add("has-file");
  els.dzIcon.textContent = "🎬";
  els.dzTitle.textContent = "Ready to transcribe";
  els.dzReadyName.textContent = currentFileName;
  els.dzFile.textContent = "";
  els.transcribeBtn.disabled = false;
  // A new file clears the previous run's view.
  els.results.classList.add("hidden");
  setDevicePill("idle", "Idle");
}

/* ---------- drag & drop ---------- */

listen("tauri://drag-drop", (e) => {
  const paths = e.payload?.paths;
  if (paths && paths.length > 0 && !busy) setFile(paths[0]);
}).catch(() => {});

listen("tauri://drag-enter", () => {
  if (!busy) els.dropzone.classList.add("dragover");
}).catch(() => {});

["tauri://drag-leave", "tauri://drag-drop"].forEach((ev) =>
  listen(ev, () => els.dropzone.classList.remove("dragover")).catch(() => {})
);

/* ---------- browse ---------- */

function pickFile() {
  if (busy) return;
  invoke("plugin:dialog|open", {
    options: {
      multiple: false,
      filters: [
        {
          name: "Media",
          extensions: [
            "mp4", "mkv", "mov", "webm", "avi", "m4v", "mpg", "mpeg", "wmv", "flv", "ts",
            "mp3", "wav", "flac", "m4a", "aac", "ogg", "wma",
          ],
        },
      ],
    },
  }).then((picked) => {
    if (picked) setFile(picked);
  });
}
els.browseBtn.addEventListener("click", pickFile);
els.changeFile.addEventListener("click", (e) => {
  e.stopPropagation();
  pickFile();
});
els.dropzone.addEventListener("click", () => {
  if (!busy) pickFile();
});

/* ---------- transcribe ---------- */

els.transcribeBtn.addEventListener("click", async () => {
  if (!currentFile || busy) return;
  busy = true;
  runStart = performance.now();
  mediaDuration = 0;
  lastFrac = 0;
  currentDevice = null;

  els.transcribeBtn.disabled = true;
  els.transcribeBtn.textContent = "Transcribing…";
  els.langSelect.disabled = true;
  els.results.classList.add("hidden");
  els.progress.classList.remove("hidden");
  els.progressLog.innerHTML = "";
  els.progressLog.classList.add("hidden");
  els.statElapsed.textContent = "0:00";
  els.statEta.textContent = "—";
  els.statSegments.textContent = "0";
  els.statDevice.textContent = "—";
  els.progressTitle.textContent = "Starting…";
  setProgress(2, "Starting…");
  setDevicePill("busy", "Starting");

  const base = currentFile.replace(/\.[^.]+$/, "");
  const output = base + "_transcript";
  const language = els.langSelect.value;

  const unlisten = await listen("transcribe-progress", (e) => {
    const ev = e.payload;
    switch (ev.event) {
      case "downloading_model":
              currentDevice = ev.device || null;
              els.progressTitle.textContent = "Downloading model";
              els.statDevice.textContent = deviceLabel(currentDevice);
              setProgress(8, "Downloading model (one-time ~3 GB)…");
              setDevicePill(currentDevice === "cuda" ? "gpu" : "cpu", `Downloading · ${deviceLabel(currentDevice)}`);
              logLine("Downloading Whisper large-v3 model (~3 GB, first run only)…");
              break;
            case "loading_model":
              currentDevice = ev.device || null;
              els.progressTitle.textContent = "Loading model";
              els.statDevice.textContent = deviceLabel(currentDevice);
              setProgress(10, `Loading Whisper large-v3 model…`);
              setDevicePill(currentDevice === "cuda" ? "gpu" : "cpu", `Loading · ${deviceLabel(currentDevice)}`);
              logLine(`Loading model… (device: ${ev.device || "auto"})`);
              break;
      case "transcribing":
        els.progressTitle.textContent = "Transcribing";
        setProgress(15, `Transcribing…`);
        setDevicePill("busy", `Working · ${deviceLabel(currentDevice)}`);
        logLine("Transcribing…");
        break;
      case "detected_language":
        if (ev.duration_s) mediaDuration = ev.duration_s;
        logLine(`Detected language: ${ev.language} (${Math.round((ev.probability || 0) * 100)}%)`);
        break;
      case "progress":
        setProgress(progressPct(ev), `${ev.segments} segments · ${Math.round(ev.elapsed_s)}s`);
        els.statElapsed.textContent = fmtClock(ev.elapsed_s);
        els.statSegments.textContent = String(ev.segments);
        updateEta(ev);
        break;
      case "error":
        logLine(`Worker error: ${ev.message || "unknown"}`);
        break;
      case "done": {
        outputPaths = { txt: ev.txt, srt: ev.srt, vtt: ev.vtt };
        setProgress(100, `Done — ${ev.segments} segments`);
        els.progressTitle.textContent = "Complete";
        els.statElapsed.textContent = fmtClock(ev.elapsed_s);
        els.statSegments.textContent = String(ev.segments);
        els.statEta.textContent = "0:00";
        renderStats(ev);
        logLine(`Finished: ${ev.segments} segments, ${Math.round(ev.elapsed_s)}s`);
        break;
      }
      default:
        // Diagnostic events (cuda_bootstrap, …) — keep the log useful, not noisy.
        if (ev.event === "cuda_bootstrap") {
          const missing = ev.missing;
          if (missing && missing.length) logLine(`CUDA: missing ${missing.join(", ")}`);
        } else {
          logLine(`[${ev.event || "unknown"}] ${JSON.stringify(ev)}`);
        }
        break;
    }
  });

  try {
    await invoke("transcribe", {
      args: { input: currentFile, output, language: language === "auto" ? null : language },
    });
  } catch (err) {
    setProgress(100, `Error`);
    els.progressTitle.textContent = "Failed";
    setDevicePill("error", "Error");
    logLine(`ERROR: ${err}`);
    toast(`<b>Transcription failed</b> — see log for details`, "err");
    return;
  } finally {
    unlisten();
  }

  busy = false;
  els.transcribeBtn.disabled = false;
  els.transcribeBtn.textContent = "Transcribe";
  els.langSelect.disabled = false;
  setDevicePill("done", "Done");
  await showResults(activeTab);
});

/* ---------- results ---------- */

function renderStats(ev) {
  const rtf = ev.duration_s && ev.elapsed_s
    ? (ev.duration_s / ev.elapsed_s).toFixed(1)
    : null;
  const chips = [
    { label: "Duration", value: fmtDuration(ev.duration_s), cls: "" },
    { label: "Language", value: ev.language || "—", cls: "accent" },
    { label: "Segments", value: String(ev.segments), cls: "" },
    { label: "Speed", value: rtf ? `${rtf}× realtime` : "—", cls: "ok" },
    { label: "Elapsed", value: fmtDuration(ev.elapsed_s), cls: "" },
  ];
  els.statsStrip.innerHTML = chips
    .map((c) => `<span class="chip ${c.cls}">${c.label} <b>${c.value}</b></span>`)
    .join("");
  els.resultsFile.textContent = currentFileName || "";
  const clean = currentFileName?.replace(/\.[^.]+$/, "") || "transcript";
  els.copyActive.textContent = `Copy ${activeTab === "txt" ? "text" : "." + activeTab}`;
  els.openFolder.dataset.hasFile = "1";
  void clean;
}

async function showResults(tab) {
  activeTab = tab;
  if (!outputPaths[tab]) return;
  try {
    const content = await invoke("read_transcript", { path: outputPaths[tab] });
    els.transcriptView.textContent = content;
    els.transcriptView.dataset.mode = tab === "txt" ? "txt" : "raw";
    els.results.classList.remove("hidden");
    els.transcriptView.scrollTop = 0;
  } catch (err) {
    els.transcriptView.textContent = `Could not read transcript: ${err}`;
    els.transcriptView.dataset.mode = "raw";
    els.results.classList.remove("hidden");
  }
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", async () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    await showResults(btn.dataset.tab);
  });
});

/* ---------- copy / open ---------- */

async function copyText(path, label) {
  if (!path) { toast(`<b>No ${label}</b> to copy yet`, "err"); return; }
  try {
    const content = await invoke("read_transcript", { path });
    await navigator.clipboard.writeText(content);
    toast(`<b>Copied</b> ${label} to clipboard`, "ok");
  } catch (err) {
    toast(`<b>Copy failed</b>: ${err}`, "err");
  }
}

els.copyActive.addEventListener("click", async () => {
  const labels = { txt: "text", srt: ".srt", vtt: ".vtt" };
  await copyText(outputPaths[activeTab], labels[activeTab] || activeTab);
});

els.openFolder.addEventListener("click", async () => {
  if (outputPaths.txt) await invoke("open_in_folder", { path: outputPaths.txt });
});