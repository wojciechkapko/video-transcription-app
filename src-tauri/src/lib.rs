use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{Emitter, Manager, State};

struct WorkerPath(PathBuf);
struct BundledPython(Option<PathBuf>);

#[derive(Deserialize)]
struct TranscribeArgs {
    input: String,
    output: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    device: Option<String>,
}

/// Resolve the Python interpreter that has `faster-whisper` installed.
///
/// Priority:
///   1. `PYTHON_TRANSCRIBE` env var (full path to a python.exe), if it exists on disk.
///   2. The bundled portable runtime (`resources/runtime/python.exe`).
///   3. The author's dev venv (debug builds only, so it's inert in release).
///   4. `python` / `py` on PATH.
///
/// On a clean install this returns an actionable error instead of a dead path.
fn resolve_python(bundled: Option<&PathBuf>) -> Result<String, String> {
    if let Ok(p) = std::env::var("PYTHON_TRANSCRIBE") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            let pb = PathBuf::from(trimmed);
            if pb.is_file() {
                return Ok(pb.to_string_lossy().into_owned());
            }
        }
    }

    // The self-contained CPU runtime bundled as a resource.
    if let Some(p) = bundled {
        if p.is_file() {
            return Ok(p.to_string_lossy().into_owned());
        }
    }

    // Dev convenience (debug builds only): the Hermes venv python, resolved
    // from LOCALAPPDATA so no username or absolute path is baked into source.
    #[cfg(debug_assertions)]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let dev = PathBuf::from(local)
                .join("hermes")
                .join("hermes-agent")
                .join("venv")
                .join("Scripts")
                .join("python.exe");
            if dev.is_file() {
                return Ok(dev.to_string_lossy().into_owned());
            }
        }
    }

    for name in ["python", "py"] {
        let ok = Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Ok(name.to_string());
        }
    }

    Err(
        "No Python interpreter found. Install Python 3, run `pip install faster-whisper`, \
         then set PYTHON_TRANSCRIBE to the full path of that python.exe"
            .to_string(),
    )
}

/// Return the last `n` lines of `text` (for surfacing a stderr tail).
fn last_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

#[tauri::command]
async fn transcribe(
    window: tauri::Window,
    args: TranscribeArgs,
    state: State<'_, WorkerPath>,
    bundled: State<'_, BundledPython>,
) -> Result<(), String> {
    let worker = state.0.clone();
    let python = resolve_python(bundled.0.as_ref())?;
    let window2 = window.clone();

    tauri::async_runtime::spawn_blocking(move || {
        // Optional device override ("cpu" | "cuda"); passed to the worker via
        // the TRANSCRIBE_DEVICE env var it already honors. Empty/"auto" means
        // probe.
        let force_device = args
            .device
            .as_ref()
            .map(|d| d.trim().to_string())
            .filter(|d| !d.is_empty() && d != "auto");

        let mut cmd = Command::new(&python);
        cmd.arg(&worker)
            .arg(&args.input)
            .arg(&args.output)
            .arg(args.language.unwrap_or_else(|| "auto".into()))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(d) = force_device {
            cmd.env("TRANSCRIBE_DEVICE", d);
        }

        let mut child = cmd.spawn().map_err(|e| format!("failed to spawn: {e}"))?;

        // Drain stderr concurrently so a chatty traceback can't fill the pipe
        // buffer and deadlock the child (which we'd otherwise only read after
        // stdout closes). We keep the full text for the error path below.
        let stderr_rx = child.stderr.take().map(|mut se| {
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            std::thread::spawn(move || {
                let mut buf = String::new();
                let _ = std::io::Read::read_to_string(&mut se, &mut buf);
                let _ = tx.send(buf);
            });
            rx
        });

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) {
                            let _ = window2.emit("transcribe-progress", &v);
                        }
                    }
                    // A malformed line is not fatal — keep streaming the rest.
                    Err(e) => eprintln!("transcribe-worker: bad stdout line: {e}"),
                }
            }
        }

        let status = child.wait().map_err(|e| format!("wait: {e}"))?;

        let stderr_text = stderr_rx.and_then(|rx| rx.recv().ok()).unwrap_or_default();

        if !status.success() {
            let tail = last_lines(stderr_text.trim(), 25);
            let base = match status.code() {
                Some(c) => format!("transcription failed (exit code {c})"),
                None => "transcription failed (killed by signal)".to_string(),
            };
            if tail.is_empty() {
                return Err(base);
            }
            let mut msg = format!("{base}\n--- worker output ---\n{tail}");
            if tail.contains("ModuleNotFoundError") || tail.contains("No module named") {
                msg.push_str(
                    "\nHint: install the Python dependency with `pip install faster-whisper`.",
                );
            }
            return Err(msg);
        }
        Ok(())
    })
    .await
    .map_err(|e| format!("task join: {e}"))?
}

#[tauri::command]
async fn check_gpu(
    state: State<'_, WorkerPath>,
    bundled: State<'_, BundledPython>,
) -> Result<bool, String> {
    let worker = state.0.clone();
    let python = match resolve_python(bundled.0.as_ref()) {
        Ok(p) => p,
        // No interpreter at all -> no CUDA (frontend disables the GPU option).
        Err(_) => return Ok(false),
    };

    tauri::async_runtime::spawn_blocking(move || {
        let out = Command::new(&python)
            .arg(&worker)
            .arg("--probe")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .map_err(|e| format!("probe spawn failed: {e}"))?;

        // The worker may also print cuda_bootstrap lines; find probe_result.
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines().rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if v.get("event").and_then(|e| e.as_str()) == Some("probe_result") {
                    return Ok(v
                        .get("cuda_available")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false));
                }
            }
        }
        Ok(false)
    })
    .await
    .map_err(|e| format!("probe task join: {e}"))?
}

#[tauri::command]
fn read_transcript(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_in_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg("/select,")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = path;
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Settings {
    #[serde(default = "default_auto")]
    device: String,
    #[serde(default = "default_auto")]
    language: String,
}

fn default_auto() -> String {
    "auto".to_string()
}

fn settings_path() -> Option<PathBuf> {
    let base = std::env::var("LOCALAPPDATA")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".config"))
        });
    base.map(|b| b.join("whisper-transcribe").join("settings.json"))
}

#[tauri::command]
fn get_settings() -> Settings {
    if let Some(path) = settings_path() {
        if let Ok(raw) = std::fs::read_to_string(&path) {
            if let Ok(settings) = serde_json::from_str::<Settings>(&raw) {
                return settings;
            }
        }
    }
    Settings {
        device: "auto".to_string(),
        language: "auto".to_string(),
    }
}

#[tauri::command]
fn set_settings(settings: Settings) -> Result<(), String> {
    let path = settings_path().ok_or("no writable settings location")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let resource_dir = app.path().resource_dir().ok();

            // Bundled portable runtime (the self-contained CPU build), if present.
            let runtime_python = resource_dir
                .as_ref()
                .map(|d| d.join("runtime").join("python.exe"))
                .filter(|p| p.is_file());

            // Resolve the worker at runtime: bundled resource first, then the
            // source tree (covers `tauri dev`), then an explicit override.
            let resource_worker = resource_dir
                .as_ref()
                .map(|d| d.join("python").join("transcribe_worker.py"));
            let dev_worker = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("python")
                .join("transcribe_worker.py");

            let worker = if let Ok(p) = std::env::var("TRANSCRIBE_WORKER_PATH") {
                PathBuf::from(p)
            } else if let Some(p) = resource_worker.as_ref().filter(|p| p.is_file()).cloned() {
                p
            } else if dev_worker.is_file() {
                dev_worker
            } else {
                eprintln!(
                    "WARNING: transcribe_worker.py not found in resources or source tree; \
                     transcription will fail."
                );
                resource_worker.unwrap_or(dev_worker)
            };

            app.manage(WorkerPath(worker));
            app.manage(BundledPython(runtime_python));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            transcribe,
            check_gpu,
            get_settings,
            set_settings,
            read_transcript,
            open_in_folder
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
