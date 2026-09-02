use serde::Deserialize;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{Emitter, Manager, State};

struct WorkerPath(PathBuf);

#[derive(Deserialize)]
struct TranscribeArgs {
    input: String,
    output: String,
    #[serde(default)]
    language: Option<String>,
}

/// Resolve the Python interpreter that has `faster-whisper` installed.
///
/// Priority:
///   1. `PYTHON_TRANSCRIBE` env var (full path to a python.exe), if it exists on disk.
///   2. The author's dev venv (existence-checked, so it's inert on other machines).
///   3. `python` / `python3` / `py` on PATH.
///
/// On a clean install this returns an actionable error instead of a dead path.
fn resolve_python() -> Result<String, String> {
    if let Ok(p) = std::env::var("PYTHON_TRANSCRIBE") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            let pb = PathBuf::from(trimmed);
            if pb.is_file() {
                return Ok(pb.to_string_lossy().into_owned());
            }
        }
    }

    // Dev-machine convenience: compiled only into debug builds, so the
    // personal venv path (and username) never ships in a release binary.
    #[cfg(debug_assertions)]
    {
        let dev = PathBuf::from(
            r"C:\Users\wojci\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe",
        );
        if dev.is_file() {
            return Ok(dev.to_string_lossy().into_owned());
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
) -> Result<(), String> {
    let worker = state.0.clone();
    let python = resolve_python()?;
    let window2 = window.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = Command::new(&python);
        cmd.arg(&worker)
            .arg(&args.input)
            .arg(&args.output)
            .arg(args.language.unwrap_or_else(|| "auto".into()))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // Resolve the worker at runtime: bundled resource first, then the
            // source tree (covers `tauri dev`), then an explicit override.
            let resource_worker = app
                .path()
                .resource_dir()
                .map(|d| d.join("python").join("transcribe_worker.py"))
                .ok();
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            transcribe,
            read_transcript,
            open_in_folder
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
