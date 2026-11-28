// DocuMind desktop shell.
//
// On start it launches the bundled Python backend (the "sidecar") with the app-data folder, waits
// for the backend's {"event":"ready","port":..,"token":..} line, and hands port + token to the UI
// through the `backend_info` command. When the app quits, the backend is stopped.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use std::time::Duration;

use serde_json::Value;
use tauri::{Manager, RunEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
struct Backend {
    info: Mutex<Option<Value>>,
    error: Mutex<Option<String>>,
    child: Mutex<Option<CommandChild>>,
    /// Last lines the backend wrote to stderr, shown with the error if it stops.
    log: Mutex<Vec<String>>,
}

/// Port and token of the local backend. Waits up to 90 s, because the first start of a
/// single-file backend unpacks itself, which can take a while on slower machines.
#[tauri::command]
async fn backend_info(state: tauri::State<'_, Backend>) -> Result<Value, String> {
    for _ in 0..900 {
        if let Some(info) = state.info.lock().unwrap().clone() {
            return Ok(info);
        }
        if let Some(err) = state.error.lock().unwrap().clone() {
            return Err(err);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("DocuMind's engine did not start within 90 seconds.".into())
}

fn start_backend(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = app.path().app_data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    let (mut rx, child) = app
        .shell()
        .sidecar("documind-backend")?
        .args(["--data-dir".to_string(), data_dir.to_string_lossy().to_string()])
        .spawn()?;
    *app.state::<Backend>().child.lock().unwrap() = Some(child);

    let handle = app.handle().clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let text = String::from_utf8_lossy(&line);
                    match serde_json::from_str::<Value>(text.trim()) {
                        Ok(v) if v.get("event").and_then(|e| e.as_str()) == Some("ready") => {
                            *handle.state::<Backend>().info.lock().unwrap() = Some(v);
                        }
                        _ => println!("[backend] {}", text.trim_end()),
                    }
                }
                CommandEvent::Stderr(line) => {
                    let text = String::from_utf8_lossy(&line).trim_end().to_string();
                    eprintln!("[backend] {}", text);
                    let state = handle.state::<Backend>();
                    let mut log = state.log.lock().unwrap();
                    log.push(text);
                    if log.len() > 12 {
                        log.remove(0);
                    }
                }
                CommandEvent::Error(err) => {
                    *handle.state::<Backend>().error.lock().unwrap() = Some(err);
                }
                CommandEvent::Terminated(payload) => {
                    let state = handle.state::<Backend>();
                    let log = state.log.lock().unwrap().join("\n");
                    let mut msg = match payload.code {
                        Some(code) => format!("DocuMind's engine stopped (exit code {code})."),
                        None => "DocuMind's engine stopped.".to_string(),
                    };
                    if !log.trim().is_empty() {
                        msg.push_str("\n\nDetails:\n");
                        msg.push_str(log.trim());
                    }
                    *state.error.lock().unwrap() = Some(msg);
                }
                _ => {}
            }
        }
    });
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Backend::default())
        .invoke_handler(tauri::generate_handler![backend_info])
        .setup(|app| {
            if let Err(err) = start_backend(app) {
                *app.state::<Backend>().error.lock().unwrap() =
                    Some(format!("Could not start DocuMind's engine: {err}"));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building DocuMind")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(child) = app.state::<Backend>().child.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
