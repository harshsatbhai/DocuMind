"""Where DocuMind keeps its data, and the user's settings.

Desktop app: the Tauri shell passes --data-dir (the OS app-data folder), which the launcher puts in
DOCUMIND_DATA_DIR. Everything DocuMind stores (SQLite database, settings) lives there, on the
user's own computer:
    macOS    ~/Library/Application Support/com.documind.app
    Windows  %APPDATA%\\com.documind.app
    Linux    ~/.local/share/com.documind.app
"""
import json
import os
import sys
import threading
from pathlib import Path

DESKTOP = bool(os.environ.get("DOCUMIND_TOKEN"))  # set by the desktop launcher only


def data_dir() -> Path:
    env = os.environ.get("DOCUMIND_DATA_DIR")
    if env:
        path = Path(env)
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "com.documind.app"
    elif sys.platform.startswith("win"):
        path = Path(os.environ.get("APPDATA", Path.home())) / "com.documind.app"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "com.documind.app"
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_SETTINGS = {
    "provider": "nvidia",                     # "nvidia" (cloud) or "ollama" (local)
    "nvidia_api_key": "",
    "ollama_url": "http://localhost:11434",
    "ollama_embed_model": "nomic-embed-text",
    "ollama_chat_model": "llama3.1:8b",
}
_lock = threading.Lock()


def _settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        settings.update(json.loads(_settings_path().read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if not settings.get("nvidia_api_key"):
        settings["nvidia_api_key"] = os.environ.get("NVIDIA_API_KEY", "")
    return settings


def save_settings(changes: dict) -> dict:
    with _lock:
        current = load_settings()
        for key in DEFAULT_SETTINGS:
            if key in changes and changes[key] is not None:
                current[key] = str(changes[key]).strip()
        if current["provider"] not in ("nvidia", "ollama"):
            current["provider"] = "nvidia"
        path = _settings_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2))
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)  # the API key is only readable by this user
        except OSError:
            pass
        return current


def public_settings() -> dict:
    """Settings safe to send to the UI: the key itself is never returned, only whether one is set."""
    s = load_settings()
    key = s.pop("nvidia_api_key", "")
    s["nvidia_key_set"] = bool(key)
    s["nvidia_key_hint"] = f"…{key[-4:]}" if len(key) >= 8 else ""
    return s
