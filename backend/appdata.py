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


def total_ram_gb() -> float:
    """Physical memory of this computer in GB (DOCUMIND_RAM_GB overrides it, for testing)."""
    if os.environ.get("DOCUMIND_RAM_GB"):
        return float(os.environ["DOCUMIND_RAM_GB"])
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 1024 ** 3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3  # macOS, Linux
    except Exception:
        return 0.0  # unknown: treat as a normal computer


RAM_GB = total_ram_gb()
LOW_MEMORY = 0 < RAM_GB <= 9  # 8 GB computers (they report a little under 8)
CONTEXT_CHOICES = ("8192", "16384")

DEFAULT_SETTINGS = {
    "provider": "nvidia",                     # "nvidia" (cloud) or "ollama" (local)
    "nvidia_api_key": "",
    "ollama_url": "http://localhost:11434",
    "ollama_embed_model": "nomic-embed-text",
    # 8 GB computers: the 3B model with an 8K window fits in memory; 8B with 16K does not.
    "ollama_chat_model": "llama3.2:3b" if LOW_MEMORY else "llama3.1:8b",
    "ollama_context": "8192" if LOW_MEMORY else "16384",   # tokens Ollama keeps in memory per answer
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
        if current["ollama_context"] not in CONTEXT_CHOICES:
            current["ollama_context"] = DEFAULT_SETTINGS["ollama_context"]
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
    s["ram_gb"] = round(RAM_GB, 1)
    s["low_memory"] = LOW_MEMORY
    s["recommended"] = {"ollama_chat_model": DEFAULT_SETTINGS["ollama_chat_model"],
                        "ollama_context": DEFAULT_SETTINGS["ollama_context"]}
    return s


def delete_settings():
    """Remove saved settings, including the NVIDIA API key; defaults apply again."""
    with _lock:
        for name in ("settings.json", "settings.tmp"):
            try:
                (data_dir() / name).unlink()
            except FileNotFoundError:
                pass
