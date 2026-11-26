"""DocuMind desktop backend launcher (the Tauri "sidecar").

Started by the desktop app with --data-dir <app data folder>. It:
  1. picks a free port on 127.0.0.1 (never reachable from other computers),
  2. generates a random token that every request must carry,
  3. prints one JSON line {"event": "ready", "port": ..., "token": ...} for the Tauri shell,
  4. exits when the app closes (stdin closes) so no background process is left behind.

Run it directly for development:  python desktop_backend.py --data-dir ./devdata
Built into a single executable with:  python scripts/build_backend.py
"""
import argparse
import json
import multiprocessing
import os
import secrets
import socket
import sys
import threading


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _exit_when_parent_closes():
    try:
        while sys.stdin.read(1):
            pass
    except Exception:
        pass
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-stdin-watch", action="store_true", help="for running from a terminal")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["DOCUMIND_DATA_DIR"] = args.data_dir
    os.environ["DOCUMIND_TOKEN"] = secrets.token_urlsafe(32)
    # The desktop app always uses local SQLite. Set (not remove) the variable, so a .env file with
    # a Postgres DATABASE_URL can't switch it back: load_dotenv never overrides existing variables.
    os.environ["DATABASE_URL"] = ""
    port = args.port or _free_port()

    import uvicorn
    from backend.main import app

    print(json.dumps({"event": "ready", "port": port, "token": os.environ["DOCUMIND_TOKEN"]}), flush=True)
    if not args.no_stdin_watch:
        threading.Thread(target=_exit_when_parent_closes, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed by PyInstaller-built executables
    main()
