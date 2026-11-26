"""Build DocuMind's Python backend into one executable and place it where Tauri expects it.

    python scripts/build_backend.py

Output: frontend/src-tauri/binaries/documind-backend-<target triple>[.exe]
Run it on each OS you ship for (a Mac build makes the Mac binary, Windows makes the .exe).
Needs: pip install -r backend/requirements-desktop.txt
"""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "frontend" / "src-tauri" / "binaries"


def target_triple() -> str:
    if os.environ.get("TARGET_TRIPLE"):  # set by CI when building for another CPU (e.g. Intel Mac)
        return os.environ["TARGET_TRIPLE"]
    try:
        out = subprocess.run(["rustc", "--print", "host-tuple"], capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    try:
        out = subprocess.run(["rustc", "-Vv"], capture_output=True, text=True, check=True).stdout
        return next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("host:"))
    except Exception:
        sys.exit("Rust is required (it provides the target triple Tauri uses). Install it from https://rustup.rs")


def main():
    triple = target_triple()
    exe = ".exe" if platform.system() == "Windows" else ""
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
        "--name", "documind-backend",
        "--paths", str(ROOT),
        "--collect-submodules", "backend",
        "--collect-submodules", "uvicorn",
        "--collect-submodules", "langchain_text_splitters",
        "--collect-all", "pypdfium2",
        "--collect-all", "pypdfium2_raw",
        "--collect-data", "docx",
        "--collect-data", "pptx",
        # Postgres support is for the web deployment only; keep it out of the desktop app.
        "--exclude-module", "psycopg2",
        "--exclude-module", "pgvector",
        "--exclude-module", "tkinter",
        "--distpath", str(ROOT / "build" / "dist"),
        "--workpath", str(ROOT / "build" / "work"),
        "--specpath", str(ROOT / "build"),
        str(ROOT / "desktop_backend.py"),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / f"documind-backend-{triple}{exe}"
    shutil.copy2(ROOT / "build" / "dist" / f"documind-backend{exe}", target)
    os.chmod(target, 0o755)
    print(f"\nBackend ready: {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
