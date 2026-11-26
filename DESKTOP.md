# DocuMind desktop app

DocuMind runs as a desktop app on macOS, Windows and Linux:

- **Tauri** shell (`frontend/src-tauri/`) shows the React UI in a native window.
- **Python backend** (`backend/`) runs hidden next to it as a *sidecar*: a single bundled
  executable that listens only on `127.0.0.1`, on a random port, and requires a random token
  on every request.
- **SQLite** (`documind.db`) stores documents, chunks, embeddings and chat history in the app-data
  folder. Settings (AI provider, NVIDIA key) are in `settings.json` in the same folder.
  - macOS: `~/Library/Application Support/com.documind.app`
  - Windows: `%APPDATA%\com.documind.app`
  - Linux: `~/.local/share/com.documind.app`
- **AI**: NVIDIA cloud (user's own key) or Ollama (local, offline), chosen in Settings.

The web version still works: when `DATABASE_URL` is set, the backend uses Postgres as before.

## One-time setup (Mac)

```bash
# Rust (needed by Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# OCR for scanned PDFs and images
brew install tesseract
# Python packages for the desktop backend (inside your venv)
pip install -r backend/requirements-desktop.txt
# Frontend packages, including the Tauri CLI
cd frontend && npm install && cd ..
```

## Run in development

```bash
python scripts/build_backend.py   # builds the backend executable (about 1-2 min); rerun after backend changes
cd frontend
npm run desktop:dev               # opens the DocuMind window with hot reload
```

## Build an installer for your own computer

```bash
python scripts/build_backend.py
cd frontend && npm run desktop:build
```

The installer is in `frontend/src-tauri/target/release/bundle/` (`dmg/` on a Mac).

## Release for macOS, Windows and Linux

A Mac can only build the Mac app, so `.github/workflows/release.yml` builds all four installers
on GitHub's machines (Apple Silicon Mac, Intel Mac, Windows, Linux) and publishes them as a
GitHub Release:

1. Push this project to GitHub.
2. Set `"version"` in `frontend/src-tauri/tauri.conf.json` (e.g. `1.0.0`) and commit.
3. `git tag v1.0.0 && git push origin v1.0.0`
4. Watch it under the repository's **Actions** tab (about 15-25 minutes).

## Download website

`website/` is a static site that always links to the newest GitHub Release.

1. In `website/index.html`, set `GITHUB_REPO = "your-username/your-repo"`.
2. Deploy with the `documind-website` service in `render.yaml` (Render > New > Blueprint), or as
   a Render Static Site with publish directory `website` and no build command.

## Signing (recommended before a public launch)

The installers are unsigned, so macOS and Windows warn on first launch (the website explains how
to continue). To remove the warnings: an Apple Developer ID (macOS signing and notarization)
and a Windows code-signing certificate, configured as GitHub secrets for `tauri-action`.
