// How the UI reaches DocuMind's backend.
//
// Desktop app (Tauri): the backend runs on this computer at 127.0.0.1 on a random port. The Tauri
// shell tells us the port and a secret token (command `backend_info`); every request carries the
// token so no other program or website can use the backend.
// Web version: VITE_API_URL (or localhost:8000), no token.

const WEB_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const isDesktop = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

let backendPromise = null;

async function waitUntilHealthy(url) {
  for (let i = 0; i < 300; i++) {
    try {
      const res = await fetch(`${url}/healthz`);
      if (res.ok) return;
    } catch {
      // not listening yet
    }
    await new Promise(r => setTimeout(r, 200));
  }
  throw new Error("DocuMind's engine did not respond.");
}

export function getBackend() {
  if (!backendPromise) {
    backendPromise = (async () => {
      if (!isDesktop) return { url: WEB_API_URL, token: null };
      const { invoke } = await import('@tauri-apps/api/core');
      const info = await invoke('backend_info');
      const url = `http://127.0.0.1:${info.port}`;
      await waitUntilHealthy(url);
      return { url, token: info.token };
    })();
    backendPromise.catch(() => { backendPromise = null; }); // allow a retry after a failure
  }
  return backendPromise;
}

export async function api(path, options = {}) {
  const backend = await getBackend();
  const headers = new Headers(options.headers || {});
  if (backend.token) headers.set('X-DocuMind-Token', backend.token);
  return fetch(`${backend.url}${path}`, { ...options, headers });
}

export async function apiJson(path, options = {}) {
  const res = await api(path, options);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  return { ok: res.ok, status: res.status, data: data || {} };
}
