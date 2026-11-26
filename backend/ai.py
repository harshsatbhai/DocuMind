"""AI providers: NVIDIA (cloud) or Ollama (local, private, offline).

Both speak the OpenAI-compatible API, so one client type serves both:
  NVIDIA  https://integrate.api.nvidia.com/v1   user's own API key   text leaves the computer
  Ollama  http://localhost:11434/v1             no key               everything stays on the computer

Embedding is the slow part of indexing, so batches are sent several at a time and retried with
backoff on rate limits (429), server errors and dropped connections.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import OpenAI

from .appdata import load_settings

NVIDIA_URL = os.environ.get("DOCUMIND_NVIDIA_URL", "https://integrate.api.nvidia.com/v1")  # override for tests
NVIDIA_EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"
NVIDIA_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

_clients: dict = {}
_clients_lock = threading.Lock()


class AIError(RuntimeError):
    """A provider problem worth showing to the user as-is."""


def _client(base_url: str, api_key: str) -> OpenAI:
    key = (base_url, api_key)
    with _clients_lock:
        if key not in _clients:
            _clients[key] = OpenAI(base_url=base_url, api_key=api_key or "none", max_retries=0, timeout=120)
        return _clients[key]


def _config(settings=None) -> dict:
    s = settings or load_settings()
    if s["provider"] == "ollama":
        base = s["ollama_url"].rstrip("/")
        return {
            "provider": "ollama", "client": _client(base + "/v1", "ollama"),
            "embed_model": s["ollama_embed_model"], "chat_model": s["ollama_chat_model"],
            "batch": 32, "parallel": 2, "extra": None, "attempts": 2, "url": base,
        }
    if not s.get("nvidia_api_key"):
        raise AIError("No NVIDIA API key set. Add your key in Settings, or switch to local AI (Ollama).")
    return {
        "provider": "nvidia", "client": _client(NVIDIA_URL, s["nvidia_api_key"]),
        "embed_model": NVIDIA_EMBED_MODEL, "chat_model": NVIDIA_CHAT_MODEL,
        "batch": 128, "parallel": 3, "extra": True, "attempts": 6, "url": NVIDIA_URL,
    }


def embed_model_id(settings=None) -> str:
    cfg = _config(settings)
    return f"{cfg['provider']}:{cfg['embed_model']}"


def _with_retry(call, what: str, cfg: dict, attempts=None):
    """Retry rate limits, timeouts and server errors with backoff. A refused connection to
    Ollama means the app isn't running, so it is reported straight away instead of retried."""
    attempts = attempts or cfg["attempts"]
    delay = 1.0
    for attempt in range(attempts):
        try:
            return call()
        except openai.APIConnectionError as e:
            if cfg["provider"] == "ollama":
                raise AIError(f"Can't reach Ollama at {cfg['url']}. Is the Ollama app running?") from e
            if attempt == attempts - 1:
                raise AIError("Can't reach NVIDIA's API. Check your internet connection.") from e
            time.sleep(delay)
            delay = min(delay * 2, 20)
        except (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError) as e:
            if attempt == attempts - 1:
                raise AIError(f"{what} failed after {attempts} tries: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 20)
        except openai.AuthenticationError as e:
            raise AIError("The NVIDIA API key was rejected. Check it in Settings.") from e
        except openai.NotFoundError as e:
            raise AIError(f"Model not found ({e}). For Ollama, run `ollama pull <model>` first.") from e


def _prefix(cfg, texts, kind):
    # nomic-embed-text was trained with task prefixes; other Ollama models take plain text.
    if cfg["provider"] == "ollama" and "nomic-embed" in cfg["embed_model"]:
        tag = "search_query: " if kind == "query" else "search_document: "
        return [tag + t for t in texts]
    return texts


def embed(texts: list[str], kind: str = "passage", progress=None, settings=None, attempts=None) -> list[list[float]]:
    """Embed texts ("passage" for document chunks, "query" for questions)."""
    cfg = _config(settings)
    texts = _prefix(cfg, texts, kind)
    batches = [texts[i:i + cfg["batch"]] for i in range(0, len(texts), cfg["batch"])]
    results: list = [None] * len(batches)
    done = [0]
    lock = threading.Lock()

    def run(i):
        kwargs = {"model": cfg["embed_model"], "input": batches[i], "encoding_format": "float"}
        if cfg["extra"]:
            kwargs["extra_body"] = {"input_type": kind, "modality": "text", "truncate": "END"}
        response = _with_retry(lambda: cfg["client"].embeddings.create(**kwargs), "Embedding", cfg, attempts)
        results[i] = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        with lock:
            done[0] += len(batches[i])
            if progress:
                progress("embedding", done[0], len(texts))

    if len(batches) <= 1:
        for i in range(len(batches)):
            run(i)
    else:
        with ThreadPoolExecutor(max_workers=cfg["parallel"]) as pool:
            list(pool.map(run, range(len(batches))))
    return [v for batch in results for v in batch]


def chat(system_prompt: str, user_prompt: str, settings=None) -> str:
    cfg = _config(settings)
    completion = _with_retry(lambda: cfg["client"].chat.completions.create(
        model=cfg["chat_model"],
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1, max_tokens=2048), "Answering", cfg)
    return completion.choices[0].message.content or ""


def test_connection(settings: dict) -> dict:
    """Check the chosen provider end to end with a tiny embedding call."""
    try:
        cfg = _config(settings)
        vec = embed(["DocuMind connection test"], kind="query", settings=settings, attempts=1)[0]
        info = {"ok": True, "provider": cfg["provider"], "embed_model": cfg["embed_model"],
                "chat_model": cfg["chat_model"], "dimensions": len(vec)}
        if cfg["provider"] == "ollama":
            names = [m.id for m in cfg["client"].models.list().data]
            info["installed_models"] = names
            missing = [m for m in (cfg["chat_model"],) if not any(n == m or n.startswith(m + ":") for n in names)]
            if missing:
                info["ok"] = False
                info["message"] = f"Chat model not installed. Run: ollama pull {missing[0]}"
        return info
    except AIError as e:
        return {"ok": False, "message": str(e)}
    except Exception as e:  # e.g. Ollama not running
        hint = " Is Ollama running? Start the Ollama app, then try again." if settings.get("provider") == "ollama" else ""
        return {"ok": False, "message": f"{type(e).__name__}: {e}.{hint}"}
