"""AI providers: NVIDIA (cloud) or Ollama (local, private, offline).

Both speak the OpenAI-compatible API, so one client type serves both:
  NVIDIA  https://integrate.api.nvidia.com/v1   user's own API key   text leaves the computer
  Ollama  http://localhost:11434/v1             no key               everything stays on the computer

Embedding is the slow part of indexing, so batches are sent several at a time and retried with
backoff on rate limits (429), server errors and dropped connections.
"""
import base64
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
from openai import OpenAI

from . import appdata
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
            "context": int(s.get("ollama_context") or 16384),
            "vision_model": None if s.get("ollama_vision_model", "off") == "off" else s.get("ollama_vision_model"),
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


def _ollama_chat(cfg, system_prompt: str, user_prompt: str) -> str:
    """Ollama's own chat API, so the context window (num_ctx) can be set on every request.
    Its OpenAI-compatible API can't, and Ollama's small default window silently cuts the
    document excerpts out of long prompts. Everything stays on this computer (localhost)."""
    if cfg.get("vision_model"):
        _make_room(cfg, cfg["chat_model"], cfg["context"], cfg["vision_model"], VISION_CONTEXT)
    body = {
        "model": cfg["chat_model"],
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": cfg["context"], "num_predict": 2048},
    }
    try:
        response = httpx.post(f"{cfg['url']}/api/chat", json=body, timeout=600)
    except httpx.ConnectError as e:
        raise AIError(f"Can't reach Ollama at {cfg['url']}. Is the Ollama app running?") from e
    except httpx.TimeoutException as e:
        raise AIError("Ollama took more than 10 minutes to answer. Try a smaller model or the 8K context window in Settings.") from e
    if response.status_code == 404:
        raise AIError(f"Model {cfg['chat_model']} isn't installed. Run: ollama pull {cfg['chat_model']}")
    if response.status_code >= 400:
        detail = response.text[:300]
        if "memory" in detail.lower():
            raise AIError("Not enough memory for this model and context window. Choose Llama 3.2 3B or the 8K window in Settings.")
        raise AIError(f"Ollama error {response.status_code}: {detail}")
    return response.json().get("message", {}).get("content", "")


def chat(system_prompt: str, user_prompt: str, settings=None) -> str:
    cfg = _config(settings)
    if cfg["provider"] == "ollama":
        return _ollama_chat(cfg, system_prompt, user_prompt)
    completion = _with_retry(lambda: cfg["client"].chat.completions.create(
        model=cfg["chat_model"],
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1, max_tokens=2048), "Answering", cfg)
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Image model (Ollama only): photos without text
# ---------------------------------------------------------------------------
# Memory is the limit on most laptops, so DocuMind runs one big model at a time. Before the image
# model starts, the chat model is unloaded from memory if the two don't fit together (and the
# other way round), using Ollama's keep_alive=0. Requests to the image model are made one at a
# time, so indexing several photos never loads it twice.
# tokens: up to 3 photos (~800 each) + question + excerpts. 8 GB computers: 1 photo and a 4K window,
# so the image model fits in the memory left next to the system (about 4.4 GB).
VISION_CONTEXT = 4096 if appdata.LOW_MEMORY else 8192
VISION_KEEP_ALIVE = "2m"       # stay loaded between photos of one upload, then free the memory
_vision_lock = threading.Lock()

# Approximate memory use (GB): 4-bit weights, cache per 1K tokens of context.
_MODEL_MEMORY = {
    "llama3.2:3b": (2.0, 0.11), "llama3.1:8b": (4.9, 0.125),
    "qwen3-vl:4b": (3.3, 0.14), "qwen3-vl:8b": (6.1, 0.14),
}
SYSTEM_RESERVE_GB = 3.5        # the operating system, DocuMind and a browser


def model_memory_gb(model: str, context: int) -> float:
    weights, per_1k = _MODEL_MEMORY.get(model, (6.0, 0.13))
    return weights + per_1k * context / 1024 + 0.5


def _fits_together(a: str, a_ctx: int, b: str, b_ctx: int) -> bool:
    ram = appdata.RAM_GB
    if ram <= 0:
        return True  # unknown computer: leave it to Ollama
    return model_memory_gb(a, a_ctx) + model_memory_gb(b, b_ctx) <= ram - SYSTEM_RESERVE_GB


def _unload(cfg, model: str):
    """Ask Ollama to free a model's memory now. Harmless if it isn't loaded or installed."""
    try:
        httpx.post(f"{cfg['url']}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


def _make_room(cfg, using: str, using_ctx: int, other: str, other_ctx: int):
    if other and other != using and not _fits_together(using, using_ctx, other, other_ctx):
        _unload(cfg, other)


def vision_model(settings=None):
    """The image model to use, or None (NVIDIA cloud, or image analysis turned off)."""
    try:
        cfg = _config(settings)
    except AIError:
        return None
    return cfg.get("vision_model") if cfg["provider"] == "ollama" else None


def _ollama_vision(cfg, messages, num_predict: int) -> str:
    model = cfg["vision_model"]
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "keep_alive": VISION_KEEP_ALIVE,
            "options": {"temperature": 0.1, "num_ctx": VISION_CONTEXT, "num_predict": num_predict}}
    with _vision_lock:
        _make_room(cfg, model, VISION_CONTEXT, cfg["chat_model"], cfg["context"])
        try:
            response = httpx.post(f"{cfg['url']}/api/chat", json=body, timeout=600)
            if response.status_code == 400 and "think" in response.text.lower():
                body.pop("think")  # a model without a thinking switch
                response = httpx.post(f"{cfg['url']}/api/chat", json=body, timeout=600)
        except httpx.ConnectError as e:
            raise AIError(f"Can't reach Ollama at {cfg['url']}. Is the Ollama app running?") from e
        except httpx.TimeoutException as e:
            raise AIError("The image model took more than 10 minutes. Try Qwen3-VL 4B in Settings.") from e
    if response.status_code == 404:
        raise AIError(f"Image model {model} isn't installed. Run: ollama pull {model}")
    if response.status_code >= 400:
        detail = response.text[:300]
        if "memory" in detail.lower():
            raise AIError("Not enough memory for the image model. Choose Qwen3-VL 4B in Settings and close other apps.")
        raise AIError(f"Ollama error {response.status_code}: {detail}")
    return (response.json().get("message", {}).get("content") or "").strip()


DESCRIBE_PROMPT = """Describe this image so that someone can later find it and ask questions about it without seeing it.
Write plain sentences, no markdown, no lists of bullet symbols. Cover, where they apply:
what the image shows overall (product photo, chart, diagram, scene, screenshot, document photo);
each main object or product: type, colour, material, shape, size cues, parts and features;
brand names, labels, numbers or any other text that is visible, copied exactly;
condition: new, worn, damaged, broken, dirty, and where;
how many of each thing there are;
for charts and diagrams: the title, axes, values, trends and what the diagram explains;
the setting and background.
Only describe what is actually visible. Do not guess prices, exact model numbers or anything you cannot see."""


def describe_image(jpeg: bytes, filename: str, settings=None) -> str:
    """One description per photo, written when it is uploaded; it is embedded and searched like text."""
    cfg = _config(settings)
    if cfg["provider"] != "ollama" or not cfg.get("vision_model"):
        raise AIError("Image analysis needs local AI (Ollama) with an image model chosen in Settings.")
    messages = [{"role": "user", "content": f"File name: {filename}\n\n{DESCRIBE_PROMPT}",
                 "images": [base64.b64encode(jpeg).decode()]}]
    text = _ollama_vision(cfg, messages, num_predict=600)
    if not text:
        raise AIError("The image model returned an empty description. Try again.")
    return text


def answer_with_images(system_prompt: str, user_prompt: str, images: list, settings=None) -> str:
    """Answer a question that is about photos: the image model looks at them again (up to 3),
    together with the text excerpts. Used instead of the chat model, so one model runs at a time."""
    cfg = _config(settings)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt,
                 "images": [base64.b64encode(data).decode() for _, data in images]}]
    return _ollama_vision(cfg, messages, num_predict=1500)


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
            vm = cfg.get("vision_model")
            if vm:
                info["vision_model"] = vm
                info["vision_ready"] = any(n == vm or n.startswith(vm + ":") for n in names)
        return info
    except AIError as e:
        return {"ok": False, "message": str(e)}
    except Exception as e:  # e.g. Ollama not running
        hint = " Is Ollama running? Start the Ollama app, then try again." if settings.get("provider") == "ollama" else ""
        return {"ok": False, "message": f"{type(e).__name__}: {e}.{hint}"}
