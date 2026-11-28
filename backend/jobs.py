"""Background indexing.

POST /upload only reads the files and returns job ids straight away; the work (parse -> embed ->
save) runs here, and the UI polls GET /jobs for real progress ("OCR page 212 of 500").
Two files are indexed at a time; within a file, OCR and embedding are already parallel.
"""
import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import ai, storage
from .parser import NoTextInImage, _header, image_for_vision, parse_file_to_chunks

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="index")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

STAGE_LABELS = {
    "queued": "Waiting",
    "reading": "Reading page {done} of {total}",
    "ocr": "OCR page {done} of {total}",
    "parsing": "Extracting text",
    "describing": "Describing the photo (local image model)",
    "embedding": "Embedding {done} of {total} chunks",
    "saving": "Saving",
    "reused": "Already indexed, reusing",
    "done": "Done",
    "failed": "Failed",
}


def _update(job_id, **fields):
    with _lock:
        job = _jobs[job_id]
        job.update(fields)
        label = STAGE_LABELS.get(job["stage"], job["stage"])
        job["label"] = label.format(done=job.get("done", 0), total=job.get("total", 0))


def submit(session_id: str, filename: str, data: bytes) -> dict:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "session_id": session_id, "filename": filename,
                         "status": "running", "stage": "queued", "label": "Waiting", "done": 0, "total": 0,
                         "doc": None, "error": None, "started": time.time()}
    _pool.submit(_run, job_id, session_id, filename, data)
    return dict(_jobs[job_id])


def get(session_id: str, job_ids=None) -> list[dict]:
    with _lock:
        jobs = [dict(j) for j in _jobs.values() if j["session_id"] == session_id
                and (not job_ids or j["job_id"] in job_ids)]
        # forget finished jobs after 10 minutes
        cutoff = time.time() - 600
        for jid in [k for k, j in _jobs.items() if j["status"] != "running" and j["started"] < cutoff]:
            del _jobs[jid]
    return jobs


def _run(job_id, session_id, filename, data):
    doc_id = str(uuid.uuid4())
    try:
        sha = hashlib.sha256(data).hexdigest()
        model = ai.embed_model_id()
        reused = storage.copy_cached_document(sha, model, session_id, doc_id, filename)
        if reused is not None:
            _update(job_id, stage="reused")
            chunks_count = reused
        else:
            _update(job_id, stage="parsing")
            try:
                chunks = parse_file_to_chunks(
                    data, filename, progress=lambda stage, done, total: _update(job_id, stage=stage, done=done, total=total))
            except NoTextInImage as e:
                chunks = _describe_photo(job_id, filename, data, sha, e)
            _update(job_id, stage="embedding", done=0, total=len(chunks))
            embeddings = ai.embed(chunks, kind="passage",
                                  progress=lambda stage, done, total: _update(job_id, stage=stage, done=done, total=total))
            _update(job_id, stage="saving")
            storage.insert_document(session_id, doc_id, filename, sha, model, chunks, embeddings)
            chunks_count = len(chunks)
        _update(job_id, status="done", stage="done",
                doc={"doc_id": doc_id, "filename": filename, "chunks_count": chunks_count})
    except Exception as e:
        message = str(e)
        if filename.lower().endswith(IMAGE_EXTS) and "Could not extract any text" in message:
            message = "No readable text found."
        print(f"Failed to process {filename}: {e}")
        _update(job_id, status="failed", stage="failed", error=message)


def _describe_photo(job_id, filename, data, sha, reason: NoTextInImage) -> list[str]:
    """A photo without text (or no Tesseract): the local image model describes it once, and the
    description is indexed like text. A resized copy is kept for questions about the photo."""
    if not ai.vision_model():
        if reason.text.strip():
            # A few words only (a one-word sign, a label): index them, as before image analysis existed.
            return [_header(filename, "OCR text from image") + "\n" + reason.text.strip()]
        if reason.ocr_missing:
            raise ValueError(str(reason))
        if ai.load_settings()["provider"] != "ollama":
            raise ValueError("No readable text found. To analyze photos without text, switch to local AI "
                             "(Ollama) in Settings and choose an image model.")
        raise ValueError("No readable text found. To analyze photos without text, choose an image model in Settings.")
    _update(job_id, stage="describing")
    jpeg = image_for_vision(data)
    description = ai.describe_image(jpeg, filename)
    storage.save_image(sha, jpeg)
    return [_header(filename, "image description") + "\n" + description]


def any_running() -> bool:
    with _lock:
        return any(j["status"] == "running" for j in _jobs.values())
