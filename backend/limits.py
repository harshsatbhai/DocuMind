"""Upload limits for DocuMind.

Limits apply only when deployed. Render always sets RENDER=true on its services, so they switch
on automatically there and stay off on a local machine. Override with APPLY_UPLOAD_LIMITS=true
(e.g. to test the limits locally, or on another host) or APPLY_UPLOAD_LIMITS=false.

Defaults are sized for Render's free plan (512 MB RAM web service, 1 GB Postgres).
Each value can be overridden with an environment variable of the same name, e.g. on a
2 GB paid instance: MAX_FILE_MB=50, MAX_TOTAL_MB=100, MAX_OCR_PAGES=100.
The frontend reads these from GET /limits, so changing them here updates both sides.
"""
import os


def _num(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


def _flag(name: str):
    value = os.environ.get(name, "").strip().lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return None


ON_RENDER = os.environ.get("RENDER", "").strip().lower() == "true"
_override = _flag("APPLY_UPLOAD_LIMITS")
ENFORCED = _override if _override is not None else ON_RENDER

MAX_FILE_MB = _num("MAX_FILE_MB", 10)        # any single document
MAX_IMAGE_MB = _num("MAX_IMAGE_MB", 10)      # any single image (OCR)
MAX_FILES = int(_num("MAX_FILES", 5))        # files per upload request
MAX_TOTAL_MB = _num("MAX_TOTAL_MB", 25)      # all files in one upload request
MAX_OCR_PAGES = int(_num("MAX_OCR_PAGES", 20))  # scanned PDF pages that need OCR
MAX_CHUNKS = int(_num("MAX_CHUNKS", 1500))   # chunks per file (~1.5 MB of text)

MB = 1024 * 1024
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def max_bytes_for(filename: str):
    """Byte limit for one file, or None when limits are off."""
    if not ENFORCED:
        return None
    limit = MAX_IMAGE_MB if filename.lower().endswith(IMAGE_EXTS) else MAX_FILE_MB
    return int(limit * MB)


def as_dict() -> dict:
    return {
        "enforced": ENFORCED,
        "max_file_mb": MAX_FILE_MB,
        "max_image_mb": MAX_IMAGE_MB,
        "max_files": MAX_FILES,
        "max_total_mb": MAX_TOTAL_MB,
        "max_ocr_pages": MAX_OCR_PAGES,
        "max_chunks": MAX_CHUNKS,
    }


def describe() -> str:
    if not ENFORCED:
        why = "APPLY_UPLOAD_LIMITS=false" if _override is False else "not running on Render"
        return f"Upload limits OFF ({why})"
    why = "APPLY_UPLOAD_LIMITS=true" if _override else "running on Render"
    return (f"Upload limits ON ({why}): {MAX_FILE_MB:g} MB/file, {MAX_IMAGE_MB:g} MB/image, "
            f"{MAX_FILES} files and {MAX_TOTAL_MB:g} MB per upload, {MAX_OCR_PAGES} OCR pages, "
            f"{MAX_CHUNKS} chunks/file")


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / MB:.1f} MB"


def fmt_limit(mb: float) -> str:
    return f"{mb:g} MB"
