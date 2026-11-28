"""File parsing and chunking for DocuMind.

Every file type is extracted in a way that keeps its natural structure, then chunked along that
structure. Each chunk starts with a one-line context header, for example
    [Source: billing.py | python | lines 40-72 | def calculate_discount]
    [Source: sales.xlsx | Sheet: Q3 | rows 2-41 | columns: Region, Revenue]
    [Source: policy.docx | Section: Leave > Maternity leave]
so both the embedding and the LLM know where the text came from.

Strategies
  code files      Python: split on real function/class boundaries (ast), with decorators and
                  leading comments kept; large classes split per method. Other languages:
                  language-aware separators. Every chunk carries its line range and symbol names.
  config files    JSON pretty-printed, then split on line boundaries (never mid-line).
  PDF             pdfium text layer; pages without text are OCR'd on all CPU cores; chunks
                  never span pages.
  DOCX            paragraphs and tables in document order; headings build a section path.
  XLSX/CSV/TSV    every row rendered as "Column: value" records; header row repeated in every
                  chunk; rows never split; plus one summary chunk per sheet (row count, totals).
  PPTX            one chunk per slide, including tables, grouped shapes and speaker notes.
  TXT/MD/HTML     paragraph packing; Markdown and HTML headings build a section path.
  LOG             line packing (a log line is never split).
  images          OCR text.
Unknown or binary files are rejected with a clear error instead of being indexed as junk.

parse_file_to_chunks(file_bytes, filename) -> list[str] keeps its original signature.
"""
import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html.parser import HTMLParser

import docx
import openpyxl
import pandas as pd
import pptx
import pypdfium2 as pdfium
import pytesseract
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from lxml import etree
from PIL import Image, ImageOps

from . import limits

# ---------------------------------------------------------------------------
# Sizes (characters). Prose is smaller for precise retrieval; code and tables keep whole units.
# ---------------------------------------------------------------------------
PROSE_SIZE, PROSE_OVERLAP = 1200, 150
CODE_SIZE, CODE_OVERLAP = 2000, 200
TABLE_SIZE = 1500

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

CODE_LANGS = {
    ".py": "python", ".pyw": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust", ".swift": "swift",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".php": "php", ".rb": "ruby", ".lua": "lua", ".pl": "perl",
    ".r": "r", ".ex": "elixir", ".exs": "elixir", ".hs": "haskell", ".sol": "sol",
    ".proto": "proto", ".ps1": "powershell",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".sql": "sql", ".css": "css", ".scss": "css",
}
CONFIG_EXTS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".properties"}
CODE_FILENAMES = {"dockerfile": "shell", "makefile": "shell"}

# Extra separators for languages LangChain has no preset for
CUSTOM_SEPARATORS = {
    "shell": ["\nfunction ", "\n\n", "\n", " ", ""],
    "sql": ["\n;\n", ";\n", "\n\n", "\n", " ", ""],
    "css": ["\n}\n", "\n\n", "\n", " ", ""],
}

SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|internal\s+)?"
    r"(?:static\s+)?(?:async\s+)?(?:abstract\s+)?"
    r"(?:function\*?|class|def|func|fn|interface|struct|enum|trait|impl|module|object|type|record)\s+"
    r"([A-Za-z_$][\w$]*)"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
    r"|^\s*(?:CREATE|ALTER)\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|FUNCTION|PROCEDURE|INDEX|TRIGGER)\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"`]+)",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _header(filename: str, *parts: str) -> str:
    bits = [f"Source: {filename}"] + [p for p in parts if p]
    return "[" + " | ".join(bits) + "]"


def _prose_splitter(size=PROSE_SIZE, overlap=PROSE_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def _decode_text(file_bytes: bytes, filename: str) -> str:
    """Decode a text file, handling BOMs and legacy encodings; reject binary files."""
    if file_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        return file_bytes.decode("utf-16")
    if file_bytes.startswith(b"\xef\xbb\xbf"):
        return file_bytes[3:].decode("utf-8", errors="replace")
    if b"\x00" in file_bytes[:8192]:
        raise ValueError(f"{filename} looks like a binary file, not text, so it can't be analyzed.")
    for enc in ("utf-8", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1")


def _pack_blocks(blocks, filename, size=PROSE_SIZE, label="Section"):
    """Pack (section_path, text) blocks into chunks without crossing more than needed.

    Consecutive small blocks share a chunk; a block bigger than `size` is split with the prose
    splitter. The header names the section(s) the chunk covers.
    """
    chunks, buf, buf_secs, buf_len = [], [], [], 0
    splitter = _prose_splitter(size)

    def flush():
        nonlocal buf, buf_secs, buf_len
        if buf:
            secs = [s for i, s in enumerate(buf_secs) if s and s not in buf_secs[:i]]
            sec = secs[0] if len(secs) == 1 else (f"{secs[0]} … {secs[-1]}" if secs else "")
            chunks.append(_header(filename, f"{label}: {sec}" if sec else "") + "\n" + "\n\n".join(buf))
        buf, buf_secs, buf_len = [], [], 0

    for sec, text in blocks:
        text = text.strip()
        if not text:
            continue
        if len(text) > size:
            flush()
            for piece in splitter.split_text(text):
                chunks.append(_header(filename, f"{label}: {sec}" if sec else "") + "\n" + piece)
            continue
        if buf_len + len(text) > size:
            flush()
        buf.append(text); buf_secs.append(sec); buf_len += len(text) + 2
    flush()
    return chunks


def _pack_lines(lines, filename, size, *meta, first_line=1):
    """Pack whole lines into chunks (never splitting a line); header carries the line range."""
    chunks, buf, start, length = [], [], first_line, 0
    for i, line in enumerate(lines, start=first_line):
        if buf and length + len(line) > size:
            chunks.append(_header(filename, *meta, f"lines {start}-{i - 1}") + "\n" + "".join(buf))
            buf, start, length = [], i, 0
        buf.append(line); length += len(line)
    if "".join(buf).strip():
        chunks.append(_header(filename, *meta, f"lines {start}-{first_line + len(lines) - 1}") + "\n" + "".join(buf))
    return chunks


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------
def _symbols(text: str, limit=4) -> str:
    names = []
    for m in SYMBOL_RE.finditer(text):
        name = next(g for g in m.groups() if g)
        if name not in names:
            names.append(name)
    if not names:
        return ""
    more = f" +{len(names) - limit} more" if len(names) > limit else ""
    return "symbols: " + ", ".join(names[:limit]) + more


def _python_units(source: str):
    """Yield (start_line, end_line, label) for each top-level unit of a Python module."""
    tree = ast.parse(source)
    lines = source.splitlines()
    units, prev_end = [], 0

    def with_leading_comments(start):
        s = start
        while s - 1 > prev_end and lines[s - 2].lstrip().startswith("#"):
            s -= 1
        return s

    for node in tree.body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        end = node.end_lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = with_leading_comments(start)
            if start > prev_end + 1:
                units.append((prev_end + 1, start - 1, "module code"))
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            units.append((start, end, f"{kind} {node.name}", node))
        else:
            continue  # module-level statements are collected as gaps
        prev_end = end
    if prev_end < len(lines):
        units.append((prev_end + 1, len(lines), "module code"))
    return units


def _chunk_python(source: str, filename: str):
    lines = source.splitlines(keepends=True)
    chunks = []

    def emit(start, end, label, prefix=""):
        body = "".join(lines[start - 1:end])
        if not body.strip():
            return
        if len(body) + len(prefix) <= CODE_SIZE:
            chunks.append(_header(filename, "python", f"lines {start}-{end}", label) + "\n" + prefix + body)
        else:  # very long block: windows of whole lines, each repeating the class line if any
            a, buf, length = start, [], 0
            for i in range(start, end + 1):
                line = lines[i - 1]
                if buf and length + len(line) > CODE_SIZE - len(prefix):
                    chunks.append(_header(filename, "python", f"lines {a}-{i - 1}", label + " (part)") + "\n" + prefix + "".join(buf))
                    a, buf, length = i, [], 0
                buf.append(line); length += len(line)
            if "".join(buf).strip():
                chunks.append(_header(filename, "python", f"lines {a}-{end}", label + " (part)") + "\n" + prefix + "".join(buf))

    for unit in _python_units(source):
        start, end, label = unit[:3]
        node = unit[3] if len(unit) > 3 else None
        size = len("".join(lines[start - 1:end]))
        if isinstance(node, ast.ClassDef) and size > CODE_SIZE:
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            class_line = lines[node.lineno - 1].rstrip("\n")
            cursor = start
            for m in methods:
                m_start = min([m.lineno] + [d.lineno for d in m.decorator_list])
                if m_start > cursor:
                    emit(cursor, m_start - 1, f"{label} (attributes)" if cursor > start else label)
                emit(m_start, m.end_lineno, f"{node.name}.{m.name}", prefix=f"{class_line}  # ...\n")
                cursor = m.end_lineno + 1
            if cursor <= end:
                emit(cursor, end, f"{label} (rest)")
        else:
            emit(start, end, label)
    return chunks


def _chunk_code(text: str, filename: str, lang: str):
    if lang == "python":
        try:
            return _chunk_python(text, filename)
        except SyntaxError:
            pass  # fall through to the generic splitter
    if lang in CUSTOM_SEPARATORS:
        splitter = RecursiveCharacterTextSplitter(
            separators=CUSTOM_SEPARATORS[lang], chunk_size=CODE_SIZE, chunk_overlap=CODE_OVERLAP,
            add_start_index=True)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language(lang), chunk_size=CODE_SIZE, chunk_overlap=CODE_OVERLAP, add_start_index=True)
    chunks = []
    for doc in splitter.create_documents([text]):
        start_idx = doc.metadata.get("start_index", 0)
        first = text.count("\n", 0, max(start_idx, 0)) + 1
        last = first + doc.page_content.count("\n")
        chunks.append(_header(filename, lang, f"lines {first}-{last}", _symbols(doc.page_content))
                      + "\n" + doc.page_content)
    return chunks


def _chunk_config(text: str, filename: str, ext: str):
    kind = ext.lstrip(".")
    if ext == ".json":
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
            if isinstance(parsed, dict):
                kind += " | top-level keys: " + ", ".join(list(parsed)[:12])
        except json.JSONDecodeError:
            pass
    return _pack_lines(text.splitlines(keepends=True), filename, CODE_SIZE, kind)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# OCR (Tesseract)
# ---------------------------------------------------------------------------
# Apps opened from Finder/Start menu don't get the shell's PATH, so Homebrew's /opt/homebrew/bin is
# invisible to them: look in the usual install locations explicitly.
_TESSERACT_CANDIDATES = [
    os.environ.get("TESSERACT_CMD", ""),
    "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def tesseract_path():
    found = shutil.which("tesseract")
    if found:
        return found
    return next((c for c in _TESSERACT_CANDIDATES if c and os.path.isfile(c)), None)


_TESSERACT = tesseract_path()
if _TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT
# Each OCR worker is its own tesseract process; stop each one also spawning a thread per core.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

OCR_DPI = 250          # 250 dpi reads normal body text as well as 300 dpi, and is ~30% faster
OCR_WORKERS = max(1, min(8, (os.cpu_count() or 2)))
INSTALL_HINT = ("Mac: brew install tesseract   Windows: install from github.com/UB-Mannheim/tesseract/wiki"
                if sys.platform != "linux" else "sudo apt install tesseract-ocr")


def ocr_available() -> bool:
    return _TESSERACT is not None


def _ocr(image: Image.Image) -> str:
    if _TESSERACT is None:
        raise ValueError("Tesseract OCR is not installed, so scanned pages and images can't be read. "
                         f"Install it ({INSTALL_HINT}), then restart DocuMind.")
    image = ImageOps.autocontrast(ImageOps.grayscale(image))
    return pytesseract.image_to_string(image)


class NoTextInImage(ValueError):
    """An image with no readable text (a product photo, a chart, a scene). The indexer can hand
    it to the local image model instead. ocr_missing: Tesseract isn't installed at all."""

    def __init__(self, message="No readable text found.", ocr_missing=False, text=""):
        super().__init__(message)
        self.ocr_missing = ocr_missing
        self.text = text  # the few words OCR did read confidently, e.g. a one-word sign


MIN_TEXT_WORDS = 4      # fewer confident words than this: treat the image as a photo without text
MIN_WORD_CONFIDENCE = 60


def _ocr_image(image: Image.Image) -> str:
    """OCR a standalone image and decide whether it really contains text.

    On photos Tesseract often "reads" stray marks as short junk words with low confidence, so an
    image counts as text only when it has at least MIN_TEXT_WORDS words read with confidence
    >= MIN_WORD_CONFIDENCE. The text is rebuilt from the same OCR pass (lines and paragraphs kept),
    so each image is read once.
    """
    if _TESSERACT is None:
        raise NoTextInImage("Tesseract OCR is not installed, so text in images can't be read. "
                            f"Install it ({INSTALL_HINT}), then restart DocuMind.", ocr_missing=True)
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    lines, confident = {}, 0
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf >= MIN_WORD_CONFIDENCE and sum(ch.isalnum() for ch in word) >= 2:
            confident += 1
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    out, last_par = [], None
    for key in sorted(lines):
        par = key[:2]
        if last_par is not None and par != last_par:
            out.append("")
        out.append(" ".join(lines[key]))
        last_par = par
    text = "\n".join(out)
    if confident < MIN_TEXT_WORDS:
        raise NoTextInImage(text=text if confident else "")
    return text


VISION_MAX_SIDE = 1024  # about 800 image tokens for Qwen3-VL: enough detail, small memory use


def image_for_vision(file_bytes: bytes) -> bytes:
    """A resized JPEG copy for the image model: right way up, at most VISION_MAX_SIDE pixels."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes)))
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1])
        image = background
    image = image.convert("RGB")
    image.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=88)
    return out.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _chunk_pdf(file_bytes: bytes, filename: str, progress=None):
    """Fast PDF path.

    1. Text layer: pdfium (Chrome's PDF engine) reads every page, typically several times faster
       than pdfplumber, with no poppler needed.
    2. Pages with no text layer (scanned or blank): rendered one at a time by pdfium and OCR'd
       by Tesseract on all CPU cores in parallel. At most two rendered pages per core are in memory.
    """
    report = progress or (lambda *a: None)
    try:
        pdf = pdfium.PdfDocument(file_bytes)
    except Exception as e:
        raise ValueError(f"Failed to open this PDF: {e}")
    try:
        total = len(pdf)
        pages = []
        for i in range(total):
            page = pdf[i]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range().replace("\r\n", "\n").replace("\r", "\n"))
            textpage.close()
            page.close()
            if i % 25 == 0 or i == total - 1:
                report("reading", i + 1, total)

        weak = [i for i, p in enumerate(pages) if len(p.strip()) < 20]
        if limits.ENFORCED and len(weak) > limits.MAX_OCR_PAGES:
            raise ValueError(
                f"This PDF has {len(weak)} pages without a text layer (scanned or blank) that need OCR; "
                f"the limit is {limits.MAX_OCR_PAGES} pages. Split the PDF into smaller parts and upload them separately.")
        if weak and _TESSERACT is None and not any(p.strip() for p in pages):
            _ocr(None)  # raises the "install Tesseract" message
        if weak and _TESSERACT is not None:
            scale = OCR_DPI / 72
            done = 0
            with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
                pending = {}
                for i in weak:
                    page = pdf[i]  # pdfium is not thread-safe: render here, OCR in the pool
                    image = page.render(scale=scale, grayscale=True).to_pil()
                    page.close()
                    pending[pool.submit(_ocr, image)] = i
                    if len(pending) >= OCR_WORKERS * 2:
                        done += _collect(pending, pages, wait_all=False)
                        report("ocr", done, len(weak))
                while pending:
                    done += _collect(pending, pages, wait_all=False)
                    report("ocr", done, len(weak))
    finally:
        pdf.close()

    if not any(p.strip() for p in pages):
        raise ValueError("Failed to extract text from PDF. Please check your file.")
    chunks = []
    splitter = _prose_splitter()
    for n, page in enumerate(pages, start=1):
        if page.strip():
            for piece in splitter.split_text(page):
                chunks.append(_header(filename, f"page {n}") + "\n" + piece)
    return chunks


def _collect(pending: dict, pages: list, wait_all: bool) -> int:
    """Wait for at least one OCR result and store it; returns how many finished."""
    from concurrent.futures import FIRST_COMPLETED, wait
    finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
    for future in finished:
        i = pending.pop(future)
        try:
            pages[i] = future.result()
        except Exception as e:
            print(f"OCR failed on page {i + 1}: {e}")
    return len(finished)


def _table_records(rows, header=None):
    """Render table rows as 'Column: value' records so every chunk is self-describing."""
    rows = [[("" if c is None else str(c).strip()) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return None, []
    header = header or rows[0]
    header = [h or f"Column {i + 1}" for i, h in enumerate(header)]
    body = rows[1:] if header == rows[0] else rows
    records = []
    for r in body:
        cells = [f"{header[i] if i < len(header) else f'Column {i + 1}'}: {v}" for i, v in enumerate(r) if v]
        if cells:
            records.append(" | ".join(cells))
    return header, records


def _chunk_docx(file_bytes: bytes, filename: str):
    d = docx.Document(io.BytesIO(file_bytes))
    path, blocks = [], []
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"(?:Heading|Title)\s*(\d*)", style)
            if m:
                level = int(m.group(1)) if m.group(1) else 1
                path = path[:level - 1] + [text]
                blocks.append((" > ".join(path), text))
            else:
                prefix = "• " if "List" in style else ""
                blocks.append((" > ".join(path), prefix + text))
        elif tag == "tbl":
            t = Table(el, d)
            rows = [[c.text for c in r.cells] for r in t.rows]
            header, records = _table_records(rows)
            if records:
                blocks.append((" > ".join(path), "Table:\n" + "\n".join(records)))
    return _pack_blocks(blocks, filename)


def _chunk_frame(df: pd.DataFrame, filename: str, where: str):
    """Rows -> records, header repeated per chunk, rows never split, plus a summary chunk."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []
    df.columns = [str(c).strip() if not str(c).startswith("Unnamed") else f"Column {i + 1}"
                  for i, c in enumerate(df.columns)]
    cols = list(df.columns)
    chunks = []

    # summary chunk: helps "how many", "total", "highest" questions
    summary = [f"Rows: {len(df)}", f"Columns: {', '.join(cols)}"]
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() >= max(1, int(0.8 * df[c].notna().sum())) and num.notna().any():
            summary.append(f"{c}: total {num.sum():g}, average {num.mean():.4g}, "
                           f"min {num.min():g}, max {num.max():g}")
    chunks.append(_header(filename, where, "summary") + "\n" + "\n".join(summary))

    col_line = ", ".join(cols)
    buf, first, length = [], None, 0
    for idx, (_, row) in enumerate(df.iterrows(), start=2):  # row 1 is the header
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v) or str(v).strip() == "":
                continue
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            cells.append(f"{c}: {v}")
        rec = " | ".join(cells)
        if buf and length + len(rec) > TABLE_SIZE:
            chunks.append(_header(filename, where, f"rows {first}-{idx - 1}", f"columns: {col_line}") + "\n" + "\n".join(buf))
            buf, length = [], 0
        if not buf:
            first = idx
        buf.append(rec); length += len(rec) + 1
    if buf:
        chunks.append(_header(filename, where, f"rows {first}-{idx}", f"columns: {col_line}") + "\n" + "\n".join(buf))
    return chunks


def _chunk_xlsx(file_bytes: bytes, filename: str):
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
    chunks = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(c is not None and str(c).strip() for c in r)]
        if not rows:
            continue
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        df = pd.DataFrame(rows[1:], columns=[h or f"Column {i + 1}" for i, h in enumerate(header)])
        chunks += _chunk_frame(df, filename, f"Sheet: {ws.title}")
    return chunks


def _chunk_delimited(file_bytes: bytes, filename: str, sep=None):
    text = _decode_text(file_bytes, filename)
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    return _chunk_frame(df, filename, "table")


# ---------------------------------------------------------------------------
# Charts in PowerPoint
# ---------------------------------------------------------------------------
# A chart made in PowerPoint is not a picture: the file keeps the numbers behind it (categories,
# series, values, number formats) in the chart part's cache. They are read straight from the XML,
# so values are exact, nothing is estimated, and it works with any AI provider.
#   * Classic charts (c: namespace): column, bar, line, pie, doughnut, area, scatter, bubble, radar,
#     stock, surface, their 3-D and stacked / 100% variants, pie-of-pie, and combo charts.
#   * Office 2016 charts (cx: "chartex"): waterfall, funnel, treemap, sunburst, histogram, pareto,
#     box & whisker and map charts. python-pptx doesn't know these, so their parts are found
#     through the slide's relationships.
# Each chart becomes a small table plus a few computed facts (highest, lowest, total, change),
# because small local models are unreliable at arithmetic over a list of numbers.

_C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_CX = "http://schemas.microsoft.com/office/drawing/2014/chartex"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS = {"c": _C, "a": _A, "cx": _CX}

_CHART_NAMES = {
    "barChart": "bar", "bar3DChart": "3-D bar", "lineChart": "line", "line3DChart": "3-D line",
    "pieChart": "pie", "pie3DChart": "3-D pie", "ofPieChart": "pie of pie", "doughnutChart": "doughnut",
    "areaChart": "area", "area3DChart": "3-D area", "scatterChart": "scatter (XY)", "bubbleChart": "bubble",
    "radarChart": "radar", "stockChart": "stock", "surfaceChart": "surface", "surface3DChart": "3-D surface",
}
_CHARTEX_NAMES = {
    "waterfall": "waterfall", "funnel": "funnel", "treemap": "treemap", "sunburst": "sunburst",
    "clusteredColumn": "histogram", "paretoLine": "pareto", "boxWhisker": "box and whisker",
    "regionMap": "map",
}


def _fmt_number(value, code: str = "", group: bool = True) -> str:
    """Show a value the way the chart's number format does: 12%, $1,250, 3.5.
    group=False for labels such as years (2026, not 2,026)."""
    if value is None:
        return ""
    code = (code or "").split(";")[0]
    if "%" in code:
        decimals = len(code.split(".")[1].split("%")[0]) if "." in code else 0
        return f"{value * 100:,.{decimals}f}%"
    m = re.search(r"\.(0+)", code)
    if m:
        decimals = len(m.group(1))
    else:
        shown = f"{round(float(value), 4):g}" if abs(value) < 1e15 else "0"
        decimals = len(shown.split(".")[1]) if "." in shown and "e" not in shown else 0
        if re.search(r"[0#]", code) and "." not in code:
            decimals = 0
    text = f"{value:,.{decimals}f}" if group and ("," in code or abs(value) >= 1000) else f"{value:.{decimals}f}"
    symbol = next((s for s in ("$", "€", "£", "₹", "¥") if s in code), "")
    if symbol:
        text = ("-" + symbol + text[1:]) if text.startswith("-") else symbol + text
    return text


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _rich_text(el) -> str:
    if el is None:
        return ""
    runs = [t.text or "" for t in el.iter(f"{{{_A}}}t")]
    if runs:
        return "".join(runs).strip()
    return " ".join((v.text or "").strip() for v in el.iter(f"{{{_C}}}v", f"{{{_CX}}}v")).strip()


def _c_points(ref):
    """{idx: value} from a strRef/numRef/strLit/numLit cache; also the format code."""
    if ref is None:
        return {}, ""
    pts, code = {}, ""
    fc = ref.find(".//c:formatCode", _NS)
    if fc is not None and fc.text:
        code = fc.text
    for pt in ref.iter(f"{{{_C}}}pt"):
        v = pt.find("c:v", _NS)
        if v is not None:
            pts[int(pt.get("idx", len(pts)))] = (v.text or "", pt.get("formatCode") or "")
    return pts, code


def _c_categories(cat, group=False):
    """Category labels by index. Multi-level categories become 'Outer / Inner'.
    group: thousands separators for numeric labels (scatter X values), not for years."""
    if cat is None:
        return {}
    multi = cat.find("c:multiLvlStrRef", _NS)
    if multi is not None:
        levels = multi.findall(".//c:lvl", _NS)  # innermost level first
        n = int((multi.find(".//c:ptCount", _NS).get("val")) if multi.find(".//c:ptCount", _NS) is not None else 0)
        filled = []
        for lvl in levels:
            labels = {int(p.get("idx")): (p.find("c:v", _NS).text or "") for p in lvl.findall("c:pt", _NS)}
            row, last = {}, ""
            for i in range(n or (max(labels) + 1 if labels else 0)):
                last = labels.get(i, last)
                row[i] = last
            filled.append(row)
        return {i: " / ".join(lvl.get(i, "") for lvl in reversed(filled) if lvl.get(i, "")) for i in filled[0]} if filled else {}
    pts, code = _c_points(cat)
    out = {}
    for i, (text, pcode) in pts.items():
        n = _num(text)
        if n is not None and (pcode or code) and re.search(r"[dmy]", (pcode or code).lower()) and n > 1000:
            # dates are stored as Excel serial days
            out[i] = (datetime(1899, 12, 30) + timedelta(days=n)).strftime("%d %b %Y")
        elif n is not None:
            out[i] = _fmt_number(n, pcode or code, group=group)
        else:
            out[i] = text
    return out


def _c_values(el):
    pts, code = _c_points(el)
    return {i: (_num(t), pc or code) for i, (t, pc) in pts.items()}


def _summary(name, cats, vals, *, total=True, trend=False):
    """Highest, lowest, total and change for one series. vals: {idx: (number, format)}."""
    items = [(cats.get(i, str(i + 1)), v, f) for i, (v, f) in sorted(vals.items()) if v is not None]
    if len(items) < 2:
        return ""
    fmt = items[0][2]
    hi = max(items, key=lambda x: x[1])
    lo = min(items, key=lambda x: x[1])
    facts = [f"highest {hi[0]} ({_fmt_number(hi[1], fmt)})", f"lowest {lo[0]} ({_fmt_number(lo[1], fmt)})"]
    if total and "%" not in fmt:
        facts.append(f"total {_fmt_number(sum(x[1] for x in items), fmt)}")
    if trend:
        first, last = items[0], items[-1]
        change = last[1] - first[1]
        sign = "+" if change >= 0 else "-"
        if "%" in fmt:
            diff = f"{sign}{abs(change) * 100:.1f} percentage points"
        else:
            diff = f"{sign}{_fmt_number(abs(change), fmt)}" + (f", {change / first[1] * 100:+.1f}%" if first[1] else "")
        facts.append(f"from {first[0]} to {last[0]}: {_fmt_number(first[1], fmt)} to {_fmt_number(last[1], fmt)} ({diff})")
    return f"{name}: " + "; ".join(facts) + "."


_TIME_LABEL = re.compile(
    r"^(q[1-4]\b|h[12]\b|fy|week|wk|\d{4}\b|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|mon|tue|wed|thu|fri|sat|sun)[a-z]*\b|\d{1,2} [a-z]{3} \d{4})",
    re.I)


def _is_time_axis(labels) -> bool:
    """Quarters, months, years or dates: a change from first to last is meaningful."""
    labels = [l for l in labels if l]
    return len(labels) >= 3 and all(_TIME_LABEL.match(l.split(" / ")[-1].strip()) or _TIME_LABEL.match(l.strip()) for l in labels)


def _classic_chart_text(space) -> str:
    """One classic chart (c:chartSpace element) as text."""
    chart = space.find("c:chart", _NS)
    if chart is None:
        return ""
    plot_area = chart.find("c:plotArea", _NS)
    plots = [p for p in plot_area if isinstance(p.tag, str) and p.tag.endswith("Chart")] if plot_area is not None else []
    series, kinds = [], []
    for plot in plots:
        kind = _CHART_NAMES.get(plot.tag.split("}")[1], plot.tag.split("}")[1])
        bar_dir = plot.find("c:barDir", _NS)
        if bar_dir is not None:
            kind = ("3-D " if "3D" in plot.tag else "") + ("column" if bar_dir.get("val") == "col" else "bar")
        grouping = plot.find("c:grouping", _NS)
        g = grouping.get("val") if grouping is not None else ""
        if g in ("stacked", "percentStacked"):
            kind = ("100% stacked " if g == "percentStacked" else "stacked ") + kind
        elif g == "clustered" and "bar" in plot.tag:
            kind = "clustered " + kind
        if plot.find("c:holeSize", _NS) is not None or kind == "doughnut":
            kind = "doughnut"
        kinds.append(kind)
        for n, ser in enumerate(plot.findall("c:ser", _NS)):
            tx = ser.find("c:tx", _NS)
            name = _rich_text(tx) if tx is not None else ""
            series.append({
                "kind": kind, "tag": plot.tag.split("}")[1], "g": g,
                "name": name or f"Series {len(series) + 1}",
                "cats": _c_categories(ser.find("c:cat", _NS)),
                "vals": _c_values(ser.find("c:val", _NS)),
                "x": _c_categories(ser.find("c:xVal", _NS), group=True) if ser.find("c:xVal", _NS) is not None else None,
                "y": _c_values(ser.find("c:yVal", _NS)),
                "size": _c_values(ser.find("c:bubbleSize", _NS)),
            })
    if not series:
        return ""

    title = ""
    t = chart.find("c:title", _NS)
    if t is not None:
        title = _rich_text(t.find("c:tx", _NS))
    if not title and len(series) == 1 and (chart.find("c:autoTitleDeleted", _NS) is None
                                           or chart.find("c:autoTitleDeleted", _NS).get("val") in ("0", "false")):
        title = series[0]["name"]
    axes = []
    for ax_tag, role in (("c:catAx", "category axis"), ("c:dateAx", "category axis"), ("c:valAx", "value axis"), ("c:serAx", "series axis")):
        for ax in plot_area.findall(ax_tag, _NS):
            at = _rich_text(ax.find("c:title", _NS)) if ax.find("c:title", _NS) is not None else ""
            if at:
                axes.append(f"{role}: {at}")
    kind_text = " + ".join(dict.fromkeys(kinds)) + (" combo" if len(set(kinds)) > 1 else "")
    lines = [f"Chart: {title} ({kind_text} chart)" if title else f"Chart: {kind_text} chart"]
    if axes:
        lines.append("Axes: " + "; ".join(axes))

    xy = [s for s in series if s["tag"] in ("scatterChart", "bubbleChart")]
    cat_series = [s for s in series if s not in xy]
    if cat_series:
        cats = {}
        for s in cat_series:
            for i, label in s["cats"].items():
                cats.setdefault(i, label)
        idxs = sorted(set(cats) | {i for s in cat_series for i in s["vals"]})
        if not cats:
            cats = {i: str(i + 1) for i in idxs}
        is_pie = all(s["tag"] in ("pieChart", "pie3DChart", "ofPieChart", "doughnutChart") for s in cat_series)
        cat_title = next((a.split(": ", 1)[1] for a in axes if a.startswith("category axis")), "Category")
        header = [cat_title] + [s["name"] + (f" ({s['kind']})" if len(set(kinds)) > 1 else "") for s in cat_series]
        lines.append(" | ".join(header))
        totals = [sum(v for v, _ in s["vals"].values() if v is not None) for s in cat_series]
        for i in idxs:
            row = [cats.get(i, str(i + 1))]
            for s, tot in zip(cat_series, totals):
                v, f = s["vals"].get(i, (None, ""))
                cell = _fmt_number(v, f)
                if is_pie and v is not None and tot and "%" not in f:
                    cell += f" ({v / tot * 100:.1f}% of total)"
                row.append(cell)
            lines.append(" | ".join(row))
        stacked = [s for s in cat_series if s["g"] in ("stacked", "percentStacked")]
        if len(cat_series) > 1 and len(stacked) == len(cat_series) and stacked[0]["g"] == "stacked":
            col_totals = {i: sum(s["vals"].get(i, (0, ""))[0] or 0 for s in cat_series) for i in idxs}
            f = cat_series[0]["vals"].get(idxs[0], (0, ""))[1] if idxs else ""
            lines.append("Stack total | " + " | ".join(f"{cats.get(i, i)}: {_fmt_number(v, f)}" for i, v in col_totals.items()))
        over_time = _is_time_axis([cats.get(i, "") for i in idxs])
        facts = [_summary(s["name"], cats, s["vals"], total=s["g"] != "percentStacked" and s["tag"] != "stockChart",
                          trend=over_time and s["tag"] not in ("pieChart", "pie3DChart", "ofPieChart", "doughnutChart"))
                 for s in cat_series if s["tag"] not in ("radarChart", "surfaceChart", "surface3DChart")]
        facts = [f for f in facts if f]
        if facts:
            lines.append("Facts: " + " ".join(facts))
    x_title, y_title = "X", "Y"
    for ax in plot_area.findall("c:valAx", _NS):
        pos = ax.find("c:axPos", _NS)
        at = _rich_text(ax.find("c:title", _NS)) if ax.find("c:title", _NS) is not None else ""
        if at and pos is not None and pos.get("val") in ("b", "t"):
            x_title = at
        elif at:
            y_title = at
    for s in xy:
        xs = s["x"] or {}
        has_size = bool(s["size"])
        lines.append(f"Series {s['name']} ({s['kind']}):")
        lines.append(" | ".join(["Point", x_title, y_title] + (["Bubble size"] if has_size else [])))
        for i in sorted(s["y"]):
            y, yf = s["y"][i]
            row = [str(i + 1), xs.get(i, str(i + 1)), _fmt_number(y, yf)]
            if has_size:
                sv, sf = s["size"].get(i, (None, ""))
                row.append(_fmt_number(sv, sf))
            lines.append(" | ".join(row))
        ys = [(v, f, i) for i, (v, f) in s["y"].items() if v is not None]
        if len(ys) >= 2:
            hi, lo = max(ys), min(ys)
            lines.append(f"Facts: {s['name']}: highest {y_title} {_fmt_number(hi[0], hi[1])} at {x_title} {xs.get(hi[2], hi[2] + 1)}; "
                         f"lowest {y_title} {_fmt_number(lo[0], lo[1])} at {x_title} {xs.get(lo[2], lo[2] + 1)}.")
    return "\n".join(lines)


def _cx_dim(dim):
    """A chartex strDim/numDim: (list of levels as {idx: value}, format code)."""
    levels, code = [], ""
    for lvl in dim.findall("cx:lvl", _NS):
        code = code or lvl.get("formatCode") or ""
        levels.append({int(p.get("idx")): (p.text or "") for p in lvl.findall("cx:pt", _NS)})
    return levels, code


def _quartiles(values):
    v = sorted(values)
    def q(p):
        k = (len(v) - 1) * p
        f = int(k)
        return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)
    return q(0), q(0.25), q(0.5), q(0.75), q(1)


def _chartex_text(xml: bytes) -> str:
    """One Office 2016 chart (waterfall, funnel, treemap, ...) as text."""
    root = etree.fromstring(xml)
    data = {}
    for d in root.findall(".//cx:chartData/cx:data", _NS):
        dims = {}
        for dim in d:
            if not isinstance(dim.tag, str):
                continue
            levels, code = _cx_dim(dim)
            dims[dim.get("type") or dim.tag.split("}")[1]] = (levels, code, dim.tag.endswith("numDim"))
        data[d.get("id")] = dims
    chart = root.find("cx:chart", _NS)
    title = _rich_text(chart.find("cx:title", _NS)) if chart is not None and chart.find("cx:title", _NS) is not None else ""
    out = []
    for ser in root.iter(f"{{{_CX}}}series"):
        layout = ser.get("layoutId", "")
        kind = _CHARTEX_NAMES.get(layout, layout)
        if layout == "paretoLine":
            continue  # the pareto line is computed from the histogram series
        name = _rich_text(ser.find("cx:tx", _NS)) or "Values"
        data_id = ser.find("cx:dataId", _NS)
        dims = data.get(data_id.get("val") if data_id is not None else "0", {})
        cat_levels = dims.get("cat", ([], "", False))[0]
        val_levels, val_code, _ = dims.get("val", dims.get("size", ([{}], "", True)))
        vals = {i: _num(t) for i, t in (val_levels[0] if val_levels else {}).items()}
        n = max([len(vals)] + [max(l) + 1 for l in cat_levels if l]) if (vals or cat_levels) else 0
        # category paths, outer level first; blank outer labels repeat the one above
        paths = []
        filled = []
        for lvl in cat_levels:
            row, last = {}, ""
            for i in range(n):
                last = lvl.get(i, last) or last
                row[i] = last
            filled.append(row)
        for i in range(n):
            parts = [lvl[i] for lvl in reversed(filled) if lvl.get(i)]
            paths.append(" > ".join(dict.fromkeys(parts)) if parts else str(i + 1))
        if not out:
            out.append(f"Chart: {title or name} ({kind} chart)")
        numbers = [(paths[i] if i < len(paths) else str(i + 1), v) for i, v in sorted(vals.items()) if v is not None]
        pareto = layout == "clusteredColumn" and root.find(".//cx:series[@layoutId='paretoLine']", _NS) is not None
        if layout == "clusteredColumn" and cat_levels:
            kind = "pareto" if pareto else "column"  # a histogram of categories: counts per category
            out[0] = f"Chart: {title or name} ({kind} chart)"
            layout = "pareto" if pareto else "categories"
        stat = lambda x: _fmt_number(round(x, 1) if "$" not in val_code and "%" not in val_code else x, val_code)
        if layout in ("clusteredColumn", "boxWhisker") and numbers:
            values = [v for _, v in numbers]
            lo, q1, med, q3, hi = _quartiles(values)
            mean = sum(values) / len(values)
            if layout == "boxWhisker" and cat_levels:
                groups = {}
                for label, v in numbers:
                    groups.setdefault(label, []).append(v)
                out.append(f"{name}: " + " ".join(
                    f"{g}: min {stat(_quartiles(vs)[0])}, Q1 {stat(_quartiles(vs)[1])}, "
                    f"median {stat(_quartiles(vs)[2])}, Q3 {stat(_quartiles(vs)[3])}, "
                    f"max {stat(_quartiles(vs)[4])}, mean {stat(sum(vs) / len(vs))} ({len(vs)} values)."
                    for g, vs in groups.items()))
            else:
                out.append(f"{name}: {len(values)} values; min {stat(lo)}, Q1 {stat(q1)}, "
                           f"median {stat(med)}, Q3 {stat(q3)}, max {stat(hi)}, "
                           f"mean {stat(mean)}.")
                if layout == "clusteredColumn":
                    bins = _histogram_bins(values)
                    out.append("Distribution (DocuMind's own bins; the chart's bins may differ): " + "; ".join(
                        f"{stat(a)} to {stat(b)}: {c}" for a, b, c in bins) + ".")
            if len(values) <= 60:
                out.append("Values: " + ", ".join(_fmt_number(v, val_code) for v in values))
            continue
        subtotals = {int(i.get("val")) for i in ser.findall(".//cx:subtotals/cx:idx", _NS)}
        header = "Category" if not cat_levels or len(cat_levels) == 1 else "Path"
        out.append(f"{header} | {name}")
        running = 0.0
        prev = None
        for i, (label, v) in enumerate(numbers):
            cell = _fmt_number(v, val_code)
            if layout == "waterfall":
                if i in subtotals:
                    cell += " (total)"
                    running = v
                else:
                    running += v
                    cell += f" (running total {_fmt_number(running, val_code)})"
            if layout == "funnel" and prev:
                cell += f" ({v / prev * 100:.1f}% of previous step)"
            if layout == "pareto":
                running += v
                total_all = sum(x for _, x in numbers)
                cell += f" ({v / total_all * 100:.1f}%, cumulative {running / total_all * 100:.1f}%)"
            prev = v
            out.append(f"{label} | {cell}")
        if layout in ("treemap", "sunburst") and len(cat_levels) > 1:
            groups = {}
            for label, v in numbers:
                top = label.split(" > ")[0]
                groups[top] = groups.get(top, 0) + v
            out.append("Totals by " + ("group" if len(groups) > 1 else "level") + ": "
                       + "; ".join(f"{g}: {_fmt_number(v, val_code)}" for g, v in groups.items()) + ".")
        if layout not in ("waterfall",) and len(numbers) >= 2:
            fact = _summary(name, {i: l for i, (l, _) in enumerate(numbers)},
                            {i: (v, val_code) for i, (_, v) in enumerate(numbers)}, total=layout != "funnel")
            if fact:
                out.append("Facts: " + fact)
        if layout == "funnel" and len(numbers) >= 2 and numbers[0][1]:
            out.append(f"Overall conversion from {numbers[0][0]} to {numbers[-1][0]}: {numbers[-1][1] / numbers[0][1] * 100:.1f}%.")
    return "\n".join(out)


def _histogram_bins(values, max_bins=8):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [(lo, hi, len(values))]
    k = min(max_bins, max(3, int(len(values) ** 0.5)))
    width = (hi - lo) / k
    counts = [0] * k
    for v in values:
        counts[min(k - 1, int((v - lo) / width))] += 1
    return [(lo + i * width, lo + (i + 1) * width, c) for i, c in enumerate(counts)]


def _slide_charts(slide) -> list[str]:
    """Office 2016 charts on a slide (python-pptx doesn't list them as shapes)."""
    out = []
    for el in slide._element.iter(f"{{{_CX}}}chart"):
        rid = el.get(f"{{{_R}}}id")
        try:
            out.append(_chartex_text(slide.part.related_part(rid).blob))
        except Exception:
            continue
    return [t for t in out if t]


class _ChartText(str):
    """Text of a chart: indexed as its own chunk(s), with the table header kept on each piece."""


def _chart_chunks(text: str, filename: str, label: str) -> list[str]:
    lines = text.split("\n")
    head_n = next((i + 1 for i, l in enumerate(lines) if " | " in l), min(2, len(lines)))
    head, rows = lines[:head_n], lines[head_n:]
    chunks, current = [], []
    for row in rows:
        if current and len("\n".join(head + current + [row])) > TABLE_SIZE:
            chunks.append(current)
            current = []
        current.append(row)
    if current or not chunks:
        chunks.append(current)
    header = _header(filename, label + " | chart")
    return [header + "\n" + "\n".join(head + c) for c in chunks]


def _shape_text(shape):
    out = []
    if shape.shape_type == 6 and hasattr(shape, "shapes"):  # group
        for s in shape.shapes:
            out += _shape_text(s)
    if getattr(shape, "has_chart", False) and shape.has_chart:
        try:
            text = _classic_chart_text(shape.chart._chartSpace)
        except Exception:
            text = ""
        if text:
            out.append(_ChartText(text))
    elif getattr(shape, "has_table", False) and shape.has_table:
        rows = [[c.text for c in r.cells] for r in shape.table.rows]
        _, records = _table_records(rows)
        if records:
            out.append("Table:\n" + "\n".join(records))
    elif getattr(shape, "has_text_frame", False) and shape.has_text_frame and shape.text_frame.text.strip():
        out.append(shape.text_frame.text.strip())
    return out


# Pictures on slides (often screenshots of charts pasted from Excel, a BI tool or a website).
# Efficient order: every picture is OCR'd first, in parallel (about a second each). The local image
# model, which is much slower, only looks at pictures that need it: ones that look like a chart or a
# table (words and several numbers) and ones with no text at all. Icons, logos repeated across
# slides and duplicates are skipped.
MIN_PICTURE_SIDE = 160              # pixels; smaller pictures are icons and bullets
DECORATION_SLIDES = 3               # the same picture on this many slides is a logo or background
MAX_PICTURES_FOR_IMAGE_MODEL = 20   # per file, to keep large decks quick
TEXT_PICTURE_WORDS = 15             # a picture with this many words and few numbers is a text screenshot
_NUMBER = re.compile(r"[-+]?[$€£₹¥]?\d[\d,.]*%?[kKmMbB]?")


def _iter_pictures(shapes):
    for shape in shapes:
        if shape.shape_type == 6 and hasattr(shape, "shapes"):  # group
            yield from _iter_pictures(shape.shapes)
            continue
        try:
            image = shape.image  # pictures and picture placeholders
        except Exception:
            continue
        yield image.blob


def _read_slide_pictures(slides, describe, progress):
    """{slide number: [text of each picture]}. slides: [(n, label, slide)]."""
    found, where = [], {}
    for n, label, slide in slides:
        for blob in _iter_pictures(slide.shapes):
            sha = hashlib.sha1(blob).hexdigest()
            where.setdefault(sha, set()).add(n)
            if len(where[sha]) == 1:
                found.append((sha, n, label, blob))
    pictures = []
    for sha, n, label, blob in found:
        if len(where[sha]) >= DECORATION_SLIDES:
            continue
        try:
            image = Image.open(io.BytesIO(blob))
            image.load()
        except Exception:
            continue  # vector formats (EMF, WMF) and broken pictures
        if min(image.size) < MIN_PICTURE_SIDE:
            continue
        pictures.append((n, label, blob, image))
    if not pictures:
        return {}

    def ocr(item):
        try:
            text = _ocr_image(item[3])
        except NoTextInImage as e:
            text = e.text
        except Exception:
            return "none", ""
        if _TESSERACT is None:
            return "none", text
        # Charts scatter their labels around the picture; Tesseract's sparse-text mode finds about
        # twice as many of them as the normal page mode. Text screenshots keep the normal reading.
        if len(re.findall(r"[A-Za-z]{2,}", text)) < TEXT_PICTURE_WORDS or len(_NUMBER.findall(text)) >= 3:
            try:
                gray = ImageOps.autocontrast(ImageOps.grayscale(item[3]))
                sparse = pytesseract.image_to_string(gray, config="--psm 11")
                sparse = "\n".join(l.strip() for l in sparse.splitlines() if l.strip())
                if len(_NUMBER.findall(sparse)) > len(_NUMBER.findall(text)):
                    text = sparse
            except Exception:
                pass
        return ("text" if text.strip() else "none"), text

    results, done = [None] * len(pictures), 0
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        for i, result in zip(range(len(pictures)), pool.map(ocr, pictures)):
            results[i] = result
            done += 1
            if progress:
                progress("pictures", done, len(pictures))

    # Which pictures the image model should look at:
    #   several numbers           -> a chart or table: read its data       ("chart")
    #   lots of words, few numbers -> a screenshot of text: OCR is enough
    #   few or no words            -> a chart without labels (pies often), a diagram or a photo
    texts, wanted = {}, []
    for (n, label, blob, _), (kind, text) in zip(pictures, results):
        numbers = len(_NUMBER.findall(text))
        words = len(re.findall(r"[A-Za-z]{2,}", text))
        if numbers >= 3:
            wanted.append((n, label, blob, "chart", text))
        elif words >= TEXT_PICTURE_WORDS:
            texts.setdefault(n, []).append("Text in picture (OCR):\n" + text)
        else:
            wanted.append((n, label, blob, "chart" if text.strip() else "photo", text))
    for i, (n, label, blob, kind, text) in enumerate(wanted):
        reading = ""
        if describe and i < MAX_PICTURES_FOR_IMAGE_MODEL:
            if progress:
                progress("describing", i + 1, min(len(wanted), MAX_PICTURES_FOR_IMAGE_MODEL))
            try:
                reading = describe(image_for_vision(blob), kind, label) or ""
            except Exception:
                reading = ""  # model missing or out of memory: keep what OCR found
        parts = []
        if reading:
            parts.append(("Chart or table in picture, read by the image model:\n" if kind == "chart"
                          else "Picture, described by the image model:\n") + reading.strip())
        if text.strip():
            parts.append("Text in picture (OCR):\n" + text.strip())
        if parts:
            texts.setdefault(n, []).append("\n\n".join(parts))
    return texts


def _chunk_pptx(file_bytes: bytes, filename: str, describe=None, progress=None):
    prs = pptx.Presentation(io.BytesIO(file_bytes))
    chunks, splitter = [], _prose_splitter()
    slides = []
    for n, slide in enumerate(prs.slides, start=1):
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        slides.append((n, f"slide {n}" + (f": {title}" if title else ""), slide))
    pictures = _read_slide_pictures(slides, describe, progress)
    for n, slide in enumerate(prs.slides, start=1):
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        parts = []
        for shape in slide.shapes:
            if slide.shapes.title is not None and shape.shape_id == slide.shapes.title.shape_id:
                continue
            parts += _shape_text(shape)
        parts += [_ChartText(t) for t in _slide_charts(slide)]
        charts = [p for p in parts if isinstance(p, _ChartText)]
        parts = [p for p in parts if not isinstance(p, _ChartText)]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Speaker notes: " + notes)
        label = f"slide {n}" + (f": {title}" if title else "")
        for chart in charts:
            chunks += _chart_chunks(chart, filename, label)
        for picture in pictures.get(n, []):
            for piece in (splitter.split_text(picture) if len(picture) > PROSE_SIZE else [picture]):
                chunks.append(_header(filename, label, "picture") + "\n" + piece)
        body = "\n".join(([title] if title else []) + parts).strip()
        if not body or ((charts or pictures.get(n)) and body == title):
            continue
        for piece in (splitter.split_text(body) if len(body) > PROSE_SIZE else [body]):
            chunks.append(_header(filename, label) + "\n" + piece)
    return chunks


def _chunk_markdown(text: str, filename: str):
    path, blocks, in_fence = [], [], False
    para = []

    def end_para():
        if para:
            blocks.append((" > ".join(path), "\n".join(para)))
            para.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            end_para()
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2).strip()]
            blocks.append((" > ".join(path), line))
        elif not line.strip() and not in_fence:
            end_para()
        else:
            para.append(line)
    end_para()
    return _pack_blocks(blocks, filename)


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "head"}
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol", "pre", "blockquote"}

    def __init__(self):
        super().__init__()
        self.blocks, self.path, self.buf, self.skip, self.heading = [], [], [], 0, None

    def _flush(self):
        text = re.sub(r"[ \t]+", " ", "".join(self.buf)).strip()
        if text:
            self.blocks.append((" > ".join(self.path), text))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush(); self.heading = int(tag[1])
        elif tag in self.BLOCK:
            self._flush()
        elif tag in ("td", "th"):
            self.buf.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif re.fullmatch(r"h[1-6]", tag) and self.heading:
            title = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            self.buf = []
            if title:
                self.path = self.path[:self.heading - 1] + [title]
                self.blocks.append((" > ".join(self.path), title))
            self.heading = None
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def _chunk_html(text: str, filename: str):
    p = _HTMLText()
    p.feed(text); p.close(); p._flush()
    return _pack_blocks(p.blocks, filename)


def _chunk_plain(text: str, filename: str):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _pack_blocks([("", p) for p in paras], filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_file_to_chunks(file_bytes: bytes, filename: str, progress=None, describe=None) -> list[str]:
    """progress(stage, done, total) is called during long steps (PDF reading and OCR).
    describe(jpeg, kind, where) -> text: the local image model, for pictures on slides that need it
    ("chart" for screenshots of charts and tables, "photo" for pictures without text); None if off."""
    name = os.path.basename(filename)
    lower = name.lower()
    ext = os.path.splitext(lower)[1]

    if ext == ".pdf":
        chunks = _chunk_pdf(file_bytes, name, progress)
    elif ext == ".docx":
        chunks = _chunk_docx(file_bytes, name)
    elif ext == ".xlsx":
        chunks = _chunk_xlsx(file_bytes, name)
    elif ext == ".csv":
        chunks = _chunk_delimited(file_bytes, name)
    elif ext == ".tsv":
        chunks = _chunk_delimited(file_bytes, name, sep="\t")
    elif ext == ".pptx":
        chunks = _chunk_pptx(file_bytes, name, describe, progress)
    elif ext == ".ppt":
        # Legacy binary PowerPoint is not supported: reading it as text only yields binary junk chunks.
        raise ValueError("PPT files are not supported. Please save the presentation as PPTX and upload again.")
    elif ext in IMAGE_EXTS:
        try:
            text = _ocr_image(Image.open(io.BytesIO(file_bytes)))
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to extract text from image: {e}")
        chunks = _pack_blocks([("", p) for p in re.split(r"\n\s*\n", text) if p.strip()], name, label="Section")
        chunks = [c.replace(_header(name), _header(name, "OCR text from image"), 1) for c in chunks]
    elif ext in (".md", ".markdown"):
        chunks = _chunk_markdown(_decode_text(file_bytes, name), name)
    elif ext in (".html", ".htm"):
        chunks = _chunk_html(_decode_text(file_bytes, name), name)
    elif ext == ".log":
        chunks = _pack_lines(_decode_text(file_bytes, name).splitlines(keepends=True), name, PROSE_SIZE, "log")
    elif ext == ".txt":
        chunks = _chunk_plain(_decode_text(file_bytes, name), name)
    elif ext in CODE_LANGS or lower in CODE_FILENAMES:
        lang = CODE_LANGS.get(ext) or CODE_FILENAMES[lower]
        chunks = _chunk_code(_decode_text(file_bytes, name), name, lang)
    elif ext in CONFIG_EXTS:
        chunks = _chunk_config(_decode_text(file_bytes, name), name, ext)
    else:
        raise ValueError(f"Unsupported file type: {ext or name}. See 'Supported files' for the list of formats DocuMind can read.")

    chunks = [c for c in chunks if c.split("\n", 1)[-1].strip()]
    if not chunks:
        raise ValueError("Could not extract any text from this file.")
    if limits.ENFORCED and len(chunks) > limits.MAX_CHUNKS:
        raise ValueError(
            f"This file contains too much text: it would create {len(chunks)} chunks; the limit is "
            f"{limits.MAX_CHUNKS} per file. Split it into smaller files and upload them separately.")
    return chunks
