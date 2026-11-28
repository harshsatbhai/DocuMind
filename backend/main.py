import os
import uuid
import re
import html
import hmac
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import ai, appdata, jobs, limits, storage
from .parser import ocr_available

DESKTOP_TOKEN = os.environ.get("DOCUMIND_TOKEN")  # set only by the desktop launcher

app = FastAPI()

# Reject oversized uploads before the body is read. Registered before CORSMiddleware so the
# 413 response still carries CORS headers and the browser can show its message.
@app.middleware("http")
async def limit_upload_size(request, call_next):
    if limits.ENFORCED and request.url.path == "/upload" and request.method == "POST":
        length = request.headers.get("content-length")
        max_body = limits.MAX_TOTAL_MB * limits.MB + limits.MB  # + 1 MB for form overhead
        if length and length.isdigit() and int(length) > max_body:
            return JSONResponse(status_code=413, content={
                "detail": f"Upload is {limits.fmt_mb(int(length))}; the limit is "
                          f"{limits.fmt_limit(limits.MAX_TOTAL_MB)} per upload."})
    return await call_next(request)

# Desktop app: the backend listens on 127.0.0.1 only, and every request must carry the random
# token the launcher generated, so other programs and websites on the computer can't use it.
@app.middleware("http")
async def require_desktop_token(request, call_next):
    if DESKTOP_TOKEN and request.method != "OPTIONS" and request.url.path != "/healthz":
        sent = request.headers.get("x-documind-token", "")
        if not hmac.compare_digest(sent, DESKTOP_TOKEN):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    if DESKTOP_TOKEN:
        # Never let the app's web view keep copies of documents or chats in its own cache,
        # so "Delete all my data" really removes everything.
        response.headers["Cache-Control"] = "no-store"
    return response

if DESKTOP_TOKEN:
    # Tauri's webview origins (macOS/Linux and Windows) plus the Vite dev server.
    _origins = ["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost", "http://localhost:5173"]
else:
    _origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_llm_response(text: str) -> str:
    """Strip markdown styling without damaging content.

    Hyphens and underscores inside words are content (snake_case names, W-220, 2026-11-24,
    phone numbers, negative numbers), so only markdown markers are removed. Code is protected:
    <pre>/<code> blocks are left untouched, and markdown code (``` fences, `inline`) is converted
    to HTML-escaped <pre><code>/<code> so it displays literally.
    """
    protected = []

    def keep(fragment: str) -> str:
        protected.append(fragment)
        return f"\x00{len(protected) - 1}\x00"

    text = re.sub(r"```[\w+#.-]*\n?(.*?)```",
                  lambda m: keep("<pre><code>" + html.escape(m.group(1).strip("\n")) + "</code></pre>"),
                  text, flags=re.DOTALL)
    text = re.sub(r"<pre>.*?</pre>", lambda m: keep(m.group(0)), text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<code>.*?</code>", lambda m: keep(m.group(0)), text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"`([^`\n]+)`", lambda m: keep("<code>" + html.escape(m.group(1)) + "</code>"), text)

    text = text.replace("**", "")
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", text)   # *emphasis*
    text = re.sub(r"^[ \t]*[-*•–—][ \t]+", "", text, flags=re.MULTILINE)            # bullet markers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)                          # headings
    text = re.sub(r"[ \t]+[–—][ \t]+", ", ", text)                                  # spaced dashes
    text = re.sub(r" +", " ", text)
    text = re.sub(r"(\d+\.)\s*\n+", r"\1 ", text)

    text = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)
    return text.strip()

@app.on_event("startup")
def startup():
    print(limits.describe())
    print(f"Storage: {storage.BACKEND}" + (f" at {appdata.data_dir()}" if storage.BACKEND == "sqlite" else ""))
    print(f"OCR (Tesseract): {'available' if ocr_available() else 'NOT installed'}")
    storage.init_db()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/limits")
def get_limits():
    return limits.as_dict()

@app.get("/status")
def status():
    settings = appdata.public_settings()
    return {"desktop": bool(DESKTOP_TOKEN), "storage": storage.BACKEND, "ocr": ocr_available(),
            "provider": settings["provider"], "nvidia_key_set": settings["nvidia_key_set"],
            "data_dir": str(appdata.data_dir()) if DESKTOP_TOKEN else None}

# ---- Settings (desktop only: on a shared web server nobody should change the server's AI key)
class SettingsIn(BaseModel):
    provider: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_embed_model: Optional[str] = None
    ollama_chat_model: Optional[str] = None
    ollama_context: Optional[str] = None
    ollama_vision_model: Optional[str] = None

def _desktop_only():
    if not DESKTOP_TOKEN:
        raise HTTPException(status_code=404, detail="Settings are only available in the desktop app.")

@app.get("/settings")
def get_settings():
    _desktop_only()
    return appdata.public_settings()

@app.put("/settings")
def put_settings(body: SettingsIn):
    _desktop_only()
    appdata.save_settings(body.model_dump(exclude_none=True))
    return appdata.public_settings()

@app.post("/settings/test")
async def test_settings(body: SettingsIn):
    _desktop_only()
    candidate = appdata.load_settings()
    candidate.update(body.model_dump(exclude_none=True))
    return await run_in_threadpool(ai.test_connection, candidate)

@app.delete("/data")
def delete_all_data():
    """Settings > Delete all my data: documents, chats, embeddings and settings (incl. API key)."""
    _desktop_only()
    if jobs.any_running():
        raise HTTPException(status_code=409, detail="Files are still being indexed. Wait for them to finish, then try again.")
    storage.delete_all()
    appdata.delete_settings()
    return {"deleted": True}

# ---- Workspace: restore documents and chat history when the app opens (SQLite only)
@app.get("/workspace")
def workspace():
    session_id = storage.last_session()
    if not session_id:
        return {"session_id": None, "documents": [], "messages": []}
    return {"session_id": session_id, "documents": storage.list_documents(session_id),
            "messages": storage.list_messages(session_id)}

async def read_limited(file: UploadFile, max_bytes: int):
    """Read an upload in 1 MB pieces; return None as soon as it exceeds max_bytes.
    With max_bytes None (limits off) the whole file is read."""
    if max_bytes is None:
        return await file.read()
    data = bytearray()
    while True:
        piece = await file.read(limits.MB)
        if not piece:
            return bytes(data)
        data.extend(piece)
        if len(data) > max_bytes:
            return None

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: str = Form(None),
):
    if limits.ENFORCED and len(files) > limits.MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files: {len(files)} selected; "
                                                    f"the limit is {limits.MAX_FILES} per upload.")
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    queued = []
    failed_files = []
    total_bytes = 0
    max_total = int(limits.MAX_TOTAL_MB * limits.MB)
    
    for file in files:
        max_bytes = limits.max_bytes_for(file.filename)
        contents = await read_limited(file, max_bytes)
        if contents is None:
            size = getattr(file, "size", None)
            size_text = f"{limits.fmt_mb(size)}, " if size else ""
            kind_limit = limits.MAX_IMAGE_MB if file.filename.lower().endswith(limits.IMAGE_EXTS) else limits.MAX_FILE_MB
            failed_files.append({"filename": file.filename,
                                 "error": f"File is too large ({size_text}limit {limits.fmt_limit(kind_limit)} per file)."})
            continue
        total_bytes += len(contents)
        if limits.ENFORCED and total_bytes > max_total:
            failed_files.append({"filename": file.filename,
                                 "error": f"Skipped: this upload is over the {limits.fmt_limit(limits.MAX_TOTAL_MB)} total limit."})
            continue
        # Indexing runs in the background; the app polls GET /jobs for progress.
        queued.append(jobs.submit(session_id, file.filename, contents))
            
    if not queued and failed_files:
        error_msg = "; ".join([f"{f['filename']}: {f['error']}" for f in failed_files])
        raise HTTPException(status_code=400, detail=f"Failed to process files. {error_msg}")

    return {
        "session_id": session_id,
        "jobs": queued,
        "failed": failed_files
    }

@app.get("/jobs")
def list_jobs(session_id: str, job_ids: List[str] = Query(default=[])):
    return {"jobs": jobs.get(session_id, job_ids)}

@app.post("/query")
async def query(
    session_id: str = Form(None),
    question: str = Form(...),
    doc_ids: List[str] = Form([])
):
    context = ""
    photos = []  # (filename, jpeg) the image model should look at again
    
    if session_id and doc_ids and len(doc_ids) > 0:
        try:
            context, photos = await run_in_threadpool(_retrieve, session_id, question, doc_ids)
        except ai.AIError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            print(f"Search error: {e}")

    system_prompt = """You are DocuMind, an intelligent AI document assistant. 

RULES FOR YOUR IDENTITY AND GENERAL CHAT:
1. If the user asks "what can you do?" or similar broad questions about your capabilities, you must respond EXACTLY with the following structure, using the exact HTML tags provided:
   <h3>DocuMind Capabilities</h3>
   DocuMind is an intelligent AI document assistant designed to read, analyze, and answer questions based on the content of uploaded documents.
   
   <h4>Supported File Types</h4>
   1. PDF
   2. XLSX
   3. TXT
   4. DOCX
   5. CSV
   6. PPTX
   7. Markdown and HTML (MD, HTML)
   8. Code files (Python, JavaScript, TypeScript, Java, Go, C, C++, SQL and more)
   9. Config and log files (JSON, YAML, TOML, XML, LOG)
   10. Images (JPG, PNG, etc.)
   
   <h4>Ideal Users</h4>
   1. Students
   2. Researchers
   3. Developers
   4. Business analysts
   5. Support teams

2. If the user specifically asks "What file types can you analyze?" or similar, respond ONLY with the list of Supported File Types. Do NOT include the "DocuMind Capabilities" heading, the description, or the "Ideal Users" section.

3. If the user just says greetings (like "hi", "hello") or asks who you are, respond ONLY with your identity as DocuMind and a brief instruction on how to use you. DO NOT list example use cases unless explicitly asked.
4. ONLY if the user specifically asks for example use cases, provide exactly 3 example use cases from the list below.
5. If you are answering questions based on provided context, answer accurately using ONLY that context.
6. Each context excerpt starts with a line like [Source: file | location]. The location can be a page, slide, sheet and rows, section, or line numbers with function and class names for code. When it helps, say where the answer comes from (for example "in billing.py, lines 40 to 72" or "on page 3"). Do not copy the bracketed line itself.
7. For code questions, explain what the code does, name the exact functions, classes, variables and files involved, and quote the relevant code when useful.

RULES FOR YOUR RESPONSE FORMATTING:
1. Use numbers (1., 2., 3.) for lists.
2. DO NOT use markdown formatting (** bold, * or - bullets, # headings). Hyphens and underscores that are part of content are fine and must be kept exactly: names like calculate_discount, codes like W-220, dates, phone numbers and negative numbers.
3. Use increased font size for Headings by wrapping them in <h3></h3> HTML tags.
4. Use slightly smaller font size for Subheadings by wrapping them in <h4></h4> HTML tags.
5. Split long answers into paragraphs using a single empty line (two newline characters: \\n\\n). Do not use multiple empty lines.
6. CRITICAL: When writing numbered lists, the number MUST be on the exact same line as the text. DO NOT put a newline after the number. Example: "1. Financial Analysis" NOT "1.\\nFinancial Analysis".
7. Show code inside <pre><code></code></pre> for blocks and <code></code> for short names, never with ``` fences. Inside them write < as &lt;, > as &gt; and & as &amp;, and copy the code exactly.

EXAMPLE USE CASES (Only use if explicitly asked):
1. Financial Analysis & Auditing
Upload: Quarterly Reports, Invoices (XLSX), Tax Documents.
Query: "What was our total expenditure on cloud hosting in Q3 compared to Q2?"

2. Customer Support & Knowledge Base
Upload: Product Manuals, FAQ Sheets, Technical Specs.
Query: "My device is blinking a red light. What does the manual say this means?"

3. Legal & Compliance
Upload: Contracts, NDAs, Terms of Service.
Query: "Does this NDA have a non-compete clause? If so, how long does it last?"
"""

    user_prompt = f"""Context from uploaded documents (if any):
{context if context else "No documents uploaded or selected."}

User Input: {question}"""

    try:
        if photos:
            # The question is about photos: the image model looks at them again and answers,
            # instead of the chat model (one model in memory at a time).
            names = "\n".join(f"{i + 1}. {name}" for i, (name, _) in enumerate(photos))
            photo_prompt = (f"The images attached to this message are these files, in this order:\n{names}\n"
                            "Look at the images to answer. Excerpts marked 'image description' are earlier notes "
                            "about photos; trust what you see in the attached images over those notes. "
                            "Say which file each part of your answer comes from.\n\n" + user_prompt)
            raw_answer = await run_in_threadpool(ai.answer_with_images, system_prompt, photo_prompt, photos)
        else:
            raw_answer = await run_in_threadpool(ai.chat, system_prompt, user_prompt)
        final_answer = clean_llm_response(raw_answer)
    except ai.AIError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {str(e)}")

    if session_id:
        storage.add_messages(session_id, [("user", question), ("assistant", final_answer)])
    return {"answer": final_answer}


MAX_PHOTOS_PER_QUESTION = 1 if appdata.LOW_MEMORY else 3   # each photo is ~800 tokens of the image model's window
PHOTO_EXCERPTS = 5 if appdata.LOW_MEMORY else 8             # text excerpts sent along with photos
PHOTO_SCORE_MARGIN = 0.05         # photos scoring this close to the best match are looked at too


def _photos_to_look_at(results) -> list:
    """When the best match for a question is a photo description, return the matching photos
    (best first, at most 3) so the image model can look at them again. Otherwise none."""
    if not results or not ai.vision_model():
        return []
    is_photo = lambda r: "| image description]" in r["content"].split("\n", 1)[0] and r.get("sha256")
    if not is_photo(results[0]):
        return []
    best, photos, seen = results[0]["similarity"], [], set()
    for r in results:
        if not is_photo(r) or r["similarity"] < best - PHOTO_SCORE_MARGIN or r["sha256"] in seen:
            continue
        data = storage.load_image(r["sha256"])
        if data:
            name = r["content"].split("\n", 1)[0][len("[Source: "):].split(" | ")[0]
            photos.append((name, data))
            seen.add(r["sha256"])
        if len(photos) == MAX_PHOTOS_PER_QUESTION:
            break
    return photos


def _retrieve(session_id: str, question: str, doc_ids: list[str]):
    """Returns (context text, photos for the image model)."""
    model = ai.embed_model_id()
    # Documents indexed with another provider's model (the user switched AI in Settings):
    # re-embed their stored chunks with the current model first, so vectors are comparable.
    for doc in storage.documents_with_other_model(doc_ids, model):
        texts = storage.chunk_texts(doc["doc_id"])
        if texts:
            print(f"Re-embedding {doc['filename']} for {model}")
            storage.replace_embeddings(doc["doc_id"], model, ai.embed(texts, kind="passage"))
    q_emb = ai.embed([question], kind="query")[0]
    results = storage.search_chunks(session_id, q_emb, doc_ids, model, top_k=15)
    photos = _photos_to_look_at(results)
    # The image model's window is smaller than the chat model's: keep its excerpts short.
    used = results[:PHOTO_EXCERPTS] if photos else results
    return "\n\n".join(r["content"] for r in used), photos

@app.delete("/session/{session_id}")
def delete_session_route(session_id: str):
    storage.clear_session(session_id)
    return {"status": "cleared"}

@app.delete("/session/{session_id}/document/{doc_id}")
def delete_document_route(session_id: str, doc_id: str):
    storage.delete_document(session_id, doc_id)
    return {"status": "deleted"}