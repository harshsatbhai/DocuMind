"""SQLite storage for the DocuMind desktop app.

One file, documind.db, in the app-data folder: documents, chunks (with embeddings), chat messages.
Nothing to install: SQLite ships with Python.

Embeddings are stored L2-normalised as float32 BLOBs, so cosine similarity is a dot product.
Search is exact (brute force) with numpy, streamed in batches so memory stays small; on a laptop
this is well under a second for tens of thousands of chunks.
Each document records which embedding model made its vectors (e.g. "nvidia:nvidia/nemotron-3.5-embed-1b"
or "ollama:nomic-embed-text"), so switching AI provider never mixes incompatible vectors.
"""
import sqlite3
import threading
import time
from contextlib import contextmanager

import numpy as np

from .appdata import data_dir

_local = threading.local()
_write_lock = threading.Lock()
SEARCH_BATCH = 4096


def _db_path():
    return str(data_dir() / "documind.db")


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_db_path(), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


@contextmanager
def _write():
    with _write_lock:
        conn = _conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db():
    with _write() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id       TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL,
            filename     TEXT NOT NULL,
            sha256       TEXT NOT NULL,
            embed_model  TEXT NOT NULL,
            dim          INTEGER NOT NULL,
            chunks_count INTEGER NOT NULL,
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS documents_session ON documents(session_id);
        CREATE INDEX IF NOT EXISTS documents_hash ON documents(sha256, embed_model);
        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY,
            doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            idx       INTEGER NOT NULL,
            content   TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)


def _normalise(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ---------------------------------------------------------------- documents
def insert_document(session_id, doc_id, filename, sha256, embed_model, chunks, embeddings):
    vecs = _normalise(embeddings)
    with _write() as c:
        c.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                  (doc_id, session_id, filename, sha256, embed_model, vecs.shape[1], len(chunks), time.time()))
        c.executemany("INSERT INTO chunks (doc_id, idx, content, embedding) VALUES (?,?,?,?)",
                      [(doc_id, i, text, vecs[i].tobytes()) for i, text in enumerate(chunks)])
        c.execute("INSERT OR REPLACE INTO meta VALUES ('last_session', ?)", (session_id,))


def copy_cached_document(sha256, embed_model, session_id, doc_id, filename):
    """Same file already indexed with the same model (any session)? Copy its chunks: no re-parsing,
    no re-embedding. Returns the chunk count, or None when there is nothing to reuse."""
    c = _conn()
    src = c.execute("SELECT doc_id, dim FROM documents WHERE sha256=? AND embed_model=? LIMIT 1",
                    (sha256, embed_model)).fetchone()
    if src is None:
        return None
    with _write() as w:
        n = w.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (src["doc_id"],)).fetchone()[0]
        w.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                  (doc_id, session_id, filename, sha256, embed_model, src["dim"], n, time.time()))
        w.execute("INSERT INTO chunks (doc_id, idx, content, embedding) "
                  "SELECT ?, idx, content, embedding FROM chunks WHERE doc_id=? ORDER BY idx",
                  (doc_id, src["doc_id"]))
        w.execute("INSERT OR REPLACE INTO meta VALUES ('last_session', ?)", (session_id,))
    return n


def list_documents(session_id):
    rows = _conn().execute("SELECT doc_id, filename, chunks_count, embed_model FROM documents "
                           "WHERE session_id=? ORDER BY created_at", (session_id,)).fetchall()
    return [dict(r) for r in rows]


def documents_with_other_model(doc_ids, embed_model):
    if not doc_ids:
        return []
    q = ",".join("?" * len(doc_ids))
    rows = _conn().execute(f"SELECT doc_id, filename FROM documents WHERE doc_id IN ({q}) AND embed_model != ?",
                           (*doc_ids, embed_model)).fetchall()
    return [dict(r) for r in rows]


def chunk_texts(doc_id):
    return [r[0] for r in _conn().execute("SELECT content FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))]


def replace_embeddings(doc_id, embed_model, embeddings):
    """Re-embed a document's stored chunks with another model (after the user switches provider)."""
    vecs = _normalise(embeddings)
    with _write() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))]
        c.executemany("UPDATE chunks SET embedding=? WHERE id=?", [(vecs[i].tobytes(), ids[i]) for i in range(len(ids))])
        c.execute("UPDATE documents SET embed_model=?, dim=? WHERE doc_id=?", (embed_model, vecs.shape[1], doc_id))


def delete_document(session_id, doc_id):
    with _write() as c:
        c.execute("DELETE FROM documents WHERE session_id=? AND doc_id=?", (session_id, doc_id))


def clear_session(session_id):
    with _write() as c:
        c.execute("DELETE FROM documents WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM meta WHERE key='last_session' AND value=?", (session_id,))


# ---------------------------------------------------------------- search
def search_chunks(session_id, query_embedding, doc_ids, embed_model, top_k=15):
    if not doc_ids:
        return []
    q = _normalise([query_embedding])[0]
    marks = ",".join("?" * len(doc_ids))
    cur = _conn().execute(
        f"SELECT c.content, c.embedding FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        f"WHERE d.session_id=? AND d.embed_model=? AND c.doc_id IN ({marks})",
        (session_id, embed_model, *doc_ids))
    best_scores = np.empty(0, dtype=np.float32)
    best_texts: list[str] = []
    while True:
        rows = cur.fetchmany(SEARCH_BATCH)
        if not rows:
            break
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        if mat.shape[1] != q.shape[0]:
            continue
        scores = np.concatenate([best_scores, mat @ q])
        texts = best_texts + [r[0] for r in rows]
        keep = np.argsort(-scores)[:top_k]
        best_scores, best_texts = scores[keep], [texts[i] for i in keep]
    return [{"content": t, "similarity": float(s)} for t, s in zip(best_texts, best_scores)]


# ---------------------------------------------------------------- chat history
def add_messages(session_id, pairs):
    with _write() as c:
        now = time.time()
        c.executemany("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                      [(session_id, role, content, now + i * 1e-6) for i, (role, content) in enumerate(pairs)])
        c.execute("INSERT OR REPLACE INTO meta VALUES ('last_session', ?)", (session_id,))


def list_messages(session_id):
    rows = _conn().execute("SELECT id, role, content FROM messages WHERE session_id=? ORDER BY created_at, id",
                           (session_id,)).fetchall()
    return [dict(r) for r in rows]


def last_session():
    row = _conn().execute("SELECT value FROM meta WHERE key='last_session'").fetchone()
    return row[0] if row else None


def delete_all():
    """Erase every document, chunk, embedding and chat message. VACUUM and a WAL checkpoint make
    SQLite rewrite the file, so deleted text doesn't linger in free pages or the write-ahead log."""
    with _write() as c:
        c.execute("DELETE FROM chunks")
        c.execute("DELETE FROM documents")
        c.execute("DELETE FROM messages")
        c.execute("DELETE FROM meta")
    with _write_lock:
        conn = _conn()
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
