"""One storage API for both setups.

* Desktop app (default): SQLite in the user's app-data folder (store_sqlite).
* Web deployment: PostgreSQL + pgvector when DATABASE_URL is set (database.py, unchanged).
  Chat history, re-use of already-indexed files and provider switching are desktop-only features;
  on Postgres those calls are no-ops.
"""
import os

USE_POSTGRES = bool(os.environ.get("DATABASE_URL"))

if USE_POSTGRES:
    from . import database as _pg

    BACKEND = "postgres"

    def init_db():
        _pg.init_db()

    def insert_document(session_id, doc_id, filename, sha256, embed_model, chunks, embeddings):
        _pg.insert_chunks(session_id, doc_id, filename, chunks, embeddings)

    def copy_cached_document(*args):
        return None

    def search_chunks(session_id, query_embedding, doc_ids, embed_model, top_k=15):
        return _pg.search_chunks(session_id, query_embedding, doc_ids, top_k=top_k)

    def documents_with_other_model(doc_ids, embed_model):
        return []

    def chunk_texts(doc_id):
        return []

    def replace_embeddings(*args):
        pass

    def delete_document(session_id, doc_id):
        _pg.delete_document(session_id, doc_id)

    def clear_session(session_id):
        _pg.clear_session(session_id)

    def list_documents(session_id):
        return []

    def add_messages(session_id, pairs):
        pass

    def list_messages(session_id):
        return []

    def last_session():
        return None

    def delete_all():
        raise RuntimeError("Deleting all data is only available in the desktop app.")

    def save_image(sha256, jpeg):
        pass

    def load_image(sha256):
        return None
else:
    from .store_sqlite import (  # noqa: F401
        add_messages, chunk_texts, clear_session, copy_cached_document, delete_all, delete_document,
        documents_with_other_model, init_db, insert_document, last_session, list_documents,
        list_messages, load_image, replace_embeddings, save_image, search_chunks,
    )

    BACKEND = "sqlite"
