import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(2048)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_session_idx ON chunks (session_id);")
    conn.commit()
    cur.close()
    conn.close()

def insert_chunks(session_id: str, doc_id: str, filename: str, chunks: list[str], embeddings: list[list[float]]):
    conn = get_conn()
    cur = conn.cursor()
    for content, emb in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO chunks (session_id, doc_id, filename, content, embedding) VALUES (%s, %s, %s, %s, %s)",
            (session_id, doc_id, filename, content, emb),
        )
    conn.commit()
    cur.close()
    conn.close()

def search_chunks(session_id: str, query_embedding: list[float], doc_ids: list[str] = None, top_k: int = 20):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    if doc_ids and len(doc_ids) > 0:
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE session_id = %s AND doc_id = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, session_id, doc_ids, query_embedding, top_k),
        )
    else:
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE session_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, session_id, query_embedding, top_k),
        )
        
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def clear_session(session_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
    conn.commit()
    cur.close()
    conn.close()

def delete_document(session_id: str, doc_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE session_id = %s AND doc_id = %s", (session_id, doc_id))
    conn.commit()
    cur.close()
    conn.close()