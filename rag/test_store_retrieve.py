import os
import json
import time
import psycopg
from sentence_transformers import SentenceTransformer
from pgvector.psycopg import register_vector

# Config
DB_NAME = os.getenv("RAG_DB", "rag")
DB_USER = os.getenv("RAG_DB_USER", "rag_user")
DB_PASS = os.getenv("RAG_DB_PASS", "")
DB_HOST = os.getenv("RAG_DB_HOST", "localhost")
DB_PORT = os.getenv("RAG_DB_PORT", "5432")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
TEST_DOC_TITLE = "test-doc"
TEST_DOC_SOURCE = "smoke_test"   # Added: source is NOT NULL
TEST_CHUNK = "This is a short test chunk about distributed systems and scaling."

def get_conn():
    conn_str = f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT}"
    if DB_PASS:
        conn_str += f" password={DB_PASS}"
    return psycopg.connect(conn_str)

def main():
    print("Loading embedder:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)
    vec = embedder.encode([TEST_CHUNK])[0]
    print("Embedding dimension detected:", len(vec))

    # Connect DB
    conn = get_conn()
    register_vector(conn)  # register pgvector adapter
    with conn.cursor() as cur:
        # Insert document (now with source)
        cur.execute(
            "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id;",
            (TEST_DOC_TITLE, TEST_DOC_SOURCE)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        print("Inserted document id:", doc_id)

        # Insert chunk (metadata can be an empty JSON object if not needed)
        cur.execute(
            "INSERT INTO chunks (document_id, content, embedding, metadata) VALUES (%s, %s, %s, %s) RETURNING id;",
            (doc_id, TEST_CHUNK, vec, json.dumps({"source": "smoke_test"}))
        )
        chunk_id = cur.fetchone()[0]
        conn.commit()
        print("Inserted chunk id:", chunk_id)

        time.sleep(0.5)

        # Similarity search
        cur.execute(
            "SELECT id, content, metadata FROM chunks ORDER BY embedding <-> %s LIMIT 3;",
            (vec,)
        )
        rows = cur.fetchall()
        print("Top results:")
        for r in rows:
            print("id:", r[0], "content:", r[1], "metadata:", r[2])

    conn.close()
    print("Smoke test completed successfully.")

if __name__ == "__main__":
    main()
