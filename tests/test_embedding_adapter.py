import pytest
import json
import psycopg
from pgvector.psycopg import register_vector
from unittest.mock import MagicMock, patch
import numpy as np

from rag.retrieve import retrieve

@pytest.fixture
def db_conn():
    # Use a test database (should be configured separately)
    # For CI, we can create a temporary database; here we assume a test DB exists.
    import os
    conn = psycopg.connect(
        dbname=os.getenv("RAG_DB", "rag"),
        user=os.getenv("RAG_DB_USER", "rag_user"),
        password=os.getenv("RAG_DB_PASS", ""),
        host=os.getenv("RAG_DB_HOST", "localhost"),
        port=os.getenv("RAG_DB_PORT", "5432"),
    )
    register_vector(conn)
    yield conn
    conn.close()

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    def encode(text, normalize_embeddings=False):
        # Return a dummy vector of length 1024 with a pattern that will match itself
        if isinstance(text, list):
            vec = np.zeros((len(text), 1024), dtype=np.float32)
            for i, t in enumerate(text):
                vec[i][0] = 1.0
            return vec
        else:
            vec = np.zeros(1024, dtype=np.float32)
            vec[0] = 1.0
            return vec
    embedder.encode = encode
    return embedder

def test_retrieve_mocked(db_conn, mock_embedder):
    # Insert a test chunk
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM chunks;")  # clean up
        cur.execute("DELETE FROM documents;")
        cur.execute("INSERT INTO documents (title, source) VALUES ('test', 'unit') RETURNING id")
        doc_id = cur.fetchone()[0]
        dummy_embedding = [0.0]*1024
        dummy_embedding[0] = 1.0
        cur.execute(
            "INSERT INTO chunks (document_id, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
            (doc_id, "unit test content", dummy_embedding, json.dumps({}))
        )
        db_conn.commit()
    results = retrieve("query", mock_embedder, db_conn, top_k=1)
    assert len(results) == 1
    assert results[0]["content"] == "unit test content"
    assert results[0]["score"] > 0.99  # because vectors are identical
