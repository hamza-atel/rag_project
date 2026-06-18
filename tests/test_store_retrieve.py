import os
import json
import tempfile
import pytest
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# Import the modules to test
from rag.store import process_file, get_db_connection, insert_document, insert_chunks
from rag.retrieve import retrieve
from rag.embeddings.chunker import TokenChunker

@pytest.fixture
def db_conn():
    conn = psycopg.connect(
        dbname=os.getenv("RAG_DB", "rag"),
        user=os.getenv("RAG_DB_USER", "rag_user"),
        password=os.getenv("RAG_DB_PASS", ""),
        host=os.getenv("RAG_DB_HOST", "localhost"),
        port=os.getenv("RAG_DB_PORT", "5432"),
    )
    register_vector(conn)
    yield conn
    # Clean up after test: delete inserted documents
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE source LIKE '/tmp/%');")
        cur.execute("DELETE FROM documents WHERE source LIKE '/tmp/%';")
        conn.commit()
    conn.close()

@pytest.fixture
def embedder():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

@pytest.fixture
def chunker(embedder):
    return TokenChunker(embedder.tokenizer, chunk_size_tokens=50, overlap_tokens=10)

def test_end_to_end_store_retrieve(db_conn, embedder, chunker):
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is an integration test document. It should be stored and retrieved.")
        temp_path = f.name
    try:
        # Process file
        process_file(temp_path, "integration_test", embedder, chunker, batch_size=1, conn=db_conn)
        # Retrieve
        results = retrieve("integration test document", embedder, db_conn, top_k=1)
        assert len(results) == 1
        assert "integration test document" in results[0]["content"].lower()
        assert results[0]["score"] > 0.5
    finally:
        os.unlink(temp_path)
