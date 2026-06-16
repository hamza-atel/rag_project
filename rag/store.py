#!/usr/bin/env python3
"""
Ingest documents into the RAG database.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.embeddings.chunker import TokenChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_db_connection():
    conn = psycopg.connect(
        dbname=os.getenv("RAG_DB", "rag"),
        user=os.getenv("RAG_DB_USER", "rag_user"),
        password=os.getenv("RAG_DB_PASS", ""),
        host=os.getenv("RAG_DB_HOST", "localhost"),
        port=os.getenv("RAG_DB_PORT", "5432"),
    )
    register_vector(conn)
    return conn


def retry_on_error(max_retries=3, delay=1, backoff=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except psycopg.OperationalError as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"DB error: {e}. Retrying in {_delay}s...")
                    time.sleep(_delay)
                    _delay *= backoff
            return None
        return wrapper
    return decorator


@retry_on_error()
def insert_document(conn, title: str, source: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id",
            (title, source),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        return doc_id


@retry_on_error()
def insert_chunks(conn, doc_id: int, chunks: List[Dict[str, Any]]):
    with conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO chunks
                (document_id, content, embedding, metadata, chunk_index, char_start, char_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc_id,
                    chunk["content"],
                    chunk["embedding"],       # already a list
                    chunk["metadata"],        # already a JSON string
                    chunk["chunk_index"],
                    chunk["char_start"],
                    chunk["char_end"],
                ),
            )
        conn.commit()


def read_text_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def process_file(
    file_path: Path,
    title: str,
    embedder: SentenceTransformer,
    chunker: TokenChunker,
    batch_size: int,
    conn,
) -> None:
    logger.info(f"Processing {file_path}")
    text = read_text_file(file_path)
    if not text.strip():
        logger.warning(f"File {file_path} is empty. Skipping.")
        return

    raw_chunks = chunker.chunk_text(text)
    if not raw_chunks:
        logger.warning(f"No chunks generated for {file_path}.")
        return

    chunk_texts = [c[0] for c in raw_chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True, batch_size=batch_size)

    doc_id = insert_document(conn, title, str(file_path))
    chunk_records = []
    for idx, ((content, start, end), emb) in enumerate(zip(raw_chunks, embeddings)):
        # Convert numpy array to list for pgvector
        emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
        # Convert metadata dict to JSON string
        metadata_json = json.dumps({"source": str(file_path)})
        chunk_records.append({
            "content": content,
            "embedding": emb_list,
            "metadata": metadata_json,
            "chunk_index": idx,
            "char_start": start,
            "char_end": end,
        })
    insert_chunks(conn, doc_id, chunk_records)
    logger.info(f"Inserted {len(chunk_records)} chunks for document {title} (id={doc_id})")


def process_directory(
    dir_path: Path,
    embedder: SentenceTransformer,
    chunker: TokenChunker,
    batch_size: int,
    conn,
) -> None:
    for file_path in dir_path.rglob("*.txt"):
        title = file_path.stem
        process_file(file_path, title, embedder, chunker, batch_size, conn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to a single text file or a directory")
    parser.add_argument("--title", help="Title for the document (ignored for directories)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    logger.info(f"Loading embedding model {args.model}...")
    embedder = SentenceTransformer(args.model)
    chunker = TokenChunker(embedder.tokenizer, args.chunk_size, args.overlap)

    conn = get_db_connection()
    path = Path(args.file)

    if path.is_file():
        if not args.title:
            args.title = path.stem
        process_file(path, args.title, embedder, chunker, args.batch_size, conn)
    elif path.is_dir():
        process_directory(path, embedder, chunker, args.batch_size, conn)
    else:
        logger.error(f"Path {path} does not exist.")
        sys.exit(1)

    conn.close()
    logger.info("Ingestion completed.")


if __name__ == "__main__":
    main()
