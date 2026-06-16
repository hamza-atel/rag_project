#!/usr/bin/env python3
"""
Retrieve relevant chunks from the database.
"""
import argparse
import json
import logging
import os

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
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


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    conn,
    top_k: int = 5,
) -> list:
    query_embedding = embedder.encode(query, normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.document_id, c.content, c.metadata,
                   1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding.tolist(), query_embedding.tolist(), top_k),
        )
        rows = cur.fetchall()
    return [
        {"document_id": row[0], "content": row[1], "metadata": row[2], "score": float(row[3])}
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    embedder = SentenceTransformer(args.model)
    conn = get_db_connection()
    results = retrieve(query=args.query, embedder=embedder, conn=conn, top_k=args.top_k)
    print(json.dumps(results, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
