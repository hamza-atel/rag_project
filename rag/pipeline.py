#!/usr/bin/env python3
"""
Orchestrate retrieval and generation using Ollama.
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

from rag.retrieve import retrieve, get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_ollama(
    model: str,
    prompt: str,
    system: str = "",
    timeout: int = 60,
    retries: int = 1,
    base_url: str = "http://localhost:11434",
) -> str:
    url = f"{base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "system": system, "stream": False}
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.warning(f"Ollama call failed (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def build_prompt(query: str, chunks: list, template_path: str) -> str:
    template = Path(template_path).read_text()
    context = "\n\n".join([f"[{i+1}] {c['content']}" for i, c in enumerate(chunks)])
    return template.replace("{{query}}", query).replace("{{context}}", context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prompt-template", default="prompts/default.txt")
    parser.add_argument("--model", default="gemma-e4b-uncensored")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    embedder = SentenceTransformer(args.embedding_model)
    conn = get_db_connection()
    chunks = retrieve(args.query, embedder, conn, top_k=args.top_k)
    conn.close()

    if not chunks:
        logger.error("No chunks retrieved.")
        return

    prompt = build_prompt(args.query, chunks, args.prompt_template)
    response = call_ollama(
        model=args.model,
        prompt=prompt,
        system="You are a helpful assistant that answers questions based only on the provided context.",
        timeout=60,
        retries=1,
        base_url=args.ollama_url,
    )

    output = {"query": args.query, "response": response, "context_chunks": chunks}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
