# RAG Project

## 📖 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** stack using:
- **Ollama** for local LLM inference
- **Gemma-4-E4B Uncensored (Q4_K_P)** model
- **PostgreSQL + pgvector** for vector storage and similarity search
- **BGE-Large-EN-v1.5** embedding model for English text
- **Python pipeline** for chunking, embedding, storing, and retrieving documents

The goal is to provide a reproducible, maintainable, and scalable foundation for building RAG applications.

Quick start
1. Activate venv: `source rag_env/bin/activate`
2. Create DB and user: see INSTALLATION.md
3. Apply schema: `psql -U rag_user -d rag -f db/schema.sql`
4. Register model: `ollama create gemma-e4b-uncensored -f ollama/Modelfile`
5. Run test script: `python rag/test_store_retrieve.py`

Architecture
- Model runner: Ollama
- Model: models/Gemma-4-...gguf
- Vector DB: PostgreSQL + pgvector
- Embeddings: BAAI/bge-large-en-v1.5

