# Schema

This document describes the database schema, embedding dimension, indexing strategy, metadata conventions, and migration notes for the RAG project.

## Purpose
Provide a stable, versioned schema for storing documents, chunks, and embeddings so retrieval is reproducible and efficient.

---

## Tables

**documents**
- **Purpose**: one row per logical document (PDF, article, file, repo).
- **Columns**
  - **id**: `SERIAL PRIMARY KEY`
  - **title**: `TEXT`
  - **created_at**: `TIMESTAMP DEFAULT NOW()`

**chunks**
- **Purpose**: store chunked content and its embedding for retrieval.
- **Columns**
  - **id**: `SERIAL PRIMARY KEY`
  - **document_id**: `INT REFERENCES documents(id)`
  - **content**: `TEXT`
  - **embedding**: `VECTOR(<DIM>)`  **replace `<DIM>` with actual embedding size**
  - **modality**: `TEXT`  **values**: `text`, `code`, `image`, `audio`, `video`
  - **language**: `TEXT`  **ISO 639-1 code when known**
  - **metadata**: `JSONB`  **freeform metadata: source, path, chunk_index, timestamps**
  - **created_at**: `TIMESTAMP DEFAULT NOW()`

**Example SQL**
```sql
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  title TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  document_id INT REFERENCES documents(id),
  content TEXT,
  embedding VECTOR(1024),
  modality TEXT,
  language TEXT,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

