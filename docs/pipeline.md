
---

### PIPELINE.md

```markdown
# Pipeline

This document describes the RAG pipeline components, chunking strategy, embedding workflow, retrieval logic, and example usage. It is focused on text-only embeddings using BGE-Large-EN-v1.5 for the current baseline.

---

## Overview
**Components**
- **store.py**: ingest documents, chunk, embed, and store into Postgres
- **retrieve.py**: embed query, run similarity search, return top chunks
- **pipeline.py**: orchestrate retrieval, assemble prompt, call model via Ollama
- **embeddings/**: utilities for embedding model and adapters
- **prompts/**: prompt templates and formatting helpers

---

## Data Flow
1. **Ingest**: read source (file, URL, repo)
2. **Chunk**: split into chunks with overlap
3. **Embed**: compute embedding vector for each chunk
4. **Store**: insert into `documents` and `chunks` tables
5. **Retrieve**: embed user query, run similarity search, filter by modality/language
6. **Augment**: assemble retrieved chunks into prompt
7. **Respond**: call Ollama model with prompt and return result

---

## Chunking Strategy
**Text documents**
- **Token-based** chunking using a tokenizer compatible with your embedding model
- **Parameters**
  - **chunk_size_tokens**: 500
  - **overlap_tokens**: 50
- **Algorithm**
  - Tokenize text
  - Slide window of `chunk_size_tokens` with `overlap_tokens`
  - Convert tokens back to text for embedding
  - Record `char_start`, `char_end`, and `chunk_index` in metadata

**Code repositories**
- Prefer function-level or file-level chunks
- Preserve code fences and file path in metadata

**Images audio video**
- Not in baseline; store as modality with precomputed embeddings if added later

**Example chunker snippet**
```python
def chunk_text(text, tokenizer, chunk_size=500, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append((chunk_text, start, end))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

