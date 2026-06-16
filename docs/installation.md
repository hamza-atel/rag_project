# RAG Project

**Purpose:** Local RAG stack using Gemma model, Ollama, PostgreSQL+pgvector, and bge embeddings.

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
# Installation

OS: Linux Mint 22.3

1. System packages
   sudo apt update
   sudo apt install -y build-essential git curl postgresql postgresql-contrib postgresql-server-dev-all

2. Python
   python3 -m venv ~/rag_project/rag_env
   source ~/rag_project/rag_env/bin/activate
   pip install -r requirements.txt

3. pgvector
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector && make && sudo make install
   psql -U rag_user -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

4. Ollama
   curl -fsSL https://ollama.com/install.sh | sh

