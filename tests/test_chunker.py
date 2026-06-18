import pytest
from rag.embeddings.chunker import TokenChunker, chunk_text_simple

# Dummy tokenizer for testing (mimics SentenceTransformers)
class DummyTokenizer:
    def __init__(self, max_length=512):
        self.max_length = max_length
    def tokenize(self, text):
        return text.split()
    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)
    def encode(self, text, add_special_tokens=False):
        # For fallback path: return list of integers (dummy)
        return [ord(c) for c in text if c != ' ']
    def decode(self, ids):
        return ''.join(chr(i) for i in ids)

def test_chunker_basic():
    chunker = TokenChunker(tokenizer=DummyTokenizer(), chunk_size_tokens=3, overlap_tokens=1)
    text = "a b c d e f g"
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0][0] == "a b c"
    assert chunks[1][0] == "c d e"
    assert chunks[2][0] == "e f g"

def test_chunker_short_text():
    chunker = TokenChunker(tokenizer=DummyTokenizer(), chunk_size_tokens=10, overlap_tokens=2)
    text = "short"
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0][0] == "short"

def test_chunker_empty_text():
    chunker = TokenChunker(tokenizer=DummyTokenizer())
    chunks = chunker.chunk_text("")
    assert chunks == []

def test_chunker_respects_max_tokens():
    # Use a tokenizer with max_length=5
    tokenizer = DummyTokenizer(max_length=5)
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size_tokens=10, overlap_tokens=0)
    text = "a b c d e f g h i j"
    chunks = chunker.chunk_text(text)
    # Each chunk should have at most 5 tokens
    for chunk_text, _, _ in chunks:
        assert len(chunk_text.split()) <= 5

def test_chunker_unicode():
    # Use real tokenizer to ensure unicode handling (optional, but we can still test with dummy)
    chunker = TokenChunker(tokenizer=DummyTokenizer(), chunk_size_tokens=5, overlap_tokens=0)
    text = "Café München 中文"
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 1
    # The dummy tokenizer splits by spaces, so ensure it contains the words
    assert "Café" in chunks[0][0] or chunks[0][0] == text  # depending on tokenization
