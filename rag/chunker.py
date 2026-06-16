"""
Token-based text chunker using a SentenceTransformers tokenizer.
Respects the tokenizer's maximum sequence length.
"""
import logging
import os
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

if os.environ.get("HF_TOKEN"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenChunker:
    def __init__(
        self,
        tokenizer=None,
        chunk_size_tokens: int = 500,
        overlap_tokens: int = 50,
        max_tokens: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens

        if max_tokens is None:
            if self.tokenizer is not None and hasattr(self.tokenizer, "max_length"):
                self.max_tokens = self.tokenizer.max_length
            else:
                self.max_tokens = 512
        else:
            self.max_tokens = max_tokens

        if self.chunk_size > self.max_tokens:
            logger.warning(
                f"chunk_size ({self.chunk_size}) > max_tokens ({self.max_tokens}). "
                f"Capping chunk size to {self.max_tokens}."
            )
            self.chunk_size = self.max_tokens

        if self.tokenizer is None:
            logger.warning("No tokenizer provided; using whitespace fallback.")
        else:
            if not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
                raise ValueError("Tokenizer must have 'encode' and 'decode' methods.")

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            return text.split()
        if hasattr(self.tokenizer, "tokenize"):
            return self.tokenizer.tokenize(text)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return [self.tokenizer.decode([i]) for i in ids]

    def _detokenize(self, tokens: List[str]) -> str:
        if self.tokenizer is None:
            return " ".join(tokens)
        if hasattr(self.tokenizer, "convert_tokens_to_string"):
            return self.tokenizer.convert_tokens_to_string(tokens)
        return "".join(tokens)

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        tokens = self._tokenize(text)
        if not tokens:
            return []

        step = self.chunk_size - self.overlap
        if step <= 0:
            step = self.chunk_size

        chunks = []
        for start_idx in range(0, len(tokens), step):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            if end_idx - start_idx > self.max_tokens:
                end_idx = start_idx + self.max_tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self._detokenize(chunk_tokens)

            start_char = text.find(chunk_text)
            if start_char == -1:
                start_char = 0
            end_char = start_char + len(chunk_text)

            chunks.append((chunk_text, start_char, end_char))
            if end_idx == len(tokens):
                break
        return chunks

    @staticmethod
    def from_model_name(
        model_name: str,
        chunk_size: int = 500,
        overlap: int = 50,
        max_tokens: Optional[int] = None,
    ):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        tokenizer = model.tokenizer
        return TokenChunker(tokenizer, chunk_size, overlap, max_tokens)


def chunk_text_simple(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
    model_name: str = "BAAI/bge-large-en-v1.5",
    max_tokens: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    chunker = TokenChunker.from_model_name(model_name, chunk_size_tokens, overlap_tokens, max_tokens)
    return chunker.chunk_text(text)
