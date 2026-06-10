"""Embedding wrapper for nomic-embed-text-v1.5.

Nomic's model is *asymmetric*: documents and queries must be prefixed with
``search_document:`` / ``search_query:`` respectively or retrieval quality
degrades badly — an easy detail to miss, so it lives in exactly one place.

The sentence-transformers import is deferred so that modules importing the
:class:`Embedder` protocol (tests, the store) never pay the torch startup
cost.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Minimal embedding interface; implementations must be deterministic."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class NomicEmbedder:
    """nomic-embed-text-v1.5 via sentence-transformers (768-dim)."""

    MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(
        self, device: str | None = None, batch_size: int = 32, max_seq_length: int = 2048
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.MODEL_NAME, trust_remote_code=True, device=device
        )
        # The model accepts up to 8192 tokens, but batches pad to the longest
        # member: one oversized table chunk in a large batch exhausts GPU
        # memory. 2048 tokens covers every chunk the chunker emits except the
        # documented irreducible-row outliers, which truncate harmlessly.
        self._model.max_seq_length = max_seq_length
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [f"search_query: {text}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vec.tolist()
