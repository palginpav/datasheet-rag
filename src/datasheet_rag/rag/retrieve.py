"""Retriever protocol and the dense baseline implementation.

The interface is the Phase-4 swap point: hybrid (BM25 + RRF) and reranked
variants will implement the same ``retrieve`` signature, so the pipeline,
eval harness, and CLI never change when the retrieval strategy does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from datasheet_rag.rag.embed import Embedder
from datasheet_rag.rag.store import ChunkStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    part: str
    manufacturer: str
    kind: str
    page: int | None
    section_path: list[str]


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]: ...


class DenseRetriever:
    """Cosine top-k over the Chroma store."""

    def __init__(self, store: ChunkStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        hits = self._store.query(self._embedder.embed_query(query), k=k)
        return [
            RetrievedChunk(
                chunk_id=h["chunk_id"],
                text=h["text"],
                score=h["score"],
                part=h["metadata"].get("part", ""),
                manufacturer=h["metadata"].get("manufacturer", ""),
                kind=h["metadata"].get("kind", ""),
                page=h["metadata"].get("page"),
                section_path=h["metadata"].get("section_path", []),
            )
            for h in hits
        ]
