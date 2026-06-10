"""Retriever factory — single place that builds named retriever configs.

Keeps the CLIs (ask, eval) and the ablation in sync: "dense" is the default,
"dense+rerank" is the Phase-4 winner (cross-encoder over the dense pool).
"""

from __future__ import annotations

from pathlib import Path

from datasheet_rag.rag.retrieve import Retriever


def build_retriever(
    name: str = "dense",
    *,
    store_dir: str | Path = "chroma",
    parsed_dir: str | Path = "corpus/parsed",
    device: str | None = None,
) -> Retriever:
    from datasheet_rag.rag.embed import NomicEmbedder
    from datasheet_rag.rag.retrieve import DenseRetriever
    from datasheet_rag.rag.store import ChunkStore

    dense = DenseRetriever(ChunkStore(store_dir), NomicEmbedder(device=device))
    if name == "dense":
        return dense
    if name == "bm25":
        from datasheet_rag.rag.bm25 import BM25Retriever, load_all_chunks

        return BM25Retriever(load_all_chunks(parsed_dir))
    if name in ("hybrid", "hybrid-w3"):
        from datasheet_rag.rag.bm25 import BM25Retriever, load_all_chunks
        from datasheet_rag.rag.hybrid import HybridRetriever

        bm25 = BM25Retriever(load_all_chunks(parsed_dir))
        weights = [3.0, 1.0] if name == "hybrid-w3" else None
        return HybridRetriever([dense, bm25], weights=weights)
    if name in ("rerank", "dense+rerank"):
        from datasheet_rag.rag.rerank import RerankRetriever

        return RerankRetriever(dense, pool=30, device=device)
    raise ValueError(f"unknown retriever: {name}")
