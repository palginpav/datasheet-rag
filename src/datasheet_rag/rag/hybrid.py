"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF combines rankings without needing the underlying scores to be
commensurable — it sums 1/(c + rank) across retrievers, so a chunk ranked
high by *either* the dense or the BM25 retriever floats up. That is exactly
what part-number queries need: dense supplies semantic recall, BM25 supplies
the exact-token hit, and RRF lets the lexical match win when it should
without discarding dense's coverage elsewhere.

The fusion is a pure function (tested); the retriever wraps any two (or more)
sub-retrievers behind the same Retriever protocol, so the pipeline, eval, and
CLI are unchanged.
"""

from __future__ import annotations

from datasheet_rag.rag.retrieve import RetrievedChunk, Retriever

RRF_C = 60  # standard constant from Cormack et al. 2009


def rrf_fuse(
    rankings: list[list[str]], c: int = RRF_C, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one ranking. Returns (id, score) sorted desc.
    Optional per-ranking weights let a stronger retriever dominate the fusion —
    useful when one ranker (here BM25) is much weaker than the other (dense)."""
    w = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, wi in zip(rankings, w, strict=True):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + wi / (c + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        retrievers: list[Retriever],
        c: int = RRF_C,
        pool: int = 30,
        weights: list[float] | None = None,
    ) -> None:
        """retrievers: the sub-retrievers to fuse. pool: how many to pull from
        each before fusing (wider than k so a chunk ranked mid-list by one
        retriever can still be rescued by the other). weights: per-retriever
        fusion weights (default equal)."""
        self._retrievers = retrievers
        self._c = c
        self._pool = pool
        self._weights = weights

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        per_retriever: list[list[RetrievedChunk]] = [
            r.retrieve(query, k=self._pool) for r in self._retrievers
        ]
        by_id: dict[str, RetrievedChunk] = {}
        for hits in per_retriever:
            for h in hits:
                by_id.setdefault(h.chunk_id, h)
        fused = rrf_fuse(
            [[h.chunk_id for h in hits] for hits in per_retriever],
            c=self._c,
            weights=self._weights,
        )
        out: list[RetrievedChunk] = []
        for cid, score in fused[:k]:
            h = by_id[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=h.chunk_id, text=h.text, score=score, part=h.part,
                    manufacturer=h.manufacturer, kind=h.kind, page=h.page,
                    section_path=h.section_path,
                )
            )
        return out
