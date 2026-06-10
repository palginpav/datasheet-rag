"""Cross-encoder reranking over a first-stage retriever.

The ablation showed the residual failure is *within-document* chunk ranking:
dense routes to the right part every time, but the answer-bearing summary
chunk competes with hundreds of same-part siblings. A bi-encoder scores query
and chunk independently; a cross-encoder reads them *together*, so it can tell
that a chunk literally stating "32 KB flash" answers "how much flash" better
than a sibling that merely mentions flash timing.

This retriever pulls a wide candidate pool from a first stage (dense) and
reorders it with a cross-encoder, returning the top-k. Heavier per query, so
it sits behind the same Retriever protocol and is opt-in.
"""

from __future__ import annotations

from datasheet_rag.rag.retrieve import RetrievedChunk, Retriever


class RerankRetriever:
    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(
        self,
        first_stage: Retriever,
        model: str | None = None,
        pool: int = 30,
        device=None,
        encoder=None,
    ) -> None:
        self._first = first_stage
        self._pool = pool
        if encoder is not None:  # injectable for testing
            self._encoder = encoder
        else:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(model or self.DEFAULT_MODEL, device=device)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        candidates = self._first.retrieve(query, k=self._pool)
        if not candidates:
            return []
        scores = self._encoder.predict([(query, c.text) for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda cs: cs[1], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id, text=c.text, score=float(s), part=c.part,
                manufacturer=c.manufacturer, kind=c.kind, page=c.page,
                section_path=c.section_path,
            )
            for c, s in ranked[:k]
        ]
