"""BM25 sparse retrieval over the chunk corpus.

Dense embeddings capture meaning but blur exact tokens — which is why the
dense baseline loses part-number queries ("STM32C031C4 flash") amid sibling
datasheets. BM25 is the complement: a part number is a rare, high-IDF token,
so a lexical match pins the right document hard.

Tokenization keeps alphanumeric runs intact (so "STM32C031C4" stays one
token) and lowercases; nothing fancy, because the win comes from exact-token
overlap, not stemming. The index is built in memory from the chunk files —
fast at corpus scale (tens of thousands of chunks) and dependency-light.
"""

from __future__ import annotations

import re
from pathlib import Path

from datasheet_rag.ingest.models import Chunk, load_chunks
from datasheet_rag.rag.retrieve import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase; keep alphanumeric runs and internal -/ joiners so part
    numbers and units (e.g. 'stm32c031c4', 'v/µs'→'v', 's') survive as tokens."""
    return _TOKEN.findall(text.lower())


def load_all_chunks(parsed_dir: str | Path = "corpus/parsed") -> list[Chunk]:
    return [c for f in sorted(Path(parsed_dir).glob("*.chunks.jsonl")) for c in load_chunks(f)]


class BM25Retriever:
    """Okapi BM25 over chunk text, returning the Retriever protocol shape."""

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi  # optional [rag] dep; kept out of import path

        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i].chunk_id,
                text=self._chunks[i].text,
                score=float(scores[i]),
                part=self._chunks[i].part,
                manufacturer=self._chunks[i].manufacturer,
                kind=self._chunks[i].kind,
                page=self._chunks[i].page,
                section_path=self._chunks[i].section_path,
            )
            for i in top
        ]
