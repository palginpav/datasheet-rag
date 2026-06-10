"""Retrieval and answer metrics — pure functions, no model dependencies.

Retrieval metrics treat ``gold_chunk_ids`` as the relevant set and the
retriever's ordered output as the ranking:

- **hit@k**: any gold chunk in the top-k
- **MRR**: reciprocal rank of the first gold chunk (0 if none retrieved)
- **context recall**: fraction of gold chunks present in the top-k
- **context precision**: fraction of top-k that are gold (rank-agnostic)

Answer-side helpers here are deterministic string checks (``must_include``
coverage, refusal detection); semantic faithfulness/correctness live in the
LLM judge.
"""

from __future__ import annotations

import re


def _normalize(s: str) -> str:
    """Lowercase, unify the micro sign (µ U+00B5 vs μ U+03BC, which differ
    across PDF extractors), and remove ALL whitespace, so a required value
    like '25µV' matches a PDF-extracted answer rendered as '25 μ V'."""
    s = s.replace("μ", "µ").replace("Ω", "ω")
    return re.sub(r"\s+", "", s).lower()


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold = set(gold_ids)
    return 1.0 if any(cid in gold for cid in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def context_recall(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    gold = set(gold_ids)
    found = sum(1 for cid in set(retrieved_ids[:k]) if cid in gold)
    return found / len(gold)


def context_precision(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    gold = set(gold_ids)
    return sum(1 for cid in top if cid in gold) / len(top)


def must_include_coverage(answer: str, must_include: list[str]) -> float:
    """Fraction of required substrings present in the answer (normalized)."""
    if not must_include:
        return 1.0
    a = _normalize(answer)
    hits = sum(1 for s in must_include if _normalize(s) in a)
    return hits / len(must_include)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
