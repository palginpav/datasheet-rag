"""Retrieval ablation: dense vs BM25 vs hybrid over the golden set.

Usage:
    python scripts/ablation.py [--golden data/golden/golden.jsonl] [--k 8]
        [--out docs/ablation.md]

Retrieval-only — no generation, no judge — so the comparison is fast and
deterministic. Reports hit@k / MRR / recall overall and per category, the
lever being MCU/part-number queries where BM25's exact-token match is
expected to rescue the dense baseline's misses.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import typer

from datasheet_rag.eval import metrics
from datasheet_rag.eval.golden import load_golden

app = typer.Typer(add_completion=False)


def _doc_hit(retrieved_parts: list[str], gold_parts: list[str], k: int) -> float:
    """Document-level routing: did the top-k include any chunk of a gold part?"""
    gold = set(gold_parts)
    return 1.0 if any(p in gold for p in retrieved_parts[:k]) else 0.0


def _eval_retriever(retriever, questions, k):
    """Return (overall dict, per-category dict) of retrieval metrics."""
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        hits = retriever.retrieve(q.question, k=k)
        ids = [r.chunk_id for r in hits]
        parts = [r.part for r in hits]
        rows.append(
            (
                q.category,
                metrics.hit_at_k(ids, q.gold_chunk_ids, k),
                metrics.reciprocal_rank(ids, q.gold_chunk_ids),
                metrics.context_recall(ids, q.gold_chunk_ids, k),
                _doc_hit(parts, q.gold_parts, k),
            )
        )
    overall = {
        "hit": round(metrics.mean([r[1] for r in rows]), 3),
        "mrr": round(metrics.mean([r[2] for r in rows]), 3),
        "recall": round(metrics.mean([r[3] for r in rows]), 3),
        "doc_hit": round(metrics.mean([r[4] for r in rows]), 3),
    }
    by_cat: dict[str, list] = defaultdict(list)
    for cat, hit, mrr, _rec, _doc in rows:
        by_cat[cat].append((hit, mrr))
    per_cat = {
        cat: round(metrics.mean([h for h, _ in v]), 3) for cat, v in by_cat.items()
    }
    return overall, per_cat


@app.command()
def main(
    golden: Path = typer.Option(Path("data/golden/golden.jsonl"), "--golden"),
    k: int = typer.Option(8, "--k"),
    out: Path = typer.Option(Path("docs/ablation.md"), "--out"),
    parsed: Path = typer.Option(Path("corpus/parsed"), "--parsed"),
    store_dir: Path = typer.Option(Path("chroma"), "--store"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder arm"),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    from datasheet_rag.rag.bm25 import BM25Retriever, load_all_chunks
    from datasheet_rag.rag.embed import NomicEmbedder
    from datasheet_rag.rag.hybrid import HybridRetriever
    from datasheet_rag.rag.retrieve import DenseRetriever
    from datasheet_rag.rag.store import ChunkStore

    questions = load_golden(golden)
    typer.echo("building retrievers...")
    dense = DenseRetriever(ChunkStore(store_dir), NomicEmbedder())
    bm25 = BM25Retriever(load_all_chunks(parsed))
    hybrid = HybridRetriever([dense, bm25])
    hybrid_w = HybridRetriever([dense, bm25], weights=[3.0, 1.0])  # dense-weighted
    configs = {"dense": dense, "bm25": bm25, "hybrid": hybrid, "hybrid-w3": hybrid_w}
    if not no_rerank:
        from datasheet_rag.rag.rerank import RerankRetriever

        configs["dense+rerank"] = RerankRetriever(dense, pool=30, device=device)

    results = {}
    for name, r in configs.items():
        typer.echo(f"  evaluating {name}...")
        results[name] = _eval_retriever(r, questions, k)

    cats = sorted({q.category for q in questions if q.answerable})
    lines = [
        "# Retrieval ablation",
        "",
        f"Golden set: {sum(1 for q in questions if q.answerable)} answerable questions · "
        f"k={k} · retrieval-only (no generation).",
        "",
        "## Overall",
        "",
        "`doc-hit@k` = top-k contained any chunk of the correct part (document "
        "routing); `hit@k` = top-k contained the specific gold chunk (chunk ranking).",
        "",
        "| Retriever | hit@k | MRR | recall@k | doc-hit@k |",
        "|---|---|---|---|---|",
    ]
    for name in configs:
        o = results[name][0]
        lines.append(f"| {name} | {o['hit']} | {o['mrr']} | {o['recall']} | {o['doc_hit']} |")
    lines += ["", "## hit@k by category", "", "| Category | " + " | ".join(configs) + " |",
              "|---|" + "---|" * len(configs)]
    for cat in cats:
        cells = " | ".join(str(results[name][1].get(cat, "—")) for name in configs)
        lines.append(f"| {cat} | {cells} |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
