"""Generate a synthetic Q/A set from corpus chunks with the local model.

Usage:
    python scripts/synthesize.py [--n 80] [--out data/golden/synthetic.jsonl]
        [--seed 0] [--table-bias 0.6] [--model qwen3:4b]

Samples chunks (biased toward table chunks, where specs live), generates one
grounded Q/A pair each, and writes them with source="synthetic",
needs_review=true. These feed Phase 5; spot-check before relying on them.
"""

from __future__ import annotations

import random
from pathlib import Path

import typer

from datasheet_rag.eval.synthesize import generate_pair, save_pairs
from datasheet_rag.ingest.models import load_chunks

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = typer.Option(80, "--n"),
    parsed: Path = typer.Option(Path("corpus/parsed"), "--parsed"),
    out: Path = typer.Option(Path("data/golden/synthetic.jsonl"), "--out"),
    seed: int = typer.Option(0, "--seed"),
    table_bias: float = typer.Option(0.6, "--table-bias", help="Fraction of samples from tables"),
    model: str = typer.Option("qwen3:4b", "--model"),
) -> None:
    from datasheet_rag.rag.generate import OllamaClient

    chunks = [c for f in sorted(parsed.glob("*.chunks.jsonl")) for c in load_chunks(f)]
    tables = [c for c in chunks if c.kind == "table"]
    texts = [c for c in chunks if c.kind == "text"]
    rng = random.Random(seed)
    n_tab = min(int(n * table_bias), len(tables))
    sample = rng.sample(tables, n_tab) + rng.sample(texts, min(n - n_tab, len(texts)))
    rng.shuffle(sample)

    llm = OllamaClient(model=model)
    pairs, failed = [], 0
    for i, chunk in enumerate(sample, start=1):
        pair = generate_pair(chunk, llm)
        if pair is None:
            failed += 1
        else:
            pairs.append(pair)
        if i % 10 == 0:
            typer.echo(f"  {i}/{len(sample)} ({len(pairs)} ok, {failed} unparseable)")

    save_pairs(pairs, out)
    typer.echo(f"wrote {out}: {len(pairs)} pairs ({failed} unparseable, from {len(sample)} chunks)")


if __name__ == "__main__":
    app()
