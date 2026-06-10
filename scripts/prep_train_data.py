"""Prepare instruction-tuning data from synthetic Q/A pairs.

Builds closed-book chat examples (question -> answer) for QLoRA. The study's
three arms are: base+RAG (no FT), FT closed-book (no retrieval), and FT+RAG.
Closed-book FT tests whether fine-tuning injects the corpus knowledge that
RAG otherwise supplies at query time.

Split discipline: the 700 synthetic pairs are split train/val (90/10) for
training monitoring only. The 100-question hand-authored golden set remains the
held-out evaluation — a disjoint question set over the *same* corpus. That
overlap is intentional and documented: closed-book FT is meant to memorize
in-domain facts, and the eval measures whether that memorization generalizes
to unseen question phrasings as well as retrieval does.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

SYSTEM = (
    "You are a precise assistant answering questions about electronic components. "
    "Answer concisely with exact values and units."
)


@app.command()
def main(
    src: Path = typer.Option(Path("data/golden/synthetic-train.jsonl"), "--src"),
    out_dir: Path = typer.Option(Path("data/train"), "--out"),
    val_frac: float = typer.Option(0.1, "--val-frac"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    pairs = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * val_frac)
    val, train = pairs[:n_val], pairs[n_val:]
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(split, name):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for p in split:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": p["question"]},
                    {"role": "assistant", "content": p["answer"]},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        return path

    tp, vp = write(train, "train.jsonl"), write(val, "val.jsonl")
    typer.echo(f"wrote {tp} ({len(train)}) and {vp} ({len(val)})")


if __name__ == "__main__":
    app()
