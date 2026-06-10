"""Freeze dense+rerank retrieval contexts for the golden set.

The fine-tuning study compares base vs FT under identical retrieval, so we
retrieve once (with the Phase-4 winner, dense+rerank) and save the contexts.
Both models then answer from the same frozen contexts — the only variable is
the model, making the RAG arms strictly comparable. Runs in the rag venv.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from datasheet_rag.eval.golden import load_golden

app = typer.Typer(add_completion=False)


@app.command()
def main(
    golden: Path = typer.Option(Path("data/golden/golden.jsonl"), "--golden"),
    out: Path = typer.Option(Path("runs/frozen-contexts.jsonl"), "--out"),
    k: int = typer.Option(8, "--k"),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    from datasheet_rag.rag.factory import build_retriever

    retriever = build_retriever("dense+rerank", device=device)
    questions = load_golden(golden)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in questions:
            hits = retriever.retrieve(q.question, k=k)
            f.write(
                json.dumps(
                    {
                        "id": q.id,
                        "question": q.question,
                        "category": q.category,
                        "answerable": q.answerable,
                        "gold_answer": q.gold_answer,
                        "must_include": q.must_include,
                        "contexts": [
                            {"part": h.part, "page": h.page, "text": h.text} for h in hits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    typer.echo(f"wrote {out}: {len(questions)} questions with frozen contexts")


if __name__ == "__main__":
    app()
