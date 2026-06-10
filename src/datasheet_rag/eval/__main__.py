"""Evaluation CLI.

Usage:
    python -m datasheet_rag.eval [--golden data/golden/golden.jsonl] [--k 8]
                                 [--judge/--no-judge] [--out docs/eval-scorecard.md]
"""

from __future__ import annotations

from pathlib import Path

import typer

from datasheet_rag.eval.run import run_eval, write_trace

app = typer.Typer(add_completion=False)


@app.command()
def main(
    golden: Path = typer.Option(Path("data/golden/golden.jsonl"), "--golden"),
    k: int = typer.Option(8, "--k"),
    judge: bool = typer.Option(False, "--judge/--no-judge", help="Run the LLM judge (slow)"),
    out: Path = typer.Option(Path("docs/eval-scorecard.md"), "--out"),
    trace: Path = typer.Option(Path("runs/eval-trace.jsonl"), "--trace"),
    store_dir: Path = typer.Option(Path("chroma"), "--store"),
    model: str = typer.Option("qwen3:4b", "--model"),
    judge_model: str = typer.Option("gpt-oss:latest", "--judge-model"),
    retriever: str = typer.Option("dense", "--retriever", help="dense|bm25|hybrid|dense+rerank"),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    results, scorecard = run_eval(
        golden, k, judge, store_dir, model, judge_model, retriever_name=retriever, device=device
    )
    out.write_text(scorecard + "\n", encoding="utf-8")
    trace.parent.mkdir(parents=True, exist_ok=True)
    write_trace(results, trace)
    refused = sum(1 for r in results if not r.answerable and r.refusal_correct)
    unans = sum(1 for r in results if not r.answerable)
    typer.echo(f"wrote {out} and {trace}")
    typer.echo(f"  {len(results)} questions · refusals {refused}/{unans} correct")


if __name__ == "__main__":
    app()
