"""Interactive human grading of answers, for judge-vs-human agreement.

Usage:
    python scripts/grade.py [--trace runs/eval-trace.jsonl] [--out data/golden/human-grades.jsonl]

Presents each answerable question's gold answer and the system's answer, and
asks for a correctness grade (1-5) on the same scale the LLM judge uses.
Resumable: questions already graded in the output file are skipped. Enter
's' to skip, 'q' to stop and save.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    trace: Path = typer.Option(Path("runs/eval-trace.jsonl"), "--trace"),
    golden: Path = typer.Option(Path("data/golden/golden.jsonl"), "--golden"),
    out: Path = typer.Option(Path("data/golden/human-grades.jsonl"), "--out"),
) -> None:
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    gold = {
        json.loads(line)["id"]: json.loads(line)
        for line in golden.read_text().splitlines()
        if line.strip()
    }
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}

    todo = [r for r in rows if r.get("answerable") and r["id"] not in done]
    typer.echo(f"{len(todo)} answerable questions to grade ({len(done)} already done).\n")

    with out.open("a", encoding="utf-8") as f:
        for r in todo:
            g = gold.get(r["id"], {})
            typer.echo(f"--- {r['id']} [{r['category']}]")
            typer.echo(f"Q:    {g.get('question', '')}")
            typer.echo(f"GOLD: {g.get('gold_answer', '')}")
            typer.echo(f"ANS:  {r['answer']}")
            typer.echo(f"(judge correctness was {r.get('correctness')})")
            while True:
                choice = typer.prompt("correctness 1-5 (s=skip, q=quit)").strip().lower()
                if choice == "q":
                    typer.echo("saved, stopping.")
                    raise typer.Exit(0)
                if choice == "s":
                    break
                if choice in {"1", "2", "3", "4", "5"}:
                    f.write(json.dumps({"id": r["id"], "human_correctness": int(choice)}) + "\n")
                    f.flush()
                    break
                typer.echo("enter 1-5, s, or q")
            typer.echo("")
    typer.echo("done.")


if __name__ == "__main__":
    app()
