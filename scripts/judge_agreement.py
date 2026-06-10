"""Judge-vs-human agreement report.

Usage:
    python scripts/judge_agreement.py [--trace runs/eval-trace.jsonl]
        [--human data/golden/human-grades.jsonl] [--out docs/judge-agreement.md]

If human grades exist, computes agreement statistics (exact, within-1, MAE,
quadratic-weighted kappa) between the LLM judge and the human. Otherwise
emits the mechanical over-strictness cross-check and instructions to run
scripts/grade.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from datasheet_rag.eval.agreement import compute_agreement, mechanical_harshness

app = typer.Typer(add_completion=False)


@app.command()
def main(
    trace: Path = typer.Option(Path("runs/eval-trace.jsonl"), "--trace"),
    human: Path = typer.Option(Path("data/golden/human-grades.jsonl"), "--human"),
    out: Path = typer.Option(Path("docs/judge-agreement.md"), "--out"),
) -> None:
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    judge_by_id = {r["id"]: r.get("correctness") for r in rows}

    lines = ["# Judge-vs-human agreement", ""]

    human_grades = {}
    if human.exists():
        human_grades = {
            json.loads(line)["id"]: json.loads(line)["human_correctness"]
            for line in human.read_text().splitlines()
            if line.strip()
        }

    paired = [
        (judge_by_id[i], human_grades[i])
        for i in human_grades
        if judge_by_id.get(i) is not None
    ]
    if paired:
        a = compute_agreement([j for j, _ in paired], [h for _, h in paired])
        lines += [
            f"Paired on {a.n} human-graded answers (correctness, 1-5 scale).",
            "",
            "| Statistic | Value |",
            "|---|---|",
            f"| exact agreement | {a.exact} |",
            f"| within-1 agreement | {a.within_one} |",
            f"| mean absolute error | {a.mean_abs_error} |",
            f"| quadratic-weighted kappa | {a.quadratic_kappa} |",
            "",
        ]
    else:
        lines += [
            "_No human grades yet._ Run `python scripts/grade.py` to grade the",
            "answers in the eval trace, then re-run this script to populate the table.",
            "",
        ]

    harsh = mechanical_harshness(rows)
    lines += [
        "## Mechanical over-strictness cross-check",
        "",
        "Answers the judge scored correctness < 4 that nonetheless contain every "
        "required value (must-include = 1.0) — candidate judge over-strictness, "
        "pending human confirmation:",
        "",
    ]
    if harsh:
        lines += ["| id | judge | answer |", "|---|---|---|"]
        lines += [f"| {h['id']} | {h['correctness']} | {h['answer'][:80]} |" for h in harsh]
    else:
        lines.append("_None._")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"wrote {out} ({len(paired)} paired, {len(harsh)} harshness candidates)")


if __name__ == "__main__":
    app()
