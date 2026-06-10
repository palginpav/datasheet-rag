"""Drive the golden set through the pipeline and emit a scorecard.

For each golden question: retrieve, generate, then score retrieval
(hit@k, MRR, context recall/precision against gold_chunk_ids) and answer
quality (must_include coverage, plus LLM-judged faithfulness/correctness
when --judge is on). Unanswerable questions are scored on whether the system
correctly refused.

Writes a per-question JSONL trace and a markdown scorecard. The scorecard is
committed so retrieval/answer regressions show up in diffs across phases.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from datasheet_rag.eval import metrics
from datasheet_rag.eval.golden import GoldenQuestion, load_golden


@dataclass
class QuestionResult:
    id: str
    category: str
    answerable: bool
    refused: bool
    refusal_correct: bool | None  # only meaningful for unanswerable questions
    hit_at_k: float | None
    mrr: float | None
    context_recall: float | None
    context_precision: float | None
    must_include: float | None
    faithfulness: int | None
    correctness: int | None
    retrieved_ids: list[str]
    answer: str


def evaluate_question(
    q: GoldenQuestion, retriever, llm, judge, k: int
) -> QuestionResult:
    from datasheet_rag.rag.pipeline import ask

    res = ask(q.question, retriever, llm, k=k)
    retrieved_ids = [r.chunk_id for r in res.retrieved]

    if not q.answerable:
        return QuestionResult(
            id=q.id, category=q.category, answerable=False, refused=res.refused,
            refusal_correct=res.refused, hit_at_k=None, mrr=None, context_recall=None,
            context_precision=None, must_include=None, faithfulness=None, correctness=None,
            retrieved_ids=retrieved_ids, answer=res.answer,
        )

    hit = metrics.hit_at_k(retrieved_ids, q.gold_chunk_ids, k)
    mrr = metrics.reciprocal_rank(retrieved_ids, q.gold_chunk_ids)
    rec = metrics.context_recall(retrieved_ids, q.gold_chunk_ids, k)
    prec = metrics.context_precision(retrieved_ids, q.gold_chunk_ids, k)
    cover = metrics.must_include_coverage(res.answer, q.must_include)

    faith = corr = None
    if judge is not None:
        from datasheet_rag.rag.generate import build_context_block

        ctx = build_context_block(res.retrieved)
        faith = judge.faithfulness(res.answer, ctx).score
        corr = judge.correctness(q.question, res.answer, q.gold_answer or "").score

    return QuestionResult(
        id=q.id, category=q.category, answerable=True, refused=res.refused,
        refusal_correct=None, hit_at_k=hit, mrr=mrr, context_recall=rec,
        context_precision=prec, must_include=cover, faithfulness=faith,
        correctness=corr, retrieved_ids=retrieved_ids, answer=res.answer,
    )


def _avg(vals: list[float | int | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return round(metrics.mean([float(v) for v in nums]), 3) if nums else None


def build_scorecard(
    results: list[QuestionResult], k: int, judged: bool, retriever: str = "dense"
) -> str:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    lines = [
        "# Evaluation scorecard",
        "",
        f"Golden set: {len(results)} questions "
        f"({len(answerable)} answerable, {len(unanswerable)} unanswerable) · "
        f"k={k} · {retriever} retrieval · "
        f"judge={'gpt-oss (faithfulness/correctness 1-5)' if judged else 'off'}",
        "",
        "## Retrieval & answer (answerable questions)",
        "",
        "| Metric | Mean |",
        "|---|---|",
        f"| hit@{k} | {_avg([r.hit_at_k for r in answerable])} |",
        f"| MRR | {_avg([r.mrr for r in answerable])} |",
        f"| context recall@{k} | {_avg([r.context_recall for r in answerable])} |",
        f"| context precision@{k} | {_avg([r.context_precision for r in answerable])} |",
        f"| must-include coverage | {_avg([r.must_include for r in answerable])} |",
    ]
    if judged:
        lines += [
            f"| faithfulness (1-5) | {_avg([r.faithfulness for r in answerable])} |",
            f"| correctness (1-5) | {_avg([r.correctness for r in answerable])} |",
        ]
    correct_refusals = sum(1 for r in unanswerable if r.refusal_correct)
    lines += [
        "",
        "## Refusal calibration (unanswerable questions)",
        "",
        f"- correctly refused: {correct_refusals}/{len(unanswerable)}",
        "",
        "## Per-category hit@k",
        "",
        "| Category | n | hit@k | MRR |",
        "|---|---|---|---|",
    ]
    cats = sorted({r.category for r in answerable})
    for cat in cats:
        rows = [r for r in answerable if r.category == cat]
        lines.append(
            f"| {cat} | {len(rows)} | {_avg([r.hit_at_k for r in rows])} "
            f"| {_avg([r.mrr for r in rows])} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(
    golden_path: Path,
    k: int,
    judged: bool,
    store_dir: Path,
    model: str,
    judge_model: str,
    retriever_name: str = "dense",
    device: str | None = None,
) -> tuple[list[QuestionResult], str]:
    from datasheet_rag.eval.judge import OllamaJudge
    from datasheet_rag.rag.factory import build_retriever
    from datasheet_rag.rag.generate import OllamaClient

    questions = load_golden(golden_path)
    retriever = build_retriever(retriever_name, store_dir=store_dir, device=device)
    llm = OllamaClient(model=model)
    judge = OllamaJudge(model=judge_model) if judged else None

    results = [evaluate_question(q, retriever, llm, judge, k) for q in questions]
    return results, build_scorecard(results, k, judged, retriever=retriever_name)


def write_trace(results: list[QuestionResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
