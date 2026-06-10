"""Closed-book evaluation: ask the model with no retrieved context.

This is the fine-tuning study's middle arm — does QLoRA inject enough corpus
knowledge that the model answers datasheet questions *without* retrieval?
Only answer-side metrics apply (must-include coverage, judged correctness,
and refusal calibration); there is no retrieval to score.
"""

from __future__ import annotations

from datasheet_rag.eval import metrics
from datasheet_rag.eval.golden import GoldenQuestion
from datasheet_rag.eval.run import QuestionResult

CLOSED_BOOK_SYSTEM = (
    "You are an expert on electronic components. Answer the question with exact "
    "values and units from memory. If you do not know, reply starting with the exact "
    "words \"NOT IN CONTEXT\"."
)


def evaluate_closed_book(q: GoldenQuestion, llm, judge) -> QuestionResult:
    from datasheet_rag.rag.generate import REFUSAL_TOKEN, parse_answer

    raw = llm.chat(
        [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": q.question},
        ]
    )
    gen = parse_answer(raw, n_chunks=0)
    refused = gen.answer.upper().startswith(REFUSAL_TOKEN)

    if not q.answerable:
        return QuestionResult(
            id=q.id, category=q.category, answerable=False, refused=refused,
            refusal_correct=refused, hit_at_k=None, mrr=None, context_recall=None,
            context_precision=None, must_include=None, faithfulness=None, correctness=None,
            retrieved_ids=[], answer=gen.answer,
        )

    cover = metrics.must_include_coverage(gen.answer, q.must_include)
    corr = None
    if judge is not None:
        corr = judge.correctness(q.question, gen.answer, q.gold_answer or "").score
    return QuestionResult(
        id=q.id, category=q.category, answerable=True, refused=refused,
        refusal_correct=None, hit_at_k=None, mrr=None, context_recall=None,
        context_precision=None, must_include=cover, faithfulness=None,
        correctness=corr, retrieved_ids=[], answer=gen.answer,
    )
