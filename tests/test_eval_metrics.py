"""Tests for retrieval/answer metrics and the golden schema (no model deps)."""

import pytest
from pydantic import ValidationError

from datasheet_rag.eval import metrics
from datasheet_rag.eval.golden import GoldenQuestion, load_golden, save_golden

# --- retrieval metrics -----------------------------------------------------


def test_hit_at_k_respects_cutoff():
    retrieved = ["a", "b", "c", "d"]
    assert metrics.hit_at_k(retrieved, ["c"], k=4) == 1.0
    assert metrics.hit_at_k(retrieved, ["c"], k=2) == 0.0
    assert metrics.hit_at_k(retrieved, ["z"], k=4) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert metrics.reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_context_recall_and_precision():
    retrieved = ["a", "b", "c", "d"]
    gold = ["a", "c", "x"]  # x not retrievable
    assert metrics.context_recall(retrieved, gold, k=4) == pytest.approx(2 / 3)
    assert metrics.context_precision(retrieved, gold, k=4) == pytest.approx(2 / 4)


def test_context_recall_empty_gold_is_one():
    assert metrics.context_recall(["a"], [], k=4) == 1.0


# --- must-include coverage (whitespace + micro-sign robust) ----------------


def test_must_include_survives_pdf_spacing_and_mu():
    # answer as a PDF extractor renders it; requirement as authored
    answer = "The offset is 25 μ V maximum [2]."  # greek mu, spaced
    assert metrics.must_include_coverage(answer, ["25", "µV"]) == 1.0  # micro sign


def test_must_include_partial():
    answer = "Dropout is 1.2 V."
    assert metrics.must_include_coverage(answer, ["1.2", "800"]) == 0.5


def test_must_include_empty_is_full():
    assert metrics.must_include_coverage("anything", []) == 1.0


# --- golden schema ---------------------------------------------------------


def _q(**kw):
    base = dict(
        id="g001", question="What is the offset?", category="spec-lookup",
        answerable=True, gold_answer="25 µV", must_include=["25"],
        gold_parts=["OPA188"], gold_chunk_ids=["abc123def456-0003"],
    )
    base.update(kw)
    return base


def test_answerable_requires_gold_answer_and_chunks():
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(_q(gold_answer=None))
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(_q(gold_chunk_ids=[]))


def test_unanswerable_must_not_have_gold_answer():
    with pytest.raises(ValidationError):
        GoldenQuestion.model_validate(
            _q(answerable=False, category="unanswerable", gold_chunk_ids=[])
        )
    # valid unanswerable
    q = GoldenQuestion.model_validate(
        _q(answerable=False, category="unanswerable", gold_answer=None,
           must_include=[], gold_chunk_ids=[])
    )
    assert not q.answerable


def test_golden_round_trip_and_dup_detection(tmp_path):
    qs = [
        GoldenQuestion.model_validate(_q(id="g001")),
        GoldenQuestion.model_validate(_q(id="g002")),
    ]
    p = tmp_path / "g.jsonl"
    save_golden(qs, p)
    assert load_golden(p) == qs

    dup = tmp_path / "dup.jsonl"
    save_golden([qs[0], qs[0]], dup)
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(dup)


def test_shipped_golden_set_is_valid():
    qs = load_golden("data/golden/golden.jsonl")
    assert len(qs) >= 20
    assert any(q.category == "unanswerable" for q in qs)
    assert any(q.category == "comparison" for q in qs)
