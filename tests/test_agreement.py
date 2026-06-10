"""Tests for judge-vs-human agreement statistics."""

import pytest

from datasheet_rag.eval.agreement import compute_agreement, mechanical_harshness


def test_perfect_agreement():
    a = compute_agreement([5, 4, 3, 2], [5, 4, 3, 2])
    assert a.exact == 1.0
    assert a.within_one == 1.0
    assert a.mean_abs_error == 0.0
    assert a.quadratic_kappa == 1.0


def test_within_one_and_mae():
    a = compute_agreement([5, 4, 3], [4, 3, 3])  # off by 1, 1, 0
    assert a.exact == pytest.approx(1 / 3, abs=1e-3)
    assert a.within_one == 1.0
    assert a.mean_abs_error == pytest.approx(2 / 3, abs=1e-3)


def test_systematic_harshness_still_positive_kappa():
    # judge consistently one below human, but tracking the ordering
    judge = [1, 2, 3, 4, 4]
    human = [2, 3, 4, 5, 5]
    a = compute_agreement(judge, human)
    assert a.exact == 0.0
    assert a.within_one == 1.0
    assert a.quadratic_kappa is not None


def test_empty_is_safe():
    a = compute_agreement([], [])
    assert a.n == 0
    assert a.quadratic_kappa is None


def test_mechanical_harshness_flags_terse_correct():
    rows = [
        {"id": "g001", "answerable": True, "correctness": 3, "must_include": 1.0, "answer": "ok"},
        {"id": "g002", "answerable": True, "correctness": 5, "must_include": 1.0, "answer": "ok"},
        {"id": "g018", "answerable": True, "correctness": 1, "must_include": 0.5, "answer": "no"},
        {"id": "g022", "answerable": False, "correctness": None, "must_include": None},
    ]
    flagged = mechanical_harshness(rows)
    assert [f["id"] for f in flagged] == ["g001"]  # only the low-score-but-complete one
