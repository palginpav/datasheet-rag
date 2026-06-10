"""Judge-vs-human agreement, and a mechanical sanity cross-check.

The LLM judge is only trustworthy if it tracks human judgement. This module
computes standard agreement statistics between the judge's correctness
scores (from an eval trace) and human grades (collected via scripts/grade.py):

- exact agreement, within-1 agreement (1-5 scales rarely need exactness)
- mean absolute error
- quadratic-weighted Cohen's kappa (the usual ordinal-rating agreement
  coefficient; rewards near-misses, penalizes far-misses)

Until human grades exist, ``mechanical_harshness`` gives a no-human proxy:
answers the judge scored below 4 that nonetheless contain every required
value (must_include coverage == 1.0) are candidates for judge over-strictness
— exactly the terse-but-correct pattern seen in the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Agreement:
    n: int
    exact: float
    within_one: float
    mean_abs_error: float
    quadratic_kappa: float | None


def _quadratic_weighted_kappa(a: list[int], b: list[int], lo: int = 1, hi: int = 5) -> float | None:
    """Cohen's kappa with quadratic weights over the integer rating scale."""
    n = len(a)
    if n == 0:
        return None
    cats = list(range(lo, hi + 1))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[idx[x]][idx[y]] += 1.0

    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    weights = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    expected = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]

    num = sum(weights[i][j] * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weights[i][j] * expected[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return 1.0  # perfect (degenerate: no disagreement possible)
    return 1.0 - num / den


def compute_agreement(judge: list[int], human: list[int]) -> Agreement:
    """Both lists are aligned 1-5 ratings of the same answers."""
    n = len(judge)
    if n == 0:
        return Agreement(0, 0.0, 0.0, 0.0, None)
    exact = sum(1 for j, h in zip(judge, human, strict=True) if j == h) / n
    within = sum(1 for j, h in zip(judge, human, strict=True) if abs(j - h) <= 1) / n
    mae = sum(abs(j - h) for j, h in zip(judge, human, strict=True)) / n
    return Agreement(
        n=n,
        exact=round(exact, 3),
        within_one=round(within, 3),
        mean_abs_error=round(mae, 3),
        quadratic_kappa=(
            round(k, 3) if (k := _quadratic_weighted_kappa(judge, human)) is not None else None
        ),
    )


def mechanical_harshness(trace_rows: list[dict]) -> list[dict]:
    """Answerable rows where the judge gave correctness < 4 despite the answer
    containing every required value — candidate judge over-strictness."""
    out = []
    for r in trace_rows:
        if not r.get("answerable"):
            continue
        corr = r.get("correctness")
        cover = r.get("must_include")
        if corr is not None and corr < 4 and cover == 1.0:
            out.append({"id": r["id"], "correctness": corr, "answer": r.get("answer", "")})
    return out
