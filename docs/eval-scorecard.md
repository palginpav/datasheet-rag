# Evaluation scorecard

Golden set: 34 questions (30 answerable, 4 unanswerable) · k=8 · dense retrieval · judge=gpt-oss (faithfulness/correctness 1-5)

## Retrieval & answer (answerable questions)

| Metric | Mean |
|---|---|
| hit@8 | 0.9 |
| MRR | 0.621 |
| context recall@8 | 0.85 |
| context precision@8 | 0.117 |
| must-include coverage | 0.878 |
| faithfulness (1-5) | 4.867 |
| correctness (1-5) | 4.133 |

## Refusal calibration (unanswerable questions)

- correctly refused: 4/4

## Per-category hit@k

| Category | n | hit@k | MRR |
|---|---|---|---|
| comparison | 4 | 1.0 | 0.625 |
| conditions | 6 | 1.0 | 0.722 |
| cross-section | 1 | 1.0 | 1.0 |
| mcu | 6 | 0.5 | 0.306 |
| pinout | 1 | 1.0 | 0.5 |
| spec-lookup | 12 | 1.0 | 0.704 |


## Reading the numbers

- **faithfulness 4.87 / correctness 4.13**: essentially no hallucination, answers usually
  right. The correctness gap is largely the judge penalizing *terse-but-correct* replies
  (e.g. "±250µV [2]" scored 3 for omitting "typical"). See docs/judge-agreement.md.
- **context precision@8 = 0.12 is expected, not a defect**: with 1–2 gold chunks at k=8 the
  ceiling is ~0.12–0.25. **Recall@8 (0.85)** is the meaningful retrieval number.

## Findings → Phase 4 motivation

1. **MCU retrieval is the weak category (hit@8 0.5).** Exact part-number queries
   (e.g. "STM32C031C4 flash") see the device-summary chunk crowded below k=8 by sibling
   STM32 datasheets — the canonical case **hybrid BM25 retrieval (Phase 4)** targets: a
   lexical match on the part number surfaces the summary regardless of dense crowding.
2. **One wrong value (LD1117 max current)** — retrieval bound a different current figure;
   worth tracing in Phase 4.
3. **Golden-annotation gaps for the human pass**: a few answers are correct but score a
   retrieval miss because gold_chunk_ids list only one of several evidence chunks.

Refusal calibration is perfect (4/4), including the adversarial "WiFi transmit power of an
op-amp" trap — refused, not hallucinated.
