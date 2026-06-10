# Evaluation scorecard

Golden set: 100 questions (91 answerable, 9 unanswerable) · k=8 · dense retrieval · judge=gpt-oss (faithfulness/correctness 1-5)

## Retrieval & answer (answerable questions)

| Metric | Mean |
|---|---|
| hit@8 | 0.923 |
| MRR | 0.657 |
| context recall@8 | 0.896 |
| context precision@8 | 0.122 |
| must-include coverage | 0.87 |
| faithfulness (1-5) | 4.912 |
| correctness (1-5) | 4.275 |

## Refusal calibration (unanswerable questions)

- correctly refused: 9/9

## Per-category hit@k

| Category | n | hit@k | MRR |
|---|---|---|---|
| comparison | 10 | 1.0 | 0.717 |
| conditions | 14 | 0.929 | 0.717 |
| cross-section | 6 | 1.0 | 0.833 |
| mcu | 16 | 0.75 | 0.451 |
| pinout | 5 | 1.0 | 0.7 |
| spec-lookup | 40 | 0.95 | 0.671 |

## Reading the numbers

- **faithfulness 4.91 / correctness 4.28** across 91 answerable questions: no
  meaningful hallucination; the correctness gap is mostly the judge under-scoring
  terse-but-correct answers (see docs/judge-agreement.md).
- **context precision@8 = 0.12 is expected** (1–2 gold chunks at k=8 → ceiling ~0.12–0.25);
  **recall@8 (0.90)** is the meaningful retrieval number.

## Findings → Phase 4 motivation

1. **MCU is the weak category (hit@8 0.75, MRR 0.45) — now robust across 16 questions.**
   Exact part-number queries (e.g. flash/SRAM of a specific STM32) see the device-summary
   chunk crowded below k=8 by sibling STM32 datasheets. This is the canonical case
   **hybrid BM25 + RRF (Phase 4)** targets — a lexical hit on the part number surfaces the
   summary regardless of dense crowding. Every other category is ≥ 0.93 hit@8.
2. **Refusal calibration perfect (9/9)** including adversarial traps (op-amp Bluetooth,
   EEPROM CPU cores, regulator firmware) — refused, not hallucinated.

This 100-question scorecard is the baseline Phase 4 must beat, especially on MCU hit@k.
