# Evaluation scorecard — dense+rerank

Golden set: 100 questions (91 answerable, 9 unanswerable) · k=8 · dense+rerank retrieval · judge=gpt-oss (faithfulness/correctness 1-5)

End-to-end run with the Phase-4 cross-encoder reranker. Versus the dense baseline
(docs/eval-scorecard.md): hit@8 0.92 → **0.97**, recall@8 0.90 → **0.95**, correctness
4.28 → **4.39**, MCU hit@8 0.75 → **0.94**. Questions the dense baseline refused
(e.g. g013 "STM32C031C4 flash") now answer correctly ("32 Kbytes").

## Retrieval & answer (answerable questions)

| Metric | Mean |
|---|---|
| hit@8 | 0.967 |
| MRR | 0.676 |
| context recall@8 | 0.945 |
| context precision@8 | 0.128 |
| must-include coverage | 0.866 |
| faithfulness (1-5) | 4.802 |
| correctness (1-5) | 4.385 |

## Refusal calibration (unanswerable questions)

- correctly refused: 9/9

## Per-category hit@k

| Category | n | hit@k | MRR |
|---|---|---|---|
| comparison | 10 | 0.9 | 0.65 |
| conditions | 14 | 1.0 | 0.646 |
| cross-section | 6 | 1.0 | 0.722 |
| mcu | 16 | 0.938 | 0.531 |
| pinout | 5 | 1.0 | 0.9 |
| spec-lookup | 40 | 0.975 | 0.715 |

