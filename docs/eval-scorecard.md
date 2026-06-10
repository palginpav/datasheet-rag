# Evaluation scorecard

Golden set: 25 questions (21 answerable, 4 unanswerable) · k=8 · dense retrieval · judge=gpt-oss (faithfulness/correctness 1-5)

## Retrieval & answer (answerable questions)

| Metric | Mean |
|---|---|
| hit@8 | 0.905 |
| MRR | 0.68 |
| context recall@8 | 0.881 |
| context precision@8 | 0.119 |
| must-include coverage | 0.905 |
| faithfulness (1-5) | 5.0 |
| correctness (1-5) | 4.048 |

## Refusal calibration (unanswerable questions)

- correctly refused: 4/4

## Per-category hit@k

| Category | n | hit@k | MRR |
|---|---|---|---|
| comparison | 2 | 1.0 | 1.0 |
| conditions | 3 | 1.0 | 0.833 |
| cross-section | 1 | 1.0 | 1.0 |
| mcu | 4 | 0.5 | 0.333 |
| spec-lookup | 11 | 1.0 | 0.677 |


## Reading the numbers

- **faithfulness 5.0 / correctness 4.05**: no hallucination anywhere (every claim
  grounded), and answers are usually right. The gap is the judge penalizing
  *terse-but-correct* replies — e.g. g001 answers "±250µV [2]" (correct) yet scores
  correctness 3 for lacking the "typical" qualifier. The Phase-3 judge-vs-human
  subset will quantify this stinginess.
- **context precision@8 = 0.12 is expected, not a defect**: with 1–2 gold chunks and
  k=8, the ceiling is ~0.12–0.25. **Recall@8 (0.88)** is the meaningful retrieval
  number here.

## Findings → Phase 4 motivation

1. **Corpus growth degraded MCU retrieval (the headline finding).** g013
   ("STM32C031C4 flash") *regressed*: on the 40-part corpus the smoke run answered
   "32 KB" correctly; at 90 parts with many STM32 siblings, the device-summary chunk
   now ranks below k=8 and the model *refused*. Exact part-number queries are the
   classic case **hybrid BM25 retrieval (Phase 4) is designed to fix** — a lexical
   match on "STM32C031C4" would surface the summary chunk regardless of dense
   crowding.
2. **One wrong value (g018): LD1117 max current answered "1300 mA" vs gold 800 mA.**
   Retrieval pulled a chunk citing a different current figure; worth tracing whether
   it's a peak/limit value mis-bound to the question.
3. **Golden-annotation gaps to fix in review** (the human pass): g014 answered the
   core correctly ("Cortex-M0+") but scored a retrieval miss because gold_chunk_ids
   listed only the title block, not the description chunk that also states it;
   g020's gold answer is ambiguously worded. These are golden-set quality items, not
   system failures.

Refusal calibration is perfect (4/4) — including the adversarial "WiFi transmit power
of an op-amp" trap, refused rather than hallucinated.
