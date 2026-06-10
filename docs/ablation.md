# Retrieval ablation

Golden set: 91 answerable questions · k=8 · retrieval-only (no generation).

## Overall

`doc-hit@k` = top-k contained any chunk of the correct part (document routing); `hit@k` = top-k contained the specific gold chunk (chunk ranking).

| Retriever | hit@k | MRR | recall@k | doc-hit@k |
|---|---|---|---|---|
| dense | 0.923 | 0.657 | 0.896 | 1.0 |
| bm25 | 0.462 | 0.241 | 0.445 | 0.879 |
| hybrid | 0.857 | 0.515 | 0.841 | 1.0 |
| hybrid-w3 | 0.912 | 0.574 | 0.89 | 1.0 |
| dense+rerank | 0.967 | 0.676 | 0.945 | 1.0 |

## hit@k by category

| Category | dense | bm25 | hybrid | hybrid-w3 | dense+rerank |
|---|---|---|---|---|---|
| comparison | 1.0 | 0.4 | 0.8 | 1.0 | 0.9 |
| conditions | 0.929 | 0.429 | 0.786 | 0.857 | 1.0 |
| cross-section | 1.0 | 0.833 | 1.0 | 1.0 | 1.0 |
| mcu | 0.75 | 0.062 | 0.625 | 0.75 | 0.938 |
| pinout | 1.0 | 0.2 | 1.0 | 1.0 | 1.0 |
| spec-lookup | 0.95 | 0.625 | 0.95 | 0.95 | 0.975 |

## Analysis

The naive hypothesis — "add BM25 to fix exact part-number queries" — **did not hold**,
and the reason is instructive:

- **Dense already routes perfectly** (doc-hit@k = 1.0): it never misses the right
  *document*. The MCU weakness was never a routing problem.
- **BM25 is weak here (hit@k 0.46) and even routes worse (0.88)** because the chunker
  prepends `[PART] section` to *every* chunk — so the part number appears in all ~300
  chunks of a document and gives BM25 no within-document discrimination, while cross-document
  token overlap (e.g. shared words with INA282, RISC-V) distracts it.
- **Fusing a weak retriever hurts**: naive RRF drags dense down (0.92 → 0.86). Weighting
  dense 3:1 nearly recovers (0.91) but never beats dense alone.

The residual failure is **within-document chunk ranking**: dense lands the right part but
the answer-bearing summary chunk competes with hundreds of same-part siblings. A
cross-encoder reads query and chunk *together* and fixes exactly this:

- **dense+rerank is the winner**: hit@k 0.92 → **0.97**, recall 0.90 → 0.94, and the target
  weak category **MCU 0.75 → 0.94**. conditions and spec-lookup also improve.
- One small regression: comparison dips 1.0 → 0.9 (a two-part question where reranking the
  pool dropped one gold chunk) — an honest tradeoff, not hidden.

**Decision:** dense remains the first stage; **dense+rerank** is the recommended retriever
(opt-in via `--retriever dense+rerank`). BM25/hybrid are not adopted on this corpus. A
natural follow-up the analysis points to — drop the per-chunk part prefix and re-test BM25
— is left for the chunking ablation.
