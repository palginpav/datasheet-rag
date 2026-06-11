% datasheet-rag — Case Study
% Pavel Palgin
% 2026

## Problem

Engineering answers live in datasheets — dense, table-bound, and conditions-qualified
("I_Q = 25 µA *at 3.3 V, −40 °C*"), where part numbers punish fuzzy matching (an LM358 is not an
LM359). Naive retrieval-augmented generation (RAG) fails on exactly these properties. I built a
RAG system over 90 semiconductor datasheets that treats those properties as the design brief, and
— just as importantly — **measured every design choice instead of assuming it**.

## Approach

**Pipeline.** Table-aware PDF parsing (Docling) keeps tables as first-class chunks; dense retrieval
(nomic-embed + Chroma) is reranked by a cross-encoder; a 4B model generates grounded answers with
inline citations and an explicit refusal path. Everything runs on open weights with a local judge —
no closed-API dependency.

**Evaluation first.** Before optimizing anything, I built a 100-question golden set with every
value and evidence chunk verified against the index, scored by a *separate* judge model
(to avoid self-preference bias), plus a synthetic generator and judge-vs-human agreement tooling.
Measurement, then optimization.

## Results — where the project earns its keep

The interesting findings are the ones that contradicted the obvious move.

**1. Hybrid retrieval didn't help; reranking did.** The textbook fix for part-number queries is to
add BM25. I measured it: the BM25+RRF hybrid *underperformed* dense alone. Root cause — dense
already routed to the correct document 100 % of the time, and my own per-chunk part-number prefix
left BM25 no within-document discrimination. The real residual failure was chunk ranking *within*
the right document, which a cross-encoder reranker fixed:

| Retriever | hit@8 | recall@8 | MCU hit@8 |
|---|---|---|---|
| dense | 0.92 | 0.90 | 0.75 |
| naive BM25+RRF hybrid | 0.86 | 0.84 | 0.62 |
| **dense + cross-encoder rerank** | **0.97** | **0.95** | **0.94** |

**2. Fine-tuning didn't help either; it hurt.** I ran a QLoRA study (Qwen3-4B, 4 arms over frozen
contexts so the model was the only variable). Retrieval dominated: closed-book correctness ~2/5 vs
RAG ~4.6/5. Worse, fine-tuning on synthetic Q/A *collapsed refusal calibration* — the model refused
only 3/9 unanswerable questions vs the base model's 8/9 — because the training set contained no
"I don't know" examples and taught the model to always answer. The evidence said: don't ship the
fine-tune. So I didn't.

**Final system:** dense + rerank — hit@8 0.97, correctness 4.39/5, refusal calibration 9/9.

## What it demonstrates

Applied LLM/RAG engineering end-to-end — document ingestion, retrieval, evaluation methodology,
controlled ablation, parameter-efficient fine-tuning — on bleeding-edge infrastructure
(CUDA 13 / Blackwell), with a discipline that matters more than any single technique: **measure,
let the evidence decide, and report what doesn't work.** Twice the obvious approach failed and I
shipped the configuration the data supported instead. That judgment — including declining to ship a
fine-tuned model because it was measurably worse — is the point.

*Source, full ablations, and reproducible scripts: github.com/palginpav/datasheet-rag*
