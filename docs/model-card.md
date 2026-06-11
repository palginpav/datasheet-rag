# Model card — datasheet-rag

**What it is.** A retrieval-augmented QA *system* (not a single model) for semiconductor
datasheets. Pipeline: Docling table-aware parsing → nomic-embed-text-v1.5 dense retrieval over
Chroma → BAAI/bge-reranker-base cross-encoder rerank → Qwen3-4B generation (local Ollama) with
inline `[n]` citations and an explicit `NOT IN CONTEXT` refusal path.

**Intended use.** Answering factual questions about electronic components from their datasheets —
spec lookups, conditions-qualified values, pinouts, cross-part comparisons — with traceable
citations. Built as an engineering portfolio piece; not a certified reference for design sign-off.

**Out of scope / limitations.**
- Answers are only as good as the retrieved excerpts; always verify against the source datasheet
  for design decisions.
- The corpus is a fixed snapshot (99 parts, mostly TI/ST + signal-chain components). Parts outside
  it are unanswerable by design (the system refuses rather than guesses).
- Generation runs a 4B model; complex multi-step reasoning over several tables can still err.

**Evaluation.** 100-question hand-authored golden set, every value and evidence chunk verified
against the index; correctness judged 1–5 by a separate model (gpt-oss ≠ the qwen3 generator, to
avoid self-preference bias). Headline (dense+rerank): hit@8 0.97, recall@8 0.95, correctness 4.39/5,
refusal calibration 9/9. See `docs/eval-scorecard.md`, `docs/ablation.md`.

**Design decisions backed by evidence.**
- *Reranking over hybrid:* a BM25+RRF hybrid did not beat dense (dense already routes perfectly);
  a cross-encoder reranker did, lifting the weak MCU category 0.75 → 0.94 (`docs/ablation.md`).
- *No fine-tuning shipped:* a QLoRA study found RAG dominates and fine-tuning on synthetic Q/A
  degraded refusal calibration (8/9 → 3/9). The system ships dense+rerank, not the adapter
  (`docs/finetune-study.md`).

**Ethical / safety notes.** The model refuses out-of-corpus and nonsensical questions rather than
hallucinating (verified on adversarial traps). Datasheet PDFs are not redistributed; the public
demo runs on the openly licensed RISC-V ISA Manual.
