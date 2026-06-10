# Build Plan

Engineering roadmap for datasheet-rag. Six phases; each phase lands as a reviewed, CI-green increment with its own results artifact. Target cadence: ~8–10 h/week.

## Phase 1 — Corpus & ingestion

**Goal:** reproducible corpus + clean parsed representation.

- [x] Manifest schema (`data/manifest.json`): `part`, `manufacturer`, `url`, `sha256`, `category`, `pages`
- [x] Downloader: httpx + retry, checksum verification, polite rate-limiting, `corpus/` git-ignored
- [ ] Populate manifest: 80–120 datasheets across TI / ST / NXP — MCUs, op-amps, LDOs, interface ICs; chosen for table density and question variety
- [x] Docling ingestion: PDF → structured doc (sections, tables, provenance)
- [x] Document dedupe by sha256 before indexing — vendors ship one PDF for part variants (observed: OPA591/OPA2591 share a single TI datasheet); index once, map both parts to it
- [x] Table-aware chunker: tables → markdown chunks carrying caption + section path; prose → section-bounded chunks (sweep sizes later, Phase 4)
- [x] Corpus stats report (`scripts/corpus_stats.py` → `docs/corpus-stats.md`): pages, tables/doc, chunk-length distributions
- [x] Open-corpus track: RISC-V ISA manual (CC-BY-4.0, release-pinned in `data/open-manifest.json`) — 901 pp, 519 tables, 2,867 chunks
- **Exit criteria:** `download → parse → chunk` runs end-to-end on 10 datasheets + RISC-V manual; chunker unit-tested; CI green

## Phase 2 — Baseline RAG

**Goal:** working end-to-end QA with citations, dense-only.

- [x] Embedding pipeline: nomic-embed-text-v1.5, Chroma persistent store (9,637 chunks indexed)
- [x] Retriever interface (swappable: dense now, hybrid later)
- [x] Generation: Qwen3-4B via Ollama; prompt contract = answer + chunk-level citations + explicit refusal when context is insufficient
- [x] CLI: `ask "What is the quiescent current of <part> at 3.3V?"`
- [x] 20 smoke questions answered qualitatively (docs/smoke-run.md): 18/18 answerable answered, 2/2 unanswerable refused; 5 observations filed as eval-set candidates
- **Exit criteria:** cited answers reproducible from a fresh clone; refusal path demonstrated

## Phase 3 — Evaluation harness

**Goal:** measurement before optimization.

- [x] Golden set: 100 hand-authored questions across all 7 categories; every value + chunk ID verified against the index (spec-lookup 40, mcu 16, conditions 14, comparison 10, unanswerable 9, cross-section 6, pinout 5)
- [x] Synthetic set: local-model generator (`scripts/synthesize.py`) — grounded Q/A pairs marked needs_review; replaces RAGAS testset gen, whose 0.4.x pipeline has a broken langchain dep chain and assumes a hosted generator
- [x] Metrics: retrieval hit@k / MRR / context recall+precision; answer faithfulness + correctness via local Ollama judge (gpt-oss) ≠ generator (qwen3)
- [x] Judge-vs-human agreement tooling: `scripts/grade.py` (interactive human grading) + `scripts/judge_agreement.py` (agreement stats + mechanical over-strictness cross-check). Human grading pass is the remaining run; 6 over-strictness candidates already flagged
- [x] Scorecard: `python -m datasheet_rag.eval [--judge]` → docs/eval-scorecard.md + per-question JSONL trace
- **Exit criteria:** baseline scorecard committed; agreement analysis written up

## Phase 4 — Hybrid retrieval & ablations

**Goal:** the headline retrieval results.

- [x] BM25 index + reciprocal-rank fusion with dense (incl. dense-weighted RRF)
- [x] Ablation matrix: dense / BM25 / hybrid / hybrid-w3 / dense+rerank (docs/ablation.md)
- [x] Part-number routing analysis (doc-hit@k metric): found dense already routes at 1.0; BM25 weak due to per-chunk part prefix
- [x] Cross-encoder reranker (bge-reranker-base): the winner — MCU hit@k 0.75 → 0.94, overall 0.92 → 0.97; wired as `--retriever dense+rerank`
- [ ] Chunking ablation: table-aware vs naive, and drop-the-part-prefix BM25 retest (follow-up the analysis points to)
- **Exit criteria:** ablation table + analysis written (docs/ablation.md); winner wired into eval and ask CLIs

## Phase 5 — Fine-tuning study

**Goal:** QLoRA vs RAG vs both — measured, not asserted.

- [x] Training set from 243 synthetic QA pairs (train/val split; disjoint from the golden eval over a shared corpus — documented)
- [x] QLoRA on Qwen3-4B (peft+TRL+bitsandbytes 4-bit, not Unsloth — torch 2.12/CUDA 13 pins); eval loss 7.5 → 2.11
- [x] Arms: base/FT × closed-book/RAG over frozen dense+rerank contexts (long-context arm dropped per scope valve)
- [x] Results chapter (docs/finetune-study.md): RAG dominates; QLoRA didn't inject knowledge and collapsed refusal calibration (3/9)
- **Exit criteria met:** all four arms measured on the golden set; honest conclusion — ship dense+rerank, not FT
- *Note: Ollama 0.11.10 can't import Qwen3 safetensors, so the FT arms run via transformers over frozen contexts rather than Ollama.*

## Phase 6 — Demo & docs

**Goal:** shippable public artifact.

- [ ] Gradio app: RISC-V corpus preloaded + upload-your-own-PDF mode; HF Spaces (ZeroGPU, CPU 4-bit fallback)
- [ ] Model card + dataset statement
- [ ] README to flagship standard: demo GIF, results tables, architecture, reproduce-in-3-commands
- [ ] 2-page case-study writeup (PDF)
- **Exit criteria:** live demo link; a stranger can reproduce headline numbers from the README

## Engineering standards (all phases)

- Tests for data structures, chunking, metric math, prompt-contract parsing; CI = ruff + pytest on every push
- Config-driven runs (YAML + seeds); pinned environment (`uv.lock`)
- No PDFs, model weights, or secrets in git; `corpus/`, `models/`, `runs/` ignored
- Conventional commit messages; small reviewed increments
