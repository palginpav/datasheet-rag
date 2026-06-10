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

- [~] Golden set: 25 hand-authored questions seeded (stratified, all values + chunk IDs verified against the index); target ~100 — extension is append-only JSONL, flagged for domain-expert review
- [ ] Synthetic set: RAGAS testset generation, 300–500 QA pairs
- [x] Metrics: retrieval hit@k / MRR / context recall+precision; answer faithfulness + correctness via local Ollama judge (gpt-oss) ≠ generator (qwen3)
- [ ] Human-graded subset → judge-vs-human agreement table (next: the judge-stinginess signal from the baseline makes this concrete)
- [x] Scorecard: `python -m datasheet_rag.eval [--judge]` → docs/eval-scorecard.md + per-question JSONL trace
- **Exit criteria:** baseline scorecard committed; agreement analysis written up

## Phase 4 — Hybrid retrieval & ablations

**Goal:** the headline retrieval results.

- [ ] BM25 index + reciprocal-rank fusion with dense
- [ ] Ablation matrix: dense-only / BM25-only / hybrid / hybrid+rerank (bge-reranker, optional arm)
- [ ] Chunking ablation: table-aware vs naive splitter (expected biggest delta)
- [ ] Chunk-size sweep
- [ ] Part-number exact-match stress set (LM358 vs LM359 class confusions)
- **Exit criteria:** ablation tables in README; analysis section written

## Phase 5 — Fine-tuning study

**Goal:** QLoRA vs RAG vs both — measured, not asserted.

- [ ] Training set from synthetic QA (held-out split discipline documented)
- [ ] QLoRA via Unsloth on Qwen3-4B (T4-class GPU); 8B if budget allows
- [ ] Arms: base+RAG · FT no-retrieval · FT+RAG · long-context arm (Phi-4-mini 128K, whole-datasheet-in-context)
- [ ] Adapters + training configs published (HF Hub); curves logged (MLflow/W&B)
- [ ] Results chapter: where FT helps, where RAG dominates, whether they compose — reported as measured
- **Exit criteria:** all arms on the golden-set scorecard; adapters public
- *Scope valves if behind: drop reranker arm, then long-context arm. The 3-arm core study is protected.*

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
