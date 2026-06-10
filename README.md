# datasheet-rag

**Ask your datasheet.** Retrieval-augmented question answering over semiconductor datasheets — with table-aware parsing, hybrid retrieval, grounded citations, and a measured answer to the question: *what does fine-tuning buy you that retrieval doesn't?*

> Status: 🚧 active development — see [docs/PLAN.md](docs/PLAN.md) for the build roadmap.

## Why

Datasheets are where engineering answers live — and where naive RAG dies. The information is dense, table-bound, and conditions-qualified ("I<sub>Q</sub> = 25 µA *at 3.3 V, −40 °C*"), and part numbers punish fuzzy matching: a system that confuses an LM358 with an LM359 is worse than no system. This project treats those properties as the design brief:

- **Table-preserving ingestion** — structure-aware parsing (Docling), tables kept as first-class chunks with section context
- **Hybrid retrieval** — dense embeddings + BM25 with reciprocal-rank fusion, because exact part-number hits are non-negotiable
- **Grounded generation** — answers cite their source chunks; questions the corpus can't answer get a refusal, not a hallucination
- **Evaluation as a feature** — a hand-authored golden QA set, synthetic scale-out, and judge-vs-human agreement reporting
- **A controlled fine-tuning study** — QLoRA vs RAG vs both, on open-weight models, results reported as measured

## Architecture

```
manifest.json ──> download_corpus ──> Docling parse ──> table-aware chunking
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                        Chroma (dense)                  BM25 index
                                              └───────────┬───────────────────┘
                                                          ▼  RRF fusion
                                                    retrieved context
                                                          ▼
                                       Qwen3 (local) ── grounded answer + citations
                                                          ▼
                                          eval harness (golden set · RAGAS · DeepEval)
```

## Corpus & licensing — read this first

Manufacturer datasheets (TI, ST, NXP, …) are **publicly downloadable but not redistributable**. This repository therefore ships **no datasheet PDFs**. Instead:

- [`data/manifest.json`](data/manifest.json) lists part numbers with canonical manufacturer URLs and checksums
- `python -m datasheet_rag.corpus.download` fetches the corpus to your machine (`corpus/`, git-ignored)
- CI and the public demo run on an openly licensed technical corpus (the [RISC-V ISA Manual](https://github.com/riscv/riscv-isa-manual), CC-BY-4.0) and user-supplied documents

The same constraint shaped the model choices: everything here runs on open weights (Apache-2.0 / MIT licensed models) with a local judge for evaluation — no closed-API dependency anywhere in the pipeline.

## Quickstart

```bash
git clone https://github.com/palginpav/datasheet-rag && cd datasheet-rag
uv sync                                      # or: pip install -e ".[dev]"
python -m datasheet_rag.corpus.download      # fetch corpus per manifest
pytest                                       # run tests
```

## Roadmap

- [x] Repo scaffold, manifest schema, corpus downloader
- [ ] Docling ingestion + table-aware chunking
- [ ] Baseline dense RAG with citations
- [ ] Golden QA set (100 questions) + eval harness (RAGAS / DeepEval, local judge)
- [ ] Hybrid retrieval (BM25 + RRF) + ablations
- [ ] QLoRA fine-tuning study (Qwen3, Unsloth) — RAG vs FT vs hybrid
- [ ] Gradio demo (HF Spaces)

## License

[Apache-2.0](LICENSE). Datasheet PDFs are *not* part of this repository and remain under their manufacturers' terms.
