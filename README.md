# datasheet-rag

**Ask your datasheet.** Retrieval-augmented question answering over semiconductor datasheets — with table-aware parsing, hybrid retrieval, grounded citations, and a measured answer to the question: *what does fine-tuning buy you that retrieval doesn't?*

> **TL;DR** — a measured RAG system over 90 datasheets. Dense + cross-encoder rerank
> (hit@8 0.97, refusals 9/9). Two "obvious" upgrades — a BM25 hybrid and a QLoRA
> fine-tune — were tested and *rejected* because the evidence said so.
> **Read:** [case study (PDF)](docs/case-study.pdf) · [ablation](docs/ablation.md) ·
> [fine-tuning study](docs/finetune-study.md) · [model card](docs/model-card.md).
> **Demo:** `python app.py` (Gradio; RISC-V corpus + upload-your-own-PDF) — deploy notes below.

```text
$ python -m datasheet_rag.rag ask "What is the accuracy of the TMP117 temperature sensor?"
The TMP117 has an accuracy of up to ±0.1 °C (maximum) from -20 °C to 50 °C, ±0.15 °C
(maximum) from -40 °C to 70 °C, ±0.2 °C (maximum) from -40 °C to 100 °C, ±0.25 °C
(maximum) from -55 °C to 125 °C, and ±0.3 °C (maximum) from -55 °C to 150 °C [2].

Sources:
  [2] TMP117 p.1 · 1 Features · f8da020789b9-0000
```

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
                          (optional) cross-encoder rerank ── grounded answer + citations
                                                          ▼
                                       eval harness (golden set · local judge)
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
pip install -e ".[dev,ingest,rag]"
python -m datasheet_rag.corpus.download      # fetch corpus per manifest (PDFs stay local)
python -m datasheet_rag.ingest               # parse + chunk (Docling)
python -m datasheet_rag.rag index            # embed into Chroma
python -m datasheet_rag.rag ask "How much flash memory does the STM32C031C4 have?" --retriever dense+rerank
pytest                                       # tests run without model/heavy deps
```

Generation runs against a local [Ollama](https://ollama.com) (`ollama pull qwen3:4b`) — no
closed-API dependency anywhere in the pipeline.

## Current state

| Layer | Status |
|---|---|
| Corpus | 99 parts / 90 unique docs, 6 manufacturers, ~8,300 pages, checksum-pinned ([stats](docs/corpus-stats.md)) |
| Ingestion | Docling parsing + table-aware chunking → 25,597 chunks (6,222 tables); sha256 dedupe with part aliasing |
| Retrieval | nomic-embed + Chroma dense, BM25, RRF hybrid, and a cross-encoder reranker behind a swappable `Retriever` protocol |
| Generation | Qwen3-4B via Ollama; grounded-answer contract: context-only, `[n]` citations, `NOT IN CONTEXT` refusal |
| Evaluation | 100-question golden set, local gpt-oss judge, synthetic generator, judge-agreement tooling ([scorecard](docs/eval-scorecard.md)) |

## Results

Retrieval over the 100-question golden set ([full ablation](docs/ablation.md)):

| Retriever | hit@8 | recall@8 | MCU hit@8 |
|---|---|---|---|
| dense (baseline) | 0.92 | 0.90 | 0.75 |
| naive BM25+RRF hybrid | 0.86 | 0.84 | 0.62 |
| **dense + cross-encoder rerank** | **0.97** | **0.95** | **0.94** |

The intuitive "add BM25 for part numbers" hybrid *didn't* help — dense already routes to the
right document every time, and the residual weakness was within-document chunk ranking. A
cross-encoder reranker fixed exactly that. End-to-end with the judge, correctness rises
4.28 → 4.39 and refusal calibration stays 9/9.

**Fine-tuning study** ([full writeup](docs/finetune-study.md)): a QLoRA on Qwen3-4B was
measured against RAG over identical frozen contexts. Retrieval dominates — closed-book
correctness ~2/5 vs RAG ~4.6/5 — and fine-tuning on synthetic Q/A *hurt*: it didn't inject
the exact spec values and it collapsed refusal calibration (3/9 vs 8/9) by training the model
to always answer. **The project ships dense + rerank, the configuration the evidence supports** —
not fine-tuning.

## Roadmap

- [x] Repo scaffold, manifest schema, corpus downloader
- [x] Docling ingestion + table-aware chunking (tables atomic, row-split with repeated headers)
- [x] Baseline dense RAG with citations + refusal path
- [x] 100-question golden set + eval harness (local judge, synthetic generator, agreement tooling)
- [x] Hybrid retrieval (BM25 + RRF) + reranker + ablation
- [x] QLoRA fine-tuning study — measured RAG vs FT vs both; RAG wins, FT not shipped
- [x] Gradio demo + model card, dataset statement, case study (Spaces-ready)

## Demo

```bash
pip install -e ".[demo,rag,ingest]"
ollama pull qwen3:4b          # optional generation backend
python app.py                 # Gradio UI at http://localhost:7860
```

The demo retrieves over the shipped RISC-V ISA Manual (CC-BY-4.0) and lets you upload your own
PDF. Generation backend auto-detects `OLLAMA_HOST`, then `HF_TOKEN` (HF Inference API), else runs
retrieval-only and shows the cited evidence.

**Deploy to Hugging Face Spaces:** create a Gradio Space, push this repo, use
[`docs/SPACE_README.md`](docs/SPACE_README.md) as the Space's `README.md` (it carries the Space
metadata header), and set `HF_TOKEN` as a Space secret for generation. `requirements.txt` covers
the Space runtime.

## Docs

- [Case study (PDF)](docs/case-study.pdf) — the 2-page narrative
- [Evaluation scorecard](docs/eval-scorecard.md) · [retrieval ablation](docs/ablation.md) · [fine-tuning study](docs/finetune-study.md)
- [Model card](docs/model-card.md) · [dataset statement](docs/dataset-statement.md)
- [Build plan](docs/PLAN.md) · [corpus stats](docs/corpus-stats.md)

## License

[Apache-2.0](LICENSE). Datasheet PDFs are *not* part of this repository and remain under their manufacturers' terms.
