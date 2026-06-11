# Dataset statement — datasheet-rag

## Corpus (retrieval)

- **Source.** 99 manufacturer part numbers → 90 unique datasheet PDFs across TI, ST, NXP,
  Microchip, onsemi, and Analog Devices, plus the RISC-V ISA Manual. Selected to cover a
  signal-chain workflow: op-amps, filters, ADCs/DACs, references, regulators, interface
  transceivers, sensors, and STM32 MCUs (M0–M7).
- **Licensing.** Manufacturer datasheets are publicly downloadable but **not redistributable**,
  so the repository ships **no PDFs** — only `data/manifest.json` (part numbers + canonical URLs +
  SHA-256) and a downloader that reproduces the corpus locally. The public demo and CI use the
  **RISC-V ISA Manual (CC-BY-4.0)**, which *is* redistributable; its parsed chunks are shipped in
  `data/demo/`.
- **Processing.** Docling parses each PDF; a table-aware chunker keeps tables atomic and prepends a
  `[PART] section` context header to every chunk. 25,597 chunks total (6,222 tables). Documents are
  deduplicated by SHA-256 (vendors ship one PDF for several part variants).

## Golden evaluation set

- 100 hand-authored questions (`data/golden/golden.jsonl`) across seven categories: spec-lookup,
  conditions, mcu, comparison, cross-section, pinout, unanswerable.
- Every answer value and every `gold_chunk_id` was verified against the live index before use.
  Unanswerable questions (including adversarial traps) test the refusal path.
- Authored by the project author from the actual datasheet contents; not crowd-sourced.

## Synthetic set (fine-tuning study only)

- 243 question→answer pairs generated from corpus chunks by the local model
  (`scripts/synthesize.py`), marked `needs_review`. Used solely to train the QLoRA adapter in the
  fine-tuning study; **not** part of the shipped system or the evaluation. Disjoint from the golden
  set as a question set, though drawn from the same corpus (documented in `docs/finetune-study.md`).

## Known biases / gaps

- Corpus skews TI/ST and toward signal-chain analog + STM32; other vendors and domains are thin.
- 57 Analog Devices parts are listed in the manifest but parked (their CDN rate-limited the batch
  download); they backfill idempotently and are absent from the current index.
