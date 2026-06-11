---
title: datasheet-rag
emoji: 🔌
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# datasheet-rag

Retrieval-augmented QA over semiconductor datasheets — table-aware parsing, dense +
cross-encoder rerank retrieval, grounded citations. Demo runs over the RISC-V ISA Manual
(CC-BY-4.0); upload your own PDF to try a datasheet.

Full project, ablations, and the fine-tuning study: https://github.com/palginpav/datasheet-rag

**Generation backend** (set as a Space secret): `HF_TOKEN` for the Hugging Face Inference
API, or `OLLAMA_HOST` for a reachable Ollama. With neither, the demo runs retrieval-only and
shows the cited evidence.
