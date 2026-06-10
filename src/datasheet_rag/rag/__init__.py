"""Baseline RAG: dense retrieval over chunked datasheets with grounded generation.

Module layout mirrors the swap points we will exercise in later phases:

- ``embed``    — embedding model wrapper (dense vectors; query/document asymmetric)
- ``store``    — Chroma persistent vector store keyed by chunk_id
- ``retrieve`` — the ``Retriever`` protocol + dense implementation (hybrid lands
  in Phase 4 behind the same interface)
- ``generate`` — Ollama chat client and the grounded-answer prompt contract
- ``pipeline`` — ask(): retrieve → prompt → generate → parse citations
"""
