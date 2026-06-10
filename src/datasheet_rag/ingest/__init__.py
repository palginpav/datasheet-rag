"""Ingestion: PDF parsing into an intermediate representation, and table-aware chunking.

The pipeline is deliberately split in two:

- ``parse``  — Docling-backed PDF → :class:`ParsedDoc` (heavy optional dependency,
  imported only here)
- ``chunk``  — :class:`ParsedDoc` → retrieval chunks (pure functions, no ML deps,
  fully unit-tested)

This keeps the chunking logic — where most of the retrieval quality lives —
testable in CI without downloading layout models.
"""

from datasheet_rag.ingest.models import Chunk, DocBlock, ParsedDoc

__all__ = ["Chunk", "DocBlock", "ParsedDoc"]
