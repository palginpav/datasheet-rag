"""Round-trip and validation tests for the ingestion models."""

import pytest
from pydantic import ValidationError

from datasheet_rag.ingest.models import (
    Chunk,
    DocBlock,
    ParsedDoc,
    load_chunks,
    make_chunk_id,
    save_chunks,
)

SHA = "cd" * 32


def test_parsed_doc_round_trip(tmp_path):
    d = ParsedDoc(
        part="STM32C031C4",
        manufacturer="st",
        sha256=SHA,
        source_path="corpus/st/STM32C031C4.pdf",
        n_pages=103,
        blocks=[
            DocBlock(kind="heading", text="1 Intro", section_path=["1 Intro"]),
            DocBlock(kind="text", text="prose", section_path=["1 Intro"], page=3),
            DocBlock(
                kind="table",
                text="| a | b |\n|---|---|\n| 1 | 2 |",
                section_path=["1 Intro"],
                caption="Table 1.",
                page=4,
            ),
        ],
    )
    p = tmp_path / "doc.json"
    d.save(p)
    assert ParsedDoc.load(p) == d


def test_parsed_doc_rejects_bad_sha():
    with pytest.raises(ValidationError):
        ParsedDoc(part="X", manufacturer="ti", sha256="zz", source_path="x.pdf")


def test_chunk_id_format():
    assert make_chunk_id(SHA, 7) == f"{SHA[:12]}-0007"


def test_chunks_jsonl_round_trip(tmp_path):
    chunks = [
        Chunk(
            chunk_id=make_chunk_id(SHA, i),
            doc_sha256=SHA,
            part="OPA620",
            manufacturer="ti",
            kind="text",
            text=f"[OPA620] chunk {i}",
            section_path=["1 A"],
            page=i + 1,
            n_chars=18,
        )
        for i in range(3)
    ]
    p = tmp_path / "c.chunks.jsonl"
    save_chunks(chunks, p)
    assert load_chunks(p) == chunks
