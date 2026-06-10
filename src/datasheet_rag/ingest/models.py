"""Intermediate representation between PDF parsing and chunking.

These models are the contract that isolates the rest of the pipeline from the
PDF parser: the chunker, indexer, and tests all consume :class:`ParsedDoc` and
never touch Docling types. Tables are carried as GitHub-flavored markdown so a
chunk's text is directly embeddable and human-readable in eval reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

BlockKind = Literal["text", "table", "heading"]
ChunkKind = Literal["text", "table"]


class DocBlock(BaseModel):
    """One content block in reading order."""

    kind: BlockKind
    text: str
    section_path: list[str] = Field(
        default_factory=list,
        description=(
            "Heading hierarchy at this point, "
            "e.g. ['6 Specifications', '6.5 Electrical Characteristics']"
        ),
    )
    page: int | None = Field(default=None, ge=1)
    caption: str | None = None


class ParsedDoc(BaseModel):
    """A parsed datasheet: ordered blocks plus document identity."""

    part: str
    manufacturer: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str
    n_pages: int | None = Field(default=None, ge=1)
    blocks: list[DocBlock] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> ParsedDoc:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class Chunk(BaseModel):
    """One retrieval unit. ``text`` is the embeddable payload, already
    prefixed with part and section context by the chunker."""

    chunk_id: str = Field(pattern=r"^[0-9a-f]{12}-\d{4}$")
    doc_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    part: str
    manufacturer: str
    kind: ChunkKind
    text: str
    section_path: list[str] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)
    n_chars: int = Field(ge=0)


def make_chunk_id(sha256: str, index: int) -> str:
    return f"{sha256[:12]}-{index:04d}"


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Chunk.model_validate(json.loads(line)))
    return out
