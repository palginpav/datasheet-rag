"""Table-aware chunking of parsed datasheets.

Datasheet QA fails at the chunking layer more often than at the model layer:
electrical characteristics live in wide tables, and a splitter that cuts a
table mid-row (or glues half a table to unrelated prose) destroys the only
context that made the numbers answerable. The rules here:

- **Tables are atomic.** One table block → one chunk, never merged with prose.
  Oversized tables split by *rows*, with the markdown header row repeated in
  every piece so each piece remains a self-describing table. A single row
  wider than the cap is kept intact — row integrity beats the size limit, so
  ``max_chars`` is a soft bound in that (rare) pathological case.
- **Prose follows sections.** Text accumulates within one ``section_path`` and
  flushes at section boundaries or the size target, splitting on sentence
  boundaries under the hard cap.
- **Every chunk is self-locating.** Chunk text begins with a context header —
  ``[PART] Section > Subsection`` — so the embedding carries part identity and
  document position even when the body is a bare table.

Pure functions over :class:`ParsedDoc`; no parser or ML dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from datasheet_rag.ingest.models import Chunk, ParsedDoc, make_chunk_id

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def context_header(part: str, section_path: list[str]) -> str:
    if section_path:
        return f"[{part}] {' > '.join(section_path)}"
    return f"[{part}]"


def split_long_text(text: str, max_len: int) -> list[str]:
    """Split text into pieces of at most ``max_len`` chars, preferring sentence
    boundaries, falling back to whitespace, then to a hard cut."""
    if len(text) <= max_len:
        return [text]

    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        window = remaining[: max_len + 1]
        # Prefer the last sentence boundary in the window.
        boundaries = list(_SENTENCE_BOUNDARY.finditer(window))
        if boundaries:
            cut = boundaries[-1].start() + 1  # keep the punctuation
        else:
            cut = window.rfind(" ")
            if cut <= 0:
                cut = max_len  # hard cut: no whitespace at all
        pieces.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def split_table_by_rows(table_md: str, max_len: int) -> list[str]:
    """Split a markdown table into row-groups, repeating the header+separator
    rows in every piece. Never cuts inside a row."""
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if len(lines) <= 2:  # header-only or degenerate "table"
        return [table_md]

    header, separator, rows = lines[0], lines[1], lines[2:]
    head_len = len(header) + len(separator) + 2  # + newlines

    pieces: list[str] = []
    current: list[str] = []
    current_len = head_len
    for row in rows:
        row_len = len(row) + 1
        if current and current_len + row_len > max_len:
            pieces.append("\n".join([header, separator, *current]))
            current, current_len = [], head_len
        current.append(row)
        current_len += row_len
    if current:
        pieces.append("\n".join([header, separator, *current]))
    return pieces


@dataclass
class _ProseBuffer:
    section_path: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    page: int | None = None

    def size(self) -> int:
        return sum(len(t) for t in self.texts) + 2 * max(len(self.texts) - 1, 0)

    def reset(self, section_path: list[str]) -> None:
        self.section_path = list(section_path)
        self.texts = []
        self.page = None


@dataclass
class _Draft:
    """A chunk before ID assignment."""

    kind: str
    body: str  # text WITHOUT the context header
    section_path: list[str]
    page: int | None
    caption: str | None = None


def _drain_prose(
    buf: _ProseBuffer, *, max_chars: int, min_chars: int, header_len: int
) -> list[_Draft]:
    if not buf.texts:
        return []
    body = "\n\n".join(buf.texts)
    budget = max(max_chars - header_len - 1, 1)
    pieces = split_long_text(body, budget)
    # Merge a tiny trailing piece into its predecessor when it still fits.
    if len(pieces) >= 2 and len(pieces[-1]) < min_chars:
        merged = pieces[-2] + " " + pieces[-1]
        if len(merged) <= budget:
            pieces = [*pieces[:-2], merged]
    return [_Draft("text", p, list(buf.section_path), buf.page) for p in pieces]


def chunk_doc(
    doc: ParsedDoc,
    *,
    target_chars: int = 1800,
    max_chars: int = 2600,
    min_chars: int = 200,
) -> list[Chunk]:
    """Chunk a parsed datasheet per the module rules. Deterministic: identical
    input and parameters produce identical chunk IDs and text."""
    drafts: list[_Draft] = []
    buf = _ProseBuffer()

    def flush() -> None:
        header_len = len(context_header(doc.part, buf.section_path))
        drafts.extend(
            _drain_prose(buf, max_chars=max_chars, min_chars=min_chars, header_len=header_len)
        )
        buf.texts = []
        buf.page = None

    for block in doc.blocks:
        if block.kind == "heading":
            flush()
            buf.reset(block.section_path)
            continue

        if block.kind == "table":
            flush()
            header = context_header(doc.part, block.section_path)
            prefix = header + "\n" + (f"{block.caption}\n" if block.caption else "")
            budget = max(max_chars - len(prefix), 1)
            for piece in split_table_by_rows(block.text, budget):
                drafts.append(
                    _Draft("table", piece, list(block.section_path), block.page, block.caption)
                )
            buf.reset(block.section_path)
            continue

        # text block
        if block.section_path != buf.section_path:
            flush()
            buf.reset(block.section_path)
        if buf.page is None:
            buf.page = block.page
        stripped = block.text.strip()
        if stripped:
            buf.texts.append(stripped)
        if buf.size() >= target_chars:
            flush()

    flush()

    # Final pass: merge tiny text drafts into the preceding text draft of the
    # same section when the result (including its context header) stays under
    # the cap.
    merged: list[_Draft] = []
    for d in drafts:
        prev = merged[-1] if merged else None
        if (
            d.kind == "text"
            and prev is not None
            and prev.kind == "text"
            and prev.section_path == d.section_path
            and len(d.body) < min_chars
        ):
            header_len = len(context_header(doc.part, prev.section_path)) + 1
            if header_len + len(prev.body) + 1 + len(d.body) <= max_chars:
                prev.body = prev.body + " " + d.body
                continue
        merged.append(d)

    chunks: list[Chunk] = []
    for i, d in enumerate(merged):
        header = context_header(doc.part, d.section_path)
        if d.kind == "table" and d.caption:
            text = f"{header}\n{d.caption}\n{d.body}"
        else:
            text = f"{header}\n{d.body}"
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.sha256, i),
                doc_sha256=doc.sha256,
                part=doc.part,
                manufacturer=doc.manufacturer,
                kind=d.kind,  # type: ignore[arg-type]
                text=text,
                section_path=d.section_path,
                page=d.page,
                n_chars=len(text),
            )
        )
    return chunks
