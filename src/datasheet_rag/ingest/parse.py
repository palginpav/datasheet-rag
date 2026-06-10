"""Docling-backed PDF parsing into the intermediate representation.

This is the only module that imports Docling (a heavy optional dependency:
``pip install -e ".[ingest]"``). Everything downstream consumes
:class:`ParsedDoc`.

Mapping notes (docling 2.x):

- ``DoclingDocument.iterate_items()`` yields items in reading order.
- ``section_header`` / ``title`` items update the section-path state; the
  ``level`` field drives hierarchy depth (title resets to the root).
- ``table`` items export to GitHub-flavored markdown via
  ``TableItem.export_to_markdown(doc)``; captions come from
  ``TableItem.caption_text(doc)``.
- Standalone ``caption`` items are skipped (already attached to their table),
  as are ``page_header`` / ``page_footer`` furniture, pictures, and formulas.
- Page numbers come from item provenance (``item.prov[0].page_no``).
"""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import SectionHeaderItem, TableItem, TextItem, TitleItem
from docling_core.types.doc.labels import DocItemLabel

from datasheet_rag.ingest.models import DocBlock, ParsedDoc

_SKIP_LABELS = {
    DocItemLabel.CAPTION,  # attached to tables via caption_text()
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.CHECKBOX_UNSELECTED,
}

_TEXT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.REFERENCE,
}


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        page = getattr(prov[0], "page_no", None)
        if isinstance(page, int) and page >= 1:
            return page
    return None


class _SectionState:
    """Tracks the heading hierarchy as items stream by in reading order."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []  # (level, heading text)

    def push(self, level: int, text: str) -> list[str]:
        self._stack = [(lv, t) for lv, t in self._stack if lv < level]
        self._stack.append((level, text))
        return self.path()

    def reset(self, text: str) -> list[str]:
        self._stack = [(0, text)]
        return self.path()

    def path(self) -> list[str]:
        return [t for _, t in self._stack]


def parse_pdf(pdf_path: str | Path, *, part: str, manufacturer: str, sha256: str) -> ParsedDoc:
    """Parse one datasheet PDF into a :class:`ParsedDoc`."""
    pdf_path = Path(pdf_path)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    sections = _SectionState()
    blocks: list[DocBlock] = []

    for item, _level in doc.iterate_items():
        if isinstance(item, TitleItem):
            path = sections.reset(item.text.strip())
            blocks.append(
                DocBlock(
                    kind="heading", text=item.text.strip(), section_path=path, page=_page_of(item)
                )
            )
            continue

        if isinstance(item, SectionHeaderItem):
            text = item.text.strip()
            if not text:
                continue
            path = sections.push(item.level, text)
            blocks.append(
                DocBlock(kind="heading", text=text, section_path=path, page=_page_of(item))
            )
            continue

        if isinstance(item, TableItem):
            md = item.export_to_markdown(doc).strip()
            if not md:
                continue
            caption = (item.caption_text(doc) or "").strip() or None
            # Docling's markdown export may already lead with the caption line;
            # keep the caption only in the structured field to avoid duplication.
            if caption and md.startswith(caption):
                md = md[len(caption) :].lstrip()
                if not md:
                    continue
            blocks.append(
                DocBlock(
                    kind="table",
                    text=md,
                    section_path=sections.path(),
                    page=_page_of(item),
                    caption=caption,
                )
            )
            continue

        if isinstance(item, TextItem):
            if item.label in _SKIP_LABELS or item.label not in _TEXT_LABELS:
                continue
            text = item.text.strip()
            if not text:
                continue
            blocks.append(
                DocBlock(kind="text", text=text, section_path=sections.path(), page=_page_of(item))
            )

    n_pages = len(doc.pages) if getattr(doc, "pages", None) else None
    return ParsedDoc(
        part=part,
        manufacturer=manufacturer,
        sha256=sha256,
        source_path=str(pdf_path),
        n_pages=n_pages,
        blocks=blocks,
    )
