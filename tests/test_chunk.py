"""Chunker tests — pure fixtures, no parser dependency."""

from datasheet_rag.ingest.chunk import chunk_doc, split_long_text, split_table_by_rows
from datasheet_rag.ingest.models import DocBlock, ParsedDoc

SHA = "ab" * 32


def doc(blocks: list[DocBlock]) -> ParsedDoc:
    return ParsedDoc(
        part="OPA2993",
        manufacturer="ti",
        sha256=SHA,
        source_path="corpus/ti/OPA2993.pdf",
        n_pages=28,
        blocks=blocks,
    )


def text_block(text: str, section: list[str], page: int = 1) -> DocBlock:
    return DocBlock(kind="text", text=text, section_path=section, page=page)


def heading(section: list[str]) -> DocBlock:
    return DocBlock(kind="heading", text=section[-1], section_path=section)


def table_block(md: str, section: list[str], caption: str | None = None, page: int = 5) -> DocBlock:
    return DocBlock(kind="table", text=md, section_path=section, caption=caption, page=page)


SMALL_TABLE = "| Param | Min | Max |\n|---|---|---|\n| Vos | -1 | 1 |\n| Iq | 0.5 | 0.7 |"


# --- split_long_text -------------------------------------------------------


def test_split_short_text_is_identity():
    assert split_long_text("hello world.", 100) == ["hello world."]


def test_split_prefers_sentence_boundaries():
    text = "First sentence here. Second sentence follows. Third one ends."
    pieces = split_long_text(text, 30)
    assert all(len(p) <= 30 for p in pieces)
    assert pieces[0].endswith(".")


def test_split_falls_back_to_whitespace_then_hard_cut():
    no_sentences = "word " * 20
    pieces = split_long_text(no_sentences.strip(), 24)
    assert all(len(p) <= 24 for p in pieces)
    unbroken = "x" * 50
    pieces = split_long_text(unbroken, 20)
    assert all(len(p) <= 20 for p in pieces)
    assert "".join(pieces) == unbroken


# --- split_table_by_rows ---------------------------------------------------


def test_small_table_not_split():
    assert split_table_by_rows(SMALL_TABLE, 1000) == [SMALL_TABLE]


def test_oversized_table_splits_by_rows_and_repeats_header():
    rows = "\n".join(f"| P{i} | {i} | {i + 1} |" for i in range(40))
    table = "| Param | Min | Max |\n|---|---|---|\n" + rows
    pieces = split_table_by_rows(table, 300)
    assert len(pieces) > 1
    for piece in pieces:
        lines = piece.splitlines()
        assert lines[0] == "| Param | Min | Max |"
        assert lines[1] == "|---|---|---|"
        assert all(ln.count("|") == 4 for ln in lines)  # no row cut mid-way


def test_single_oversized_row_is_kept_intact():
    giant_row = "| P | " + "x" * 500 + " |"
    table = "| Param | Val |\n|---|---|\n" + giant_row
    pieces = split_table_by_rows(table, 200)
    # row integrity beats the size cap: the row must appear unbroken
    assert any(giant_row in p for p in pieces)


def test_merged_text_chunks_respect_cap_including_header():
    big = "Sentence of prose here. " * 95  # ~2280 chars
    d = doc(
        [
            text_block(big.strip(), ["8 Layout", "8.1 Guidelines"]),
            text_block("Short tail fragment.", ["8 Layout", "8.1 Guidelines"]),
        ]
    )
    chunks = chunk_doc(d, target_chars=2400, max_chars=2600, min_chars=200)
    assert all(c.n_chars <= 2600 for c in chunks if c.kind == "text")


def test_table_rows_are_never_lost_in_split():
    rows = [f"| P{i} | {i} | {i + 1} |" for i in range(25)]
    table = "| Param | Min | Max |\n|---|---|---|\n" + "\n".join(rows)
    pieces = split_table_by_rows(table, 250)
    recovered = [ln for p in pieces for ln in p.splitlines()[2:]]
    assert recovered == rows


# --- chunk_doc -------------------------------------------------------------


def test_headings_are_not_emitted_as_chunks():
    d = doc([heading(["1 Overview"]), text_block("Some prose here.", ["1 Overview"])])
    chunks = chunk_doc(d)
    assert len(chunks) == 1
    assert chunks[0].kind == "text"


def test_chunk_text_carries_part_and_section_header():
    d = doc([text_block("Low offset voltage amplifier.", ["6 Specs", "6.5 Electrical"])])
    [c] = chunk_doc(d)
    assert c.text.startswith("[OPA2993] 6 Specs > 6.5 Electrical\n")


def test_table_is_atomic_and_separate_from_prose():
    d = doc(
        [
            text_block("Intro prose before the table.", ["6 Specs"]),
            table_block(SMALL_TABLE, ["6 Specs"], caption="Table 6-1. Limits"),
            text_block("Prose after the table.", ["6 Specs"]),
        ]
    )
    chunks = chunk_doc(d, min_chars=10)
    kinds = [c.kind for c in chunks]
    assert kinds == ["text", "table", "text"]
    table_chunk = chunks[1]
    assert "Table 6-1. Limits" in table_chunk.text
    assert "| Vos | -1 | 1 |" in table_chunk.text


def test_section_change_flushes_prose():
    d = doc(
        [
            text_block("Alpha section prose.", ["1 Alpha"]),
            text_block("Beta section prose.", ["2 Beta"]),
        ]
    )
    chunks = chunk_doc(d, min_chars=5)
    assert len(chunks) == 2
    assert chunks[0].section_path == ["1 Alpha"]
    assert chunks[1].section_path == ["2 Beta"]


def test_long_prose_respects_max_chars():
    long_text = ("This is a sentence about electrical behavior. " * 200).strip()
    d = doc([text_block(long_text, ["7 Application"])])
    chunks = chunk_doc(d, target_chars=1800, max_chars=2600)
    assert len(chunks) > 1
    assert all(c.n_chars <= 2600 for c in chunks)


def test_tiny_fragment_merges_into_previous_same_section_chunk():
    d = doc(
        [
            text_block("A reasonably sized paragraph of prose. " * 10, ["3 Detail"]),
            text_block("Tiny tail.", ["3 Detail"]),
        ]
    )
    chunks = chunk_doc(d, target_chars=400, max_chars=2600, min_chars=200)
    assert any("Tiny tail." in c.text for c in chunks)
    # the tiny fragment must not stand alone
    assert all(c.n_chars >= 100 for c in chunks)


def test_chunk_ids_are_deterministic_and_sequential():
    d = doc(
        [
            text_block("First block of prose.", ["1 A"]),
            table_block(SMALL_TABLE, ["2 B"]),
        ]
    )
    a = chunk_doc(d, min_chars=5)
    b = chunk_doc(d, min_chars=5)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a[0].chunk_id == f"{SHA[:12]}-0000"
    assert a[1].chunk_id == f"{SHA[:12]}-0001"


def test_empty_doc_yields_no_chunks():
    assert chunk_doc(doc([])) == []
