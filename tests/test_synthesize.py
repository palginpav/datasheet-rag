"""Tests for synthetic Q/A parsing (no model calls)."""

from datasheet_rag.eval.synthesize import parse_pair


def test_parse_clean_json():
    raw = '{"question": "What is the offset of the OPA188?", "answer": "25 µV max"}'
    p = parse_pair(raw, "OPA188", "abc123def456-0003")
    assert p is not None
    assert p.part == "OPA188"
    assert p.gold_chunk_id == "abc123def456-0003"
    assert p.source == "synthetic" and p.needs_review is True


def test_parse_strips_think_and_prose():
    raw = '<think>x</think>\n{"question": "What is the GBW of OPA211?", "answer": "80 MHz"}'
    p = parse_pair(raw, "OPA211", "x-0001")
    assert p is not None and p.answer == "80 MHz"


def test_parse_handles_orphan_closing_think_tag():
    # qwen3's real failure mode: reasoning prose (with braces) then a lone </think>
    raw = (
        "Let me see. The table {peripherals} lists values.\n</think>\n"
        '{"question": "What is the supply current of the STM32L476RG in sleep?", '
        '"answer": "81 µA"}'
    )
    p = parse_pair(raw, "STM32L476RG", "x-0001")
    assert p is not None and p.answer == "81 µA"


def test_parse_rejects_garbage():
    assert parse_pair("no json here", "X", "x-0000") is None


def test_parse_rejects_too_short_question():
    raw = '{"question": "huh", "answer": "x"}'
    assert parse_pair(raw, "X", "x-0000") is None
