"""Tests for the prompt contract, citation parsing, and pipeline plumbing.

No network, no models: the LLM and retriever are stubbed. These tests pin
the *contract* — refusal token, citation extraction, think-block stripping,
context numbering — which the eval harness will later rely on.
"""

from dataclasses import dataclass

from datasheet_rag.rag.generate import (
    REFUSAL_TOKEN,
    build_context_block,
    build_messages,
    parse_answer,
)
from datasheet_rag.rag.pipeline import ask
from datasheet_rag.rag.retrieve import RetrievedChunk


def chunk(i: int, part: str = "OPA2993", page: int | None = 7) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{'ab' * 6}-{i:04d}",
        text=f"[{part}] 6 Specs\n| Vos | {i} mV |",
        score=0.9 - i * 0.05,
        part=part,
        manufacturer="ti",
        kind="table",
        page=page,
        section_path=["6 Specs"],
    )


# --- parse_answer ----------------------------------------------------------


def test_parse_extracts_citations_in_range():
    r = parse_answer("The offset is 1 mV [1] at 25°C [3]. See also [9].", n_chunks=4)
    assert r.cited_indices == [1, 3]  # [9] out of range, dropped
    assert not r.refused


def test_parse_detects_refusal():
    r = parse_answer(f"{REFUSAL_TOKEN} — the excerpts lack quiescent current data.", n_chunks=3)
    assert r.refused
    assert r.cited_indices == []


def test_parse_strips_think_blocks():
    raw = "<think>Let me reason about Vos...</think>The value is 120 µV [2]."
    r = parse_answer(raw, n_chunks=3)
    assert r.answer == "The value is 120 µV [2]."
    assert r.cited_indices == [2]
    assert r.raw == raw


def test_refusal_detection_survives_think_block():
    raw = f"<think>hmm</think>{REFUSAL_TOKEN}: no thermal data present."
    assert parse_answer(raw, n_chunks=2).refused


def test_parse_handles_orphan_closing_think_tag():
    # Observed with qwen3 via Ollama: reasoning emitted with only the
    # closing tag. Everything before the last </think> is reasoning.
    raw = "Let me check chunk [1] and [3]... the value is clear.\n</think>\n\n±250µV [2]"
    r = parse_answer(raw, n_chunks=4)
    assert r.answer == "±250µV [2]"
    assert r.cited_indices == [2]


# --- prompt construction ---------------------------------------------------


def test_context_block_numbers_chunks_and_shows_pages():
    block = build_context_block([chunk(1), chunk(2, part="LM358", page=None)])
    assert "--- [1] OPA2993 (ti, p.7) ---" in block
    assert "--- [2] LM358 (ti) ---" in block


def test_messages_contain_contract_and_question():
    msgs = build_messages("What is Vos?", [chunk(1)])
    assert msgs[0]["role"] == "system"
    assert REFUSAL_TOKEN in msgs[0]["content"]
    assert msgs[1]["content"].endswith("Question: What is Vos?")


# --- pipeline with stubs ---------------------------------------------------


@dataclass
class StubRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        return self.chunks[:k]


class StubLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def chat(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self.reply


def test_ask_maps_citations_to_chunk_ids():
    chunks = [chunk(0), chunk(1), chunk(2)]
    result = ask("Vos?", StubRetriever(chunks), StubLLM("It is 1 mV [2]."), k=3)
    assert not result.refused
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == chunks[1].chunk_id
    assert result.citations[0].index == 2


def test_ask_handles_empty_index_as_refusal():
    result = ask("Vos?", StubRetriever([]), StubLLM("unused"))
    assert result.refused
    assert result.citations == []


def test_ask_propagates_refusal():
    result = ask(
        "Thermal resistance?",
        StubRetriever([chunk(0)]),
        StubLLM(f"{REFUSAL_TOKEN}: not in the excerpts."),
        k=1,
    )
    assert result.refused
