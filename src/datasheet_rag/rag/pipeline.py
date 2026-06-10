"""End-to-end ask pipeline: retrieve → prompt → generate → cite.

Returns a structured :class:`AskResult` so the CLI, the demo app, and the
eval harness all consume the same object.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasheet_rag.rag.generate import GenerationResult, OllamaClient, build_messages, parse_answer
from datasheet_rag.rag.retrieve import RetrievedChunk, Retriever


@dataclass
class Citation:
    index: int  # 1-based position in the context block
    chunk_id: str
    part: str
    page: int | None
    section_path: list[str]
    score: float


@dataclass
class AskResult:
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunk]


def ask(
    question: str,
    retriever: Retriever,
    llm: OllamaClient,
    *,
    k: int = 8,
) -> AskResult:
    chunks = retriever.retrieve(question, k=k)
    if not chunks:
        return AskResult(
            question=question,
            answer="NOT IN CONTEXT — the index returned no results.",
            refused=True,
            citations=[],
            retrieved=[],
        )
    raw = llm.chat(build_messages(question, chunks))
    gen: GenerationResult = parse_answer(raw, n_chunks=len(chunks))
    citations = [
        Citation(
            index=i,
            chunk_id=chunks[i - 1].chunk_id,
            part=chunks[i - 1].part,
            page=chunks[i - 1].page,
            section_path=chunks[i - 1].section_path,
            score=chunks[i - 1].score,
        )
        for i in gen.cited_indices
    ]
    return AskResult(
        question=question,
        answer=gen.answer,
        refused=gen.refused,
        citations=citations,
        retrieved=chunks,
    )
