"""Grounded generation: the prompt contract and the Ollama chat client.

The contract the answer model must honor:

1. Answer ONLY from the provided context chunks.
2. Cite sources inline as ``[n]`` markers referring to the numbered chunks.
3. If the context does not contain the answer, reply starting with the
   exact token ``NOT IN CONTEXT`` — the refusal path is part of the API, not
   an error condition.

Qwen3 emits ``<think>…</think>`` reasoning blocks; we request non-thinking
mode and additionally strip any block defensively, since the citation parser
must only ever see the final answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from datasheet_rag.rag.retrieve import RetrievedChunk

REFUSAL_TOKEN = "NOT IN CONTEXT"
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = f"""You are a precise assistant answering questions about electronic components \
using ONLY the provided datasheet excerpts.

Rules:
- Use only facts present in the numbered context chunks below. Do not use prior knowledge \
about any component.
- Cite every factual claim with the chunk number in square brackets, e.g. [2].
- Quote numeric values exactly as written, including units and test conditions.
- If the context does not contain the information needed, reply starting with the exact \
words "{REFUSAL_TOKEN}" followed by a one-sentence explanation of what is missing.
- Be concise: answer first, no preamble."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = f", p.{c.page}" if c.page else ""
        parts.append(f"--- [{i}] {c.part} ({c.manufacturer}{page}) ---\n{c.text}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user = (
        f"Context chunks:\n\n{build_context_block(chunks)}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass
class GenerationResult:
    answer: str
    refused: bool
    cited_indices: list[int]  # 1-based indices into the context chunk list
    raw: str


def parse_answer(raw: str, n_chunks: int) -> GenerationResult:
    # Qwen3 via Ollama sometimes emits reasoning with only a CLOSING
    # </think> tag (the opening tag is swallowed upstream). Anything before
    # the last closing tag is reasoning, never answer — drop it first, then
    # strip any well-formed blocks that remain.
    answer = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
    answer = _THINK_BLOCK.sub("", answer).strip()
    refused = answer.upper().startswith(REFUSAL_TOKEN)
    cited = sorted(
        {int(m) for m in _CITATION.findall(answer) if 1 <= int(m) <= n_chunks}
    )
    return GenerationResult(answer=answer, refused=refused, cited_indices=cited, raw=raw)


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_ctx": 16384},
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
