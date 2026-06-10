"""Synthetic Q/A generation from corpus chunks, via the local LLM.

Phase 5's fine-tuning study needs a larger Q/A pool than the hand-authored
golden set. Rather than RAGAS's knowledge-graph testset pipeline — whose
0.4.x release carries a heavy, presently-broken langchain dependency chain
and assumes a hosted generator — we generate grounded pairs directly with
the local model: sample a chunk, ask for one factual question and its exact
answer drawn only from that chunk, and keep the chunk as gold evidence.

Output items mark ``source="synthetic"`` and ``needs_review=true``: these are
training/eval candidates, not curated ground truth, and should be spot-checked
before any results lean on them. Generation parsing lives here (testable);
the model call is injected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

GEN_PROMPT = """You are creating a question-answer pair to test a datasheet assistant.
Below is one excerpt from the datasheet of component {part}.

EXCERPT:
{text}

Write ONE specific, factual question that this excerpt answers (about a value,
specification, pin, or feature), and the exact answer taken only from the excerpt.
The question must name the part ({part}). Keep the answer concise and include units.

Reply with ONLY a JSON object: {{"question": "...", "answer": "..."}}"""


@dataclass
class SyntheticPair:
    question: str
    answer: str
    part: str
    gold_chunk_id: str
    source: str = "synthetic"
    needs_review: bool = True


def parse_pair(raw: str, part: str, chunk_id: str) -> SyntheticPair | None:
    # qwen3 often leaks reasoning ending in a lone </think> (no opening tag);
    # take only what follows the last closing tag, then strip well-formed blocks.
    text = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
    text = _THINK.sub("", text)
    m = _JSON_OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        q, a = str(obj["question"]).strip(), str(obj["answer"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if len(q) < 8 or len(a) < 1:
        return None
    return SyntheticPair(question=q, answer=a, part=part, gold_chunk_id=chunk_id)


def generate_pair(chunk, llm) -> SyntheticPair | None:
    """chunk: a Chunk (or object with .part/.text/.chunk_id); llm: OllamaClient."""
    prompt = GEN_PROMPT.format(part=chunk.part, text=chunk.text[:2000])
    raw = llm.chat([{"role": "user", "content": prompt}])
    return parse_pair(raw, chunk.part, chunk.chunk_id)


def save_pairs(pairs: list[SyntheticPair], path) -> None:
    from dataclasses import asdict
    from pathlib import Path

    with Path(path).open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
