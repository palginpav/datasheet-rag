"""Golden question set: schema and loader.

Each golden question pins what a correct response must contain *and* which
chunks hold the evidence, so retrieval and answer quality can be scored
independently:

- ``gold_chunk_ids`` drives retrieval metrics (did we surface the evidence?)
- ``gold_answer`` + ``must_include`` drive answer correctness (did the model
  state the right values?)
- ``answerable=False`` questions test the refusal path; they have no gold
  answer and the system is expected to decline.

Categories mirror the question types a datasheet RAG must handle:
spec-lookup, conditions (qualified values), mcu, pinout, comparison
(cross-datasheet), cross-section (synthesis within one doc), unanswerable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal[
    "spec-lookup",
    "conditions",
    "mcu",
    "pinout",
    "comparison",
    "cross-section",
    "unanswerable",
]


class GoldenQuestion(BaseModel):
    id: str = Field(pattern=r"^g\d{3}$")
    question: str = Field(min_length=8)
    category: Category
    answerable: bool = True
    gold_answer: str | None = None
    must_include: list[str] = Field(
        default_factory=list,
        description="Substrings the answer must contain (units/values), whitespace-normalized",
    )
    gold_parts: list[str] = Field(
        default_factory=list, description="Parts the evidence/answer should reference"
    )
    gold_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks containing the evidence (retrieval ground truth)"
    )
    notes: str = ""

    @model_validator(mode="after")
    def _check_answerable(self) -> GoldenQuestion:
        if self.answerable:
            if not self.gold_answer:
                raise ValueError(f"{self.id}: answerable question needs a gold_answer")
            if not self.gold_chunk_ids:
                raise ValueError(f"{self.id}: answerable question needs gold_chunk_ids")
        else:
            if self.gold_answer is not None:
                raise ValueError(f"{self.id}: unanswerable question must not have a gold_answer")
        return self


def load_golden(path: str | Path) -> list[GoldenQuestion]:
    out: list[GoldenQuestion] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            q = GoldenQuestion.model_validate_json(line)
            if q.id in seen:
                raise ValueError(f"duplicate golden id: {q.id}")
            seen.add(q.id)
            out.append(q)
    return out


def save_golden(questions: list[GoldenQuestion], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump(mode="json"), ensure_ascii=False) + "\n")
