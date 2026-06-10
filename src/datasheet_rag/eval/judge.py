"""LLM-as-judge for answer faithfulness and correctness.

Run locally against Ollama, with the judge model deliberately *different*
from the generation model (default: gpt-oss judging qwen3 output) to avoid
the well-documented self-preference bias of single-model evaluation.

Two scores per answer, each 1-5 with a one-line justification:

- **faithfulness**: is every claim grounded in the provided context, with no
  fabrication? (judged against the retrieved chunks)
- **correctness**: does the answer match the gold answer? (judged against the
  golden reference)

The judge returns strict JSON; a defensive parser tolerates fences and stray
prose so a single malformed reply degrades to a recorded null rather than
crashing a long eval run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

FAITHFULNESS_PROMPT = """You are grading whether an ANSWER is faithful to the provided CONTEXT.
Faithful means every factual claim in the answer is supported by the context, with no \
invented values or specifications.

CONTEXT:
{context}

ANSWER:
{answer}

Score faithfulness from 1 to 5:
5 = every claim supported by context
3 = mostly supported, minor unsupported detail
1 = contains fabricated or contradicted claims

Reply with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""

CORRECTNESS_PROMPT = """You are grading whether an ANSWER matches the GOLD reference answer for a \
question about an electronic component. Focus on the technical values, units, and conditions — \
wording may differ.

QUESTION:
{question}

GOLD ANSWER:
{gold}

ANSWER:
{answer}

Score correctness from 1 to 5:
5 = states the same values/units/conditions as the gold answer
3 = partially correct (right quantity, wrong condition; or incomplete)
1 = wrong or missing the key value

Reply with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


@dataclass
class JudgeScore:
    score: int | None
    reason: str


def _parse_score(raw: str) -> JudgeScore:
    m = _JSON_OBJ.search(raw)
    if not m:
        return JudgeScore(None, f"unparseable judge reply: {raw[:80]!r}")
    try:
        obj = json.loads(m.group(0))
        score = int(obj["score"])
        if not 1 <= score <= 5:
            return JudgeScore(None, f"score out of range: {score}")
        return JudgeScore(score, str(obj.get("reason", "")))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        return JudgeScore(None, f"judge parse error: {exc}")


class OllamaJudge:
    def __init__(
        self,
        model: str = "gpt-oss:latest",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _ask(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    def faithfulness(self, answer: str, context: str) -> JudgeScore:
        return _parse_score(self._ask(FAITHFULNESS_PROMPT.format(context=context, answer=answer)))

    def correctness(self, question: str, answer: str, gold: str) -> JudgeScore:
        return _parse_score(
            self._ask(CORRECTNESS_PROMPT.format(question=question, gold=gold, answer=answer))
        )
