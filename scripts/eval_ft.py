"""Evaluate a model (base or fine-tuned) over frozen contexts.

The fine-tuning study's engine. Loads a model with transformers and answers
the golden set either closed-book (no context) or RAG (frozen dense+rerank
contexts). Scores must-include coverage locally and correctness via the same
Ollama judge (HTTP) used elsewhere, so numbers are comparable across arms.

Runs in .venv-train (transformers + the trained model). Four arms:
    --model Qwen/Qwen3-4B            (base)   + --mode closed-book | rag
    --model models/qwen3-ft-merged   (FT)     + --mode closed-book | rag
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

CLOSED_SYSTEM = (
    "You are an expert on electronic components. Answer with exact values and units "
    'from memory. If you do not know, reply starting with "NOT IN CONTEXT".'
)
RAG_SYSTEM = (
    "Answer the question using only the provided datasheet excerpts. Quote exact values "
    'and units. If the excerpts do not contain the answer, reply starting with "NOT IN CONTEXT".'
)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip(raw: str) -> str:
    text = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
    return _THINK.sub("", text).strip()


def _normalize(s: str) -> str:
    s = s.replace("μ", "µ").replace("Ω", "ω")
    return re.sub(r"\s+", "", s).lower()


def _coverage(answer: str, must: list[str]) -> float:
    if not must:
        return 1.0
    a = _normalize(answer)
    return sum(1 for s in must if _normalize(s) in a) / len(must)


def _judge_correct(question: str, answer: str, gold: str, judge_model: str) -> int | None:
    import httpx

    prompt = (
        "Score whether the ANSWER matches the GOLD answer for a question about an "
        "electronic component (values/units/conditions; wording may differ). "
        f"QUESTION: {question}\nGOLD: {gold}\nANSWER: {answer}\n"
        "Score 1-5 (5=same values/units). Reply ONLY JSON: "
        '{"score": <int>, "reason": "<one sentence>"}'
    )
    payload = {"model": judge_model, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "think": False, "options": {"temperature": 0.0}}
    try:
        r = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=300)
        raw = r.json()["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return int(json.loads(m.group(0))["score"])
    except Exception:
        return None


@app.command()
def main(
    model: str = typer.Option("Qwen/Qwen3-4B", "--model"),
    mode: str = typer.Option("rag", "--mode", help="rag | closed-book"),
    contexts: Path = typer.Option(Path("runs/frozen-contexts.jsonl"), "--contexts"),
    out: Path = typer.Option(Path("runs/ft-arm.jsonl"), "--out"),
    judge_model: str = typer.Option("gpt-oss:latest", "--judge-model"),
    device: str = typer.Option("cuda:1", "--device"),
    max_new: int = typer.Option(256, "--max-new"),
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in contexts.read_text().splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16, device_map={"": device})
    mdl.eval()

    results = []
    for r in rows:
        if mode == "closed-book":
            messages = [
                {"role": "system", "content": CLOSED_SYSTEM},
                {"role": "user", "content": r["question"]},
            ]
        else:
            ctx = "\n\n".join(
                f"[{i + 1}] {c['part']}: {c['text']}" for i, c in enumerate(r["contexts"])
            )
            messages = [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Excerpts:\n{ctx}\n\nQuestion: {r['question']}"},
            ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        answer = _strip(tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        refused = answer.upper().startswith("NOT IN CONTEXT")
        row = {"id": r["id"], "category": r["category"], "answerable": r["answerable"],
               "refused": refused, "answer": answer}
        if r["answerable"]:
            row["coverage"] = _coverage(answer, r["must_include"])
            row["correctness"] = _judge_correct(
                r["question"], answer, r["gold_answer"] or "", judge_model)
        else:
            row["refusal_correct"] = refused
        results.append(row)
        typer.echo(f"  {r['id']} {'ref' if refused else 'ans'}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ans = [r for r in results if r["answerable"]]
    corr = [r["correctness"] for r in ans if r.get("correctness") is not None]
    cov = [r["coverage"] for r in ans]
    refusals = sum(1 for r in results if not r["answerable"] and r["refusal_correct"])
    unans = sum(1 for r in results if not r["answerable"])
    typer.echo(
        f"\n{model} [{mode}]: correctness {sum(corr) / len(corr):.3f} · "
        f"coverage {sum(cov) / len(cov):.3f} · refusals {refusals}/{unans}"
    )


if __name__ == "__main__":
    app()
