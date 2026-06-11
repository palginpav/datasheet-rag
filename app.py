"""Gradio demo for datasheet-rag.

Ask questions over the RISC-V ISA Manual (CC-BY-4.0, shipped) or upload your own
PDF. Retrieval is dense + cross-encoder rerank — the configuration the project's
ablation selected. Generation backend is auto-detected:

  - OLLAMA_HOST set        -> local Ollama (qwen3:4b)
  - HF_TOKEN set           -> Hugging Face Inference API
  - neither                -> retrieval-only (shows cited chunks, no synthesis)

The demo ships only redistributable content (RISC-V spec chunks); vendor
datasheets are never bundled — upload your own to try them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import gradio as gr

from datasheet_rag.ingest.models import Chunk, load_chunks

DEMO_CHUNKS = Path("data/demo/riscv-chunks.jsonl")
DEMO_VECS = Path("data/demo/riscv-doc-vecs.npy")
K = 6
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    text = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    return _THINK.sub("", text).strip()


class InMemoryRetriever:
    """Dense + rerank over an in-memory chunk set (no Chroma; Spaces-friendly)."""

    def __init__(self, chunks: list[Chunk], doc_vecs=None) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self._chunks = chunks
        self._embed = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        self._embed.max_seq_length = 2048
        self._reranker = CrossEncoder("BAAI/bge-reranker-base")
        if doc_vecs is not None:  # precomputed — avoids embedding the corpus on a slow CPU
            import numpy as np

            self._doc_vecs = doc_vecs.astype(np.float32)
        else:
            self._doc_vecs = self._embed.encode(
                [f"search_document: {c.text}" for c in chunks],
                normalize_embeddings=True, show_progress_bar=False,
            )

    def add(self, chunks: list[Chunk]) -> None:
        new = self._embed.encode(
            [f"search_document: {c.text}" for c in chunks],
            normalize_embeddings=True, show_progress_bar=False,
        )
        import numpy as np

        self._chunks = self._chunks + chunks
        self._doc_vecs = np.vstack([self._doc_vecs, new])

    def retrieve(self, query: str, k: int = K, pool: int = 30) -> list[Chunk]:
        import numpy as np

        qv = self._embed.encode(
            [f"search_query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = self._doc_vecs @ qv
        top = np.argsort(-sims)[:pool]
        cands = [self._chunks[i] for i in top]
        scores = self._reranker.predict([(query, c.text) for c in cands])
        ranked = [
            c for c, _ in sorted(zip(cands, scores, strict=True),
                                 key=lambda cs: cs[1], reverse=True)
        ]
        return ranked[:k]


def _generate(question: str, chunks: list[Chunk]) -> str | None:
    context = "\n\n".join(
        f"[{i + 1}] {c.part}" + (f" p.{c.page}" if c.page else "") + f": {c.text}"
        for i, c in enumerate(chunks)
    )
    system = (
        "Answer using only the provided excerpts. Quote exact values and units, cite "
        'sources as [n]. If the excerpts lack the answer, reply "NOT IN CONTEXT".'
    )
    user = f"Excerpts:\n{context}\n\nQuestion: {question}"

    if os.environ.get("OLLAMA_HOST"):
        import httpx

        host = os.environ["OLLAMA_HOST"].rstrip("/")
        r = httpx.post(
            f"{host}/api/chat",
            json={"model": "qwen3:4b", "stream": False, "think": False,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=120,
        )
        return r.json()["message"]["content"]
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=os.environ["HF_TOKEN"])
            out = client.chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                model="Qwen/Qwen2.5-7B-Instruct", max_tokens=300,
            )
            return out.choices[0].message.content
        except Exception as exc:  # API unavailable -> fall back to retrieval-only
            return f"_(generation backend unavailable: {type(exc).__name__}; showing evidence)_"
    return None  # retrieval-only


def _format_sources(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        loc = f" · p.{c.page}" if c.page else ""
        section = " > ".join(c.section_path) if c.section_path else ""
        snippet = c.text.split("\n", 1)[-1][:300].strip()
        lines.append(f"**[{i}] {c.part}{loc}** — {section}\n\n> {snippet}…")
    return "\n\n".join(lines)


_retriever: InMemoryRetriever | None = None
_retriever_lock = __import__("threading").Lock()


def _get_retriever() -> InMemoryRetriever:
    global _retriever
    with _retriever_lock:  # only one builder; concurrent callers wait, not double-build
        if _retriever is None:
            import numpy as np

            vecs = np.load(DEMO_VECS) if DEMO_VECS.exists() else None
            _retriever = InMemoryRetriever(load_chunks(DEMO_CHUNKS), doc_vecs=vecs)
    return _retriever


def answer(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Ask a question above.", ""
    chunks = _get_retriever().retrieve(question)
    gen = _generate(question, chunks)
    answer_md = _strip_thinking(gen) if gen else (
        "_(no generation backend configured — showing retrieved evidence only)_"
    )
    return answer_md, _format_sources(chunks)


def add_pdf(file) -> str:
    if file is None:
        return "No file."
    from datasheet_rag.ingest.chunk import chunk_doc
    from datasheet_rag.ingest.parse import parse_pdf

    doc = parse_pdf(file.name, part=Path(file.name).stem.upper(), manufacturer="other",
                    sha256="0" * 64)
    chunks = chunk_doc(doc)
    _get_retriever().add(chunks)
    return f"Indexed {len(chunks)} chunks from {Path(file.name).name}. Ask away."


with gr.Blocks(title="datasheet-rag") as demo:
    gr.Markdown(
        "# datasheet-rag\n"
        "Ask your datasheet. Dense + cross-encoder rerank retrieval over the "
        "[RISC-V ISA Manual](https://github.com/riscv/riscv-isa-manual) (CC-BY-4.0), "
        "or upload your own PDF. [Source & write-up](https://github.com/palginpav/datasheet-rag)."
    )
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder="What is the width of a RISC-V instruction?")
    btn = gr.Button("Ask", variant="primary")
    ans = gr.Markdown(label="Answer")
    with gr.Accordion("Retrieved evidence", open=True):
        src = gr.Markdown()
    with gr.Accordion("Add your own PDF", open=False):
        up = gr.File(label="PDF", file_types=[".pdf"])
        up_status = gr.Markdown()
    gr.Examples(
        ["What is the width of a base RISC-V instruction?",
         "How many integer registers does RV32I define?",
         "What does the FENCE instruction do?"],
        inputs=q,
    )
    def _busy():
        return (
            gr.update(value="Processing…", interactive=False),
            "⏳ Retrieving and generating…",
            "",
        )

    def _idle():
        return gr.update(value="Ask", interactive=True)

    for trigger in (btn.click, q.submit):
        trigger(_busy, outputs=[btn, ans, src]).then(
            answer, inputs=q, outputs=[ans, src]
        ).then(_idle, outputs=btn)
    up.upload(add_pdf, inputs=up, outputs=up_status)


if __name__ == "__main__":
    # Warm the index in the background so launch() returns immediately and the
    # Space health check passes (a blocking pre-warm can trip the restart loop).
    import threading

    if DEMO_CHUNKS.exists():
        threading.Thread(target=_get_retriever, daemon=True).start()
    # SSR (default in Gradio 6) breaks event wiring on some Spaces — disable it.
    demo.launch(ssr_mode=False)
