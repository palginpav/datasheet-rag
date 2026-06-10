"""RAG CLI: build the index and ask questions.

Usage:
    python -m datasheet_rag.rag index [--parsed corpus/parsed] [--store chroma]
    python -m datasheet_rag.rag ask "What is the offset voltage of OPA2993?" [--k 8]
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def index(
    parsed: Path = typer.Option(Path("corpus/parsed"), "--parsed"),
    store_dir: Path = typer.Option(Path("chroma"), "--store"),
    device: str | None = typer.Option(None, "--device", help="cuda / cpu / None=auto"),
) -> None:
    """Embed all chunk files and upsert them into the Chroma store."""
    from datasheet_rag.ingest.models import load_chunks
    from datasheet_rag.rag.embed import NomicEmbedder
    from datasheet_rag.rag.store import ChunkStore

    files = sorted(parsed.glob("*.chunks.jsonl"))
    if not files:
        typer.echo("No chunk files found — run `python -m datasheet_rag.ingest` first.", err=True)
        raise typer.Exit(1)

    embedder = NomicEmbedder(device=device)
    store = ChunkStore(store_dir)
    total = 0
    for f in files:
        chunks = load_chunks(f)
        n = store.upsert_chunks(chunks, embedder)
        total += n
        typer.echo(f"  indexed {f.stem.split('.')[0]}: {n} chunks")
    typer.echo(f"done: {total} chunks upserted; store now holds {store.count()}")


@app.command()
def ask(
    question: str = typer.Argument(...),
    k: int = typer.Option(8, "--k"),
    store_dir: Path = typer.Option(Path("chroma"), "--store"),
    model: str = typer.Option("qwen3:4b", "--model"),
    show_chunks: bool = typer.Option(False, "--show-chunks"),
) -> None:
    """Ask a question against the indexed corpus."""
    from datasheet_rag.rag.embed import NomicEmbedder
    from datasheet_rag.rag.generate import OllamaClient
    from datasheet_rag.rag.pipeline import ask as run_ask
    from datasheet_rag.rag.retrieve import DenseRetriever
    from datasheet_rag.rag.store import ChunkStore

    retriever = DenseRetriever(ChunkStore(store_dir), NomicEmbedder())
    result = run_ask(question, retriever, OllamaClient(model=model), k=k)

    typer.echo(result.answer)
    if result.citations:
        typer.echo("\nSources:")
        for c in result.citations:
            section = " > ".join(c.section_path) or "—"
            page = f" p.{c.page}" if c.page else ""
            typer.echo(f"  [{c.index}] {c.part}{page} · {section} · {c.chunk_id}")
    if show_chunks:
        typer.echo("\nRetrieved:")
        for i, r in enumerate(result.retrieved, start=1):
            typer.echo(f"  [{i}] {r.score:.3f} {r.part} {r.chunk_id} ({r.kind})")


if __name__ == "__main__":
    app()
