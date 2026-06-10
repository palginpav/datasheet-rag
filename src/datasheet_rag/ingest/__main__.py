"""Ingestion CLI: parse the downloaded corpus into ParsedDocs and chunks.

Usage:
    python -m datasheet_rag.ingest [--manifest data/manifest.json] [--corpus corpus]
                                   [--out corpus/parsed] [--limit N] [--parts OPA2993,...]

Documents are deduplicated by sha256 before parsing: vendors ship one PDF for
several part variants (e.g. OPA591/OPA2591 share a single TI datasheet), so
each unique document is parsed once under a primary part, and
``aliases.json`` records the sha256 → [parts] mapping for the indexer.
Re-runs are idempotent: documents whose outputs already exist are skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from datasheet_rag.corpus.manifest import Manifest
from datasheet_rag.ingest.chunk import chunk_doc
from datasheet_rag.ingest.models import save_chunks

app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Path = typer.Option(Path("data/manifest.json"), "--manifest"),
    corpus: Path = typer.Option(Path("corpus"), "--corpus"),
    out: Path = typer.Option(Path("corpus/parsed"), "--out"),
    limit: int | None = typer.Option(None, "--limit", help="Parse at most N unique documents"),
    parts: str | None = typer.Option(None, "--parts", help="Comma-separated part filter"),
) -> None:
    from datasheet_rag.ingest.parse import parse_pdf  # heavy import deferred to runtime

    manifest = Manifest.load(manifest_path)
    wanted = {p.strip().upper() for p in parts.split(",")} if parts else None

    # Dedupe by sha256; first entry becomes the primary part.
    by_sha: dict[str, list] = {}
    for e in manifest.entries:
        if e.sha256 is None:
            typer.echo(f"skip  {e.part}: no pinned sha256 (run the downloader first)", err=True)
            continue
        by_sha.setdefault(e.sha256, []).append(e)

    out.mkdir(parents=True, exist_ok=True)
    aliases = {sha: [e.part for e in entries] for sha, entries in by_sha.items()}
    (out / "aliases.json").write_text(json.dumps(aliases, indent=2) + "\n", encoding="utf-8")

    selected = []
    for sha, entries in by_sha.items():
        primary = entries[0]
        if wanted and not any(e.part in wanted for e in entries):
            continue
        selected.append((sha, primary, [e.part for e in entries]))
    if limit is not None:
        selected = selected[:limit]

    if not selected:
        typer.echo("Nothing selected to parse.")
        raise typer.Exit(0)

    failed = 0
    for sha, primary, alias_parts in selected:
        doc_json = out / f"{sha[:12]}.json"
        chunks_jsonl = out / f"{sha[:12]}.chunks.jsonl"
        if doc_json.exists() and chunks_jsonl.exists():
            typer.echo(f"skip  {primary.part}: outputs exist")
            continue

        pdf_path = corpus / primary.manufacturer / f"{primary.part}.pdf"
        t0 = time.monotonic()
        try:
            doc = parse_pdf(
                pdf_path, part=primary.part, manufacturer=primary.manufacturer, sha256=sha
            )
            chunks = chunk_doc(doc)
            doc.save(doc_json)
            save_chunks(chunks, chunks_jsonl)
        except Exception as exc:  # noqa: BLE001 — CLI boundary: report and continue
            typer.echo(f"FAIL  {primary.part}: {exc}", err=True)
            failed += 1
            continue
        elapsed = time.monotonic() - t0
        n_tables = sum(1 for b in doc.blocks if b.kind == "table")
        alias_note = f" (aliases: {', '.join(alias_parts)})" if len(alias_parts) > 1 else ""
        typer.echo(
            f"  ok  {primary.part}: pages={doc.n_pages} blocks={len(doc.blocks)} "
            f"tables={n_tables} chunks={len(chunks)} {elapsed:.1f}s{alias_note}"
        )

    raise typer.Exit(1 if failed else 0)


if __name__ == "__main__":
    app()
