"""Reproducible corpus downloader.

Usage:
    python -m datasheet_rag.corpus.download [--manifest data/manifest.json] [--dest corpus]

Downloads every manifest entry to ``<dest>/<manufacturer>/<part>.pdf``,
verifying (or recording) SHA-256 checksums. Polite by design: sequential
requests with a short delay — these are vendor documentation servers.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx
import typer

from datasheet_rag.corpus.manifest import Manifest, ManifestEntry

app = typer.Typer(add_completion=False)

USER_AGENT = "datasheet-rag/0.1 (corpus reproduction; see repository README)"
RETRY_STATUSES = {429, 500, 502, 503, 504}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(entry: ManifestEntry, dest_dir: Path, client: httpx.Client, retries: int = 3) -> Path:
    """Download one entry, returning the local path. Verifies checksum when pinned."""
    target = dest_dir / entry.manufacturer / f"{entry.part}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and entry.sha256 and sha256_of(target) == entry.sha256:
        return target  # already present and verified

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(str(entry.url), follow_redirects=True)
            if resp.status_code in RETRY_STATUSES:
                raise httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            target.write_bytes(resp.content)
            digest = sha256_of(target)
            if entry.sha256 and digest != entry.sha256:
                target.unlink(missing_ok=True)
                raise ValueError(
                    f"{entry.part}: checksum mismatch (expected {entry.sha256[:12]}…, "
                    f"got {digest[:12]}…) — upstream PDF may have been revised; "
                    "re-pin deliberately if so"
                )
            return target
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if isinstance(exc, ValueError):
                break  # checksum mismatch is not retryable
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"failed to fetch {entry.manufacturer}/{entry.part}: {last_error}")


@app.command()
def main(
    manifest_path: Path = typer.Option(Path("data/manifest.json"), "--manifest"),
    dest: Path = typer.Option(Path("corpus"), "--dest"),
    delay_s: float = typer.Option(1.0, "--delay", help="Pause between downloads (politeness)"),
) -> None:
    manifest = Manifest.load(manifest_path)
    if not manifest.entries:
        typer.echo("Manifest has no entries yet — nothing to download.")
        raise typer.Exit(0)

    ok, failed = 0, 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0) as client:
        for entry in manifest.entries:
            try:
                path = fetch(entry, dest, client)
                typer.echo(f"  ok  {entry.manufacturer}/{entry.part} -> {path}")
                ok += 1
            except RuntimeError as exc:
                typer.echo(f"FAIL  {exc}", err=True)
                failed += 1
            time.sleep(delay_s)

    typer.echo(f"done: {ok} ok, {failed} failed, {len(manifest.entries)} total")
    raise typer.Exit(1 if failed else 0)


if __name__ == "__main__":
    app()
