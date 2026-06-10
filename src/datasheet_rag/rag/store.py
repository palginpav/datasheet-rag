"""Chroma persistent vector store for chunks.

Chunks are keyed by their deterministic ``chunk_id``, so re-indexing is an
upsert: unchanged documents overwrite themselves and corpus growth is
incremental. Metadata carries everything the retriever and the citation
renderer need without re-reading the chunk files.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasheet_rag.ingest.models import Chunk
from datasheet_rag.rag.embed import Embedder

DEFAULT_STORE_DIR = Path("chroma")
COLLECTION = "datasheet_chunks"


class ChunkStore:
    def __init__(self, store_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(store_dir))
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(self, chunks: list[Chunk], embedder: Embedder, batch: int = 256) -> int:
        """Embed and upsert chunks; returns the number written."""
        written = 0
        for i in range(0, len(chunks), batch):
            group = chunks[i : i + batch]
            self._collection.upsert(
                ids=[c.chunk_id for c in group],
                embeddings=embedder.embed_documents([c.text for c in group]),
                documents=[c.text for c in group],
                metadatas=[
                    {
                        "part": c.part,
                        "manufacturer": c.manufacturer,
                        "kind": c.kind,
                        "page": c.page if c.page is not None else -1,
                        "doc_sha256": c.doc_sha256,
                        "section_path": json.dumps(c.section_path, ensure_ascii=False),
                    }
                    for c in group
                ],
            )
            written += len(group)
        return written

    def query(self, query_embedding: list[float], k: int = 8) -> list[dict]:
        """Return the top-k hits as dicts: chunk_id, text, score, metadata."""
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        for chunk_id, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
            strict=True,
        ):
            meta = dict(meta)
            meta["section_path"] = json.loads(meta.get("section_path", "[]"))
            if meta.get("page") == -1:
                meta["page"] = None
            hits.append(
                {"chunk_id": chunk_id, "text": text, "score": 1.0 - dist, "metadata": meta}
            )
        return hits
