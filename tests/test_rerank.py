"""Test reranking logic with an injected stub cross-encoder (no model)."""

from datasheet_rag.rag.rerank import RerankRetriever
from datasheet_rag.rag.retrieve import RetrievedChunk


def _chunk(cid, text):
    return RetrievedChunk(
        chunk_id=cid, text=text, score=0.5, part="X", manufacturer="ti",
        kind="text", page=1, section_path=["1 A"],
    )


class _StubFirstStage:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, k=8):
        return self._chunks[:k]


class _StubEncoder:
    """Scores a (query, text) pair by how many query words appear in text."""

    def predict(self, pairs):
        out = []
        for q, t in pairs:
            qwords = set(q.lower().split())
            out.append(sum(1 for w in t.lower().split() if w in qwords))
        return out


def test_rerank_promotes_content_match():
    # first stage returns the answer-bearing chunk last; rerank should lift it
    candidates = [
        _chunk("c1", "flash timing characteristics section"),
        _chunk("c2", "unrelated power supply text"),
        _chunk("c3", "the device has 32 KB flash memory"),
    ]
    r = RerankRetriever(_StubFirstStage(candidates), pool=10, encoder=_StubEncoder())
    out = r.retrieve("how much flash memory", k=3)
    assert out[0].chunk_id == "c3"  # best content match promoted to top


def test_rerank_empty_candidates():
    r = RerankRetriever(_StubFirstStage([]), encoder=_StubEncoder())
    assert r.retrieve("q", k=5) == []
