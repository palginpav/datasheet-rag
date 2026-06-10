"""Tests for BM25 tokenization and RRF fusion (no models)."""

from datasheet_rag.rag.bm25 import tokenize
from datasheet_rag.rag.hybrid import HybridRetriever, rrf_fuse
from datasheet_rag.rag.retrieve import RetrievedChunk

# --- tokenize --------------------------------------------------------------


def test_part_number_stays_one_token():
    assert "stm32c031c4" in tokenize("What is the flash of the STM32C031C4?")


def test_hyphenated_part_kept():
    assert "tlv1117-33" in tokenize("TLV1117-33 dropout")


def test_lowercases_and_drops_punct():
    assert tokenize("OPA188: 25µV!") == ["opa188", "25", "v"]  # µ is non-ascii, dropped


# --- rrf_fuse --------------------------------------------------------------


def test_rrf_rewards_agreement():
    # 'b' is rank 1 in both rankings -> unambiguous winner
    fused = rrf_fuse([["b", "a", "c"], ["b", "a", "d"]])
    ids = [cid for cid, _ in fused]
    assert ids[0] == "b"


def test_rrf_rescues_exact_match_from_one_ranker():
    # 'x' only in the second ranker but at rank 1 -> must appear, not be dropped
    fused = rrf_fuse([["a", "b", "c"], ["x", "a", "b"]])
    ids = [cid for cid, _ in fused]
    assert "x" in ids
    assert ids.index("x") <= ids.index("c")


def test_rrf_scores_descend():
    fused = rrf_fuse([["a", "b"], ["a", "c"]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0][0] == "a"  # only id in both rankings


# --- HybridRetriever with stub sub-retrievers ------------------------------


def _chunk(cid, part="X"):
    return RetrievedChunk(
        chunk_id=cid, text=f"[{part}] body", score=1.0, part=part,
        manufacturer="ti", kind="text", page=1, section_path=["1 A"],
    )


class _Stub:
    def __init__(self, ids):
        self._ids = ids

    def retrieve(self, query, k=8):
        return [_chunk(cid) for cid in self._ids[:k]]


def test_hybrid_fuses_and_dedupes():
    dense = _Stub(["a", "b", "c"])
    bm25 = _Stub(["d", "a", "b"])  # 'd' is bm25's exact hit
    hy = HybridRetriever([dense, bm25], pool=10)
    out = hy.retrieve("q", k=4)
    ids = [c.chunk_id for c in out]
    assert "d" in ids  # exact-match rescue
    assert len(ids) == len(set(ids))  # deduped
    assert "a" == ids[0]  # ranked by both -> top
