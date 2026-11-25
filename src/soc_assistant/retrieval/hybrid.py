"""Combine BM25 and dense search with Reciprocal Rank Fusion (RRF).

BM25 scores and cosine similarities live on different scales, so we do not add them.
RRF only uses each result's *rank*:   score(chunk) = sum over retrievers of 1 / (k + rank)
A chunk ranked high by both retrievers wins; one ranked high by a single retriever still
gets in. k = 60 is the standard constant from the original RRF paper (Cormack et al., 2009).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from soc_assistant.config import PROJECT_ROOT, load_config
from soc_assistant.ingest.documents import Chunk, read_jsonl
from soc_assistant.retrieval.bm25 import BM25Index
from soc_assistant.retrieval.dense import DenseIndex, Embedder

MODES = ("bm25", "dense", "hybrid")


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked lists of item ids. Ranks start at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    chunk: Chunk
    score: float                 # score used for ranking in the chosen mode
    dense_score: float | None    # cosine similarity (for the refusal gate), if computed
    bm25_rank: int | None
    dense_rank: int | None


def chunks_path() -> Path:
    return PROJECT_ROOT / load_config()["paths"]["processed"] / "chunks.jsonl"


@lru_cache(maxsize=1)
def load_chunks() -> list[Chunk]:
    path = chunks_path()
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `make corpus` first.")
    return read_jsonl(path, Chunk)


class Retriever:
    def __init__(self, embedding_model: str | None = None, chunks: list[Chunk] | None = None):
        cfg = load_config()["retrieval"]
        self.candidates = cfg.get("candidates", 30)
        self.rrf_k = cfg.get("rrf_k", 60)
        self.chunks = chunks or load_chunks()
        self.bm25 = BM25Index(self.chunks)
        self.dense = DenseIndex(Embedder(embedding_model), self.chunks)

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Hit]:
        return self.search_with_score(query, k, mode)[0]

    def search_with_score(self, query: str, k: int = 4, mode: str = "hybrid") -> tuple[list[Hit], float]:
        """Like `search`, plus the best dense cosine score over the whole corpus.

        The refusal gate needs that score in every mode. A hit's own `dense_score` is not
        enough: in hybrid mode the top chunk may come from BM25 only (dense_score None).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = max(self.candidates, k)
        bm = self.bm25.search(query, n) if mode in ("bm25", "hybrid") else []
        de = self.dense.search(query, n if mode != "bm25" else 1)
        best_dense = de[0][1] if de else 0.0
        if mode == "bm25":
            de = []      # only used for the gate score, not for ranking
        bm_rank = {i: r for r, (i, _) in enumerate(bm, start=1)}
        de_rank = {i: r for r, (i, _) in enumerate(de, start=1)}
        de_score = dict(de)

        if mode == "bm25":
            ranked = bm
        elif mode == "dense":
            ranked = de
        else:
            ranked = rrf([[i for i, _ in bm], [i for i, _ in de]], self.rrf_k)

        hits = [Hit(self.chunks[i], s, de_score.get(i), bm_rank.get(i), de_rank.get(i))
                for i, s in ranked[:k]]
        return hits, best_dense
