"""2단계 Hybrid + Reranker — BM25 + 임베딩 점수 결합 후 Cross-Encoder 재순위.

키워드(BM25)와 의미(임베딩)를 결합해 정확도를 끌어올린다.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk
from rag.retrieval.reranker import Reranker


class HybridRetriever:
    def __init__(self, bm25_weight: float = 0.5, reranker: Reranker | None = None) -> None:
        self.bm25_weight = bm25_weight
        self.reranker = reranker or Reranker()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # TODO: BM25 점수 + 임베딩 점수 정규화·결합 -> 후보 -> reranker.rerank
        raise NotImplementedError
