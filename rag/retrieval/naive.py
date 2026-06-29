"""1단계 Naive — 단순 임베딩 Top-k 검색 (베이스라인)."""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class NaiveRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # TODO: 질문 임베딩 -> 벡터 DB cosine Top-k
        raise NotImplementedError
