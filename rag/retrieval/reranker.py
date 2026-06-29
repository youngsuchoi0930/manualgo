"""Cross-Encoder Reranker — 후보 청크를 질문과의 관련도로 재정렬한다 (예: bge-reranker-v2-m3)."""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError
