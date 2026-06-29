"""3단계 Agentic — 모델 식별·재질의·다단계 검색.

복합 질문을 분해하고, 제품/기종을 식별해 검색 범위를 좁히며,
검색 실패 시 질의를 재구성해 재시도한다.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk
from rag.retrieval.hybrid import HybridRetriever


class AgenticRetriever:
    def __init__(self, base: HybridRetriever | None = None) -> None:
        self.base = base or HybridRetriever()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # TODO: (1) 제품/기종 식별 (2) 질의 분해/재작성 (3) 다단계 검색 (4) 실패 시 재시도
        raise NotImplementedError
