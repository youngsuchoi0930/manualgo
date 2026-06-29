"""검색기 공통 인터페이스 — 3단계(naive/hybrid/agentic) 검색기가 구현한다."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedChunk:
    chunk_id: str
    manual_id: str
    page: int
    section: str | None
    text: str
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """질문에 대해 상위 청크를 반환한다."""
        ...
