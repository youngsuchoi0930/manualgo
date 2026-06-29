"""벡터 DB 래퍼 — chroma/faiss 백엔드를 공통 인터페이스로 추상화한다."""
from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...
    def search(self, query_vector: list[float], top_k: int) -> list[dict]: ...
