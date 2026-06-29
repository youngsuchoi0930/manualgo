"""벡터 DB 래퍼 — 공통 인터페이스(Protocol)와 ChromaDB 구현.

ChromaDB를 cosine 공간으로 영속화한다. 임베딩은 외부(Embedder)에서 계산해 주입한다.
나중에 FAISS 구현을 같은 인터페이스로 추가할 수 있다.
"""
from __future__ import annotations

from typing import Protocol

# Gemini 임베딩(768d)용 컬렉션. 기존 BGE-M3(1024d) "manuals"와 차원이 달라 분리한다.
COLLECTION = "manuals_gemini"


class VectorStore(Protocol):
    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None: ...
    def search(self, query_vector: list[float], top_k: int) -> list[dict]: ...


class ChromaStore:
    """ChromaDB 영속 저장소 (cosine)."""

    def __init__(self, persist_dir: str = "data/index", collection: str = COLLECTION) -> None:
        import chromadb

        self._name = collection
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """컬렉션을 비우고 새로 만든다 (폴더 전체로 재구축할 때 사용)."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        self._col.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        res = self._col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        hits: list[dict] = []
        for i in range(len(res["ids"][0])):
            md = res["metadatas"][0][i] or {}
            hits.append(
                {
                    "chunk_id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "score": 1.0 - res["distances"][0][i],  # cosine 거리 → 유사도
                    "manual_id": md.get("manual_id"),
                    "page": md.get("page"),
                    "section": md.get("section") or None,
                }
            )
        return hits

    def count(self) -> int:
        return self._col.count()
