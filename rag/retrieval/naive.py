"""1단계 Naive — 단순 임베딩 Top-k 검색 (베이스라인).

질문을 임베딩해 벡터 DB에서 cosine Top-k 청크를 가져온다. BM25·재순위 없음.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class NaiveRetriever:
    def __init__(self, embedder=None, store=None) -> None:
        from rag.indexing.embedder import Embedder
        from rag.vectorstore.store import ChromaStore

        self.embedder = embedder or Embedder()
        self.store = store or ChromaStore()

    def retrieve(self, query: str, top_k: int = 5, manual_ids=None) -> list[RetrievedChunk]:
        query_vec = self.embedder.encode([query], task_type="RETRIEVAL_QUERY")[0]
        hits = self.store.search(query_vec, top_k=top_k, manual_ids=manual_ids)
        return [
            RetrievedChunk(
                chunk_id=h["chunk_id"],
                manual_id=h["manual_id"],
                page=h["page"],
                section=h.get("section"),
                text=h["text"],
                score=h["score"],
            )
            for h in hits
        ]
