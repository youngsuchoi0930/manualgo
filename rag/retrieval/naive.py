"""1단계 Naive — 단순 임베딩 Top-k 검색 (베이스라인).

질문을 임베딩해 벡터 DB에서 cosine Top-k 청크를 가져온다. BM25·재순위 없음.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class NaiveRetriever:
    def __init__(self, embedder=None, store=None) -> None:
        from rag.indexing.backend import make_embedder
        from rag.vectorstore.store import ChromaStore

        # 임베더·컬렉션은 backend가 짝으로 정한다 (EMBED_BACKEND=onnx|gemini)
        self.embedder = embedder or make_embedder()
        self.store = store or ChromaStore()
        # 인덱스를 만든 구성과 지금 임베더가 다르면 검색이 조용히 엉뚱해진다 → 즉시 경고
        warn = self.store.check_model(getattr(self.embedder, "model", "?"),
                                     getattr(self.embedder, "dim", 0),
                                     getattr(self.embedder, "weights", None))
        if warn:
            print(f"[!] {warn}", flush=True)

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
