"""2단계 Hybrid — BM25(키워드) + 임베딩(의미)을 Reciprocal Rank Fusion으로 결합.

키워드와 의미를 합쳐 정확도를 끌어올린다. RRF는 점수 정규화가 필요 없어 안정적이다.
(Cross-Encoder Reranker는 torch라 Windows 크래시 → 서버/이후로 미룸. 여기선 BM25+임베딩만.)

질의마다 임베딩 1회가 필요하다(임베딩 부분). BM25 부분은 쿼터 0.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class HybridRetriever:
    def __init__(self, bm25=None, dense=None, rrf_k: int = 60, pool: int = 20) -> None:
        from rag.retrieval.bm25 import BM25Retriever
        from rag.retrieval.naive import NaiveRetriever

        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or NaiveRetriever()
        self.rrf_k = rrf_k      # RRF 상수 (관례상 60)
        self.pool = pool        # 각 검색기에서 가져올 후보 수

    def retrieve(self, query: str, top_k: int = 5, manual_ids=None) -> list[RetrievedChunk]:
        lists = [
            self.bm25.retrieve(query, top_k=self.pool, manual_ids=manual_ids),
            self.dense.retrieve(query, top_k=self.pool, manual_ids=manual_ids),
        ]
        rrf: dict[str, float] = {}
        objs: dict[str, RetrievedChunk] = {}
        for ranked in lists:
            for rank, c in enumerate(ranked, start=1):
                rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                objs.setdefault(c.chunk_id, c)
        top = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=cid,
                manual_id=objs[cid].manual_id,
                page=objs[cid].page,
                section=objs[cid].section,
                text=objs[cid].text,
                score=rrf[cid],
            )
            for cid in top
        ]
