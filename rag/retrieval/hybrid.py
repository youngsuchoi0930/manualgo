"""2단계 Hybrid — BM25(키워드) + 임베딩(의미)을 Reciprocal Rank Fusion으로 결합.

키워드와 의미를 합쳐 정확도를 끌어올린다. RRF는 점수 정규화가 필요 없어 안정적이다.
선택적으로 Cross-Encoder Reranker(ONNX, torch 우회)를 얹어 상위 후보를 재정렬한다.

질의마다 임베딩 1회가 필요하다(임베딩 부분). BM25·리랭커 부분은 쿼터 0.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class HybridRetriever:
    def __init__(self, bm25=None, dense=None, rrf_k: int = 60, pool: int = 20,
                 reranker=None, rerank_pool: int = 20) -> None:
        from rag.retrieval.bm25 import BM25Retriever
        from rag.retrieval.naive import NaiveRetriever

        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or NaiveRetriever()
        self.rrf_k = rrf_k          # RRF 상수 (관례상 60)
        self.pool = pool            # 각 검색기에서 가져올 후보 수
        self.reranker = reranker    # None이면 RRF 결과 그대로
        self.rerank_pool = rerank_pool  # 리랭커에 넘길 RRF 상위 후보 수

    def retrieve(self, query: str, top_k: int = 5, manual_ids=None) -> list[RetrievedChunk]:
        # 리랭커가 있으면 더 넓게(rerank_pool) 후보를 만들어 재정렬한다
        want = self.rerank_pool if self.reranker else top_k
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
        top = sorted(rrf, key=rrf.get, reverse=True)[:want]
        candidates = [
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
        if self.reranker:
            return self.reranker.rerank(query, candidates, top_n=top_k)
        return candidates
