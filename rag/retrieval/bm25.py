"""BM25 키워드 검색기 (torch·임베딩·API 불필요 → 쿼터 0).

Chroma에 저장된 청크 텍스트로 BM25 인덱스를 만든다. 추출 텍스트가 띄어쓰기 없는
한국어라, 토큰화는 (1) 영숫자 런(에러코드·숫자: UE, 40, E1) + (2) 한글 문자 바이그램으로 한다.
에러코드·사양·수치형처럼 '정확 토큰'이 중요한 질문에 강하다.
"""
from __future__ import annotations

import re

from rag.retrieval.base import RetrievedChunk

_ALNUM = re.compile(r"[a-z0-9]+")
_HANGUL = re.compile(r"[가-힣]+")


def tokenize_ko(text: str) -> list[str]:
    text = (text or "").lower()
    tokens: list[str] = _ALNUM.findall(text)  # 에러코드/숫자/영문 = 통째 토큰
    for run in _HANGUL.findall(text):          # 한글 = 바이그램(띄어쓰기 없어도 매칭)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Retriever:
    def __init__(self, index_dir: str = "data/index") -> None:
        import chromadb
        from rank_bm25 import BM25Okapi

        from rag.vectorstore.store import COLLECTION

        client = chromadb.PersistentClient(path=index_dir)
        col = client.get_or_create_collection(COLLECTION)
        data = col.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict] = data["metadatas"]
        self._bm25 = BM25Okapi([tokenize_ko(d) for d in self.docs])

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(tokenize_ko(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for i in order:
            md = self.metas[i] or {}
            out.append(
                RetrievedChunk(
                    chunk_id=self.ids[i],
                    manual_id=md.get("manual_id"),
                    page=md.get("page"),
                    section=md.get("section") or None,
                    text=self.docs[i],
                    score=float(scores[i]),
                )
            )
        return out
