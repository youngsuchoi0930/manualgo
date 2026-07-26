"""실시간 추론 오케스트레이터.

흐름: 텍스트 질문(STT는 브라우저 Web Speech가 처리) -> 제품 스코핑 -> Hybrid RAG
      -> (선택) 크로스인코더 재정렬 -> 답변+출처.
TTS도 브라우저(speechSynthesis)가 처리하므로 서버는 텍스트만 다룬다.

무거운 리소스(BM25 인덱스·Chroma·리랭커·Gemini 클라이언트)는 첫 요청 때 1회만 로드한다.

리랭커(RERANK=1, 기본 켜짐)
  n=640 평가에서 R@1을 유의하게 끌어올린다(매뉴얼 스코프 0.744→0.820, McNemar p<0.0001).
  후보 수 기본 10 — pool 20과 R@1 차이가 유의하지 않은데(p=0.25) 재정렬 지연은 절반이라
  라이브에선 10을 쓴다(재정렬 772ms · 검색 총 907ms, 생성까지 합쳐 ~2.4s).
  GPU가 없으면 재정렬이 8초를 넘어 음성 UX가 불가하니 RERANK=0으로 끄는 것이 낫다.
"""
from __future__ import annotations

import os


class Orchestrator:
    def __init__(self) -> None:
        self._pipeline = None  # 지연 초기화 (서버 기동을 빠르게)

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from rag.pipeline import RagPipeline
        from rag.retrieval.agentic import AgenticRetriever
        from rag.retrieval.hybrid import HybridRetriever

        reranker = None
        if os.environ.get("RERANK", "1") == "1":
            try:
                from rag.retrieval.reranker import Reranker

                reranker = Reranker()
                print(f"[orchestrator] 리랭커 사용 (provider={reranker.provider}, "
                      f"weights={reranker.weights})", flush=True)
            except Exception as e:  # 모델 다운로드 실패 등 — 검색 자체는 계속 되게 한다
                print(f"[orchestrator] 리랭커 비활성화: {e}", flush=True)

        pool = int(os.environ.get("RERANK_POOL") or 10)
        hybrid = HybridRetriever(reranker=reranker, rerank_pool=pool)
        # Agentic: 칩 선택 시 그 매뉴얼로, 미선택 시 질문에서 제품 자동 식별해 스코핑
        self._retriever = AgenticRetriever(hybrid=hybrid)
        self._pipeline = RagPipeline(self._retriever)

    def list_manuals(self) -> list[str]:
        """인덱스에 있는 매뉴얼 id 목록 (제품 선택 UI용)."""
        self._ensure_loaded()
        metas = self._retriever.hybrid.bm25.metas
        return sorted({(m or {}).get("manual_id") for m in metas if m and m.get("manual_id")})

    def handle(self, *, text: str, manual_ids: list[str] | None = None) -> dict:
        """텍스트 질의를 받아 답변·출처를 반환한다.

        manual_ids가 있으면 그 매뉴얼(들)로 스코핑 — 모델 1개 또는 카테고리 전체.
        """
        self._ensure_loaded()
        return self._pipeline.run(text, manual_ids=manual_ids or None)
