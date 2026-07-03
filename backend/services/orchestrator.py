"""실시간 추론 오케스트레이터.

흐름: 텍스트 질문(STT는 브라우저 Web Speech가 처리) -> 제품 스코핑 -> Hybrid RAG -> 답변+출처.
TTS도 브라우저(speechSynthesis)가 처리하므로 서버는 텍스트만 다룬다.

무거운 리소스(BM25 인덱스·Chroma·Gemini 클라이언트)는 첫 요청 때 1회만 로드한다.
"""
from __future__ import annotations


class Orchestrator:
    def __init__(self) -> None:
        self._pipeline = None  # 지연 초기화 (서버 기동을 빠르게)

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from rag.pipeline import RagPipeline
        from rag.retrieval.agentic import AgenticRetriever

        # Agentic: 칩 선택 시 그 매뉴얼로, 미선택 시 질문에서 제품 자동 식별해 스코핑
        self._retriever = AgenticRetriever()
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
