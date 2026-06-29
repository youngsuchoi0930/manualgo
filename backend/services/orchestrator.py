"""실시간 추론 오케스트레이터.

흐름: STT(음성->텍스트) -> 제품·기종 식별 -> RAG 검색·답변 생성 -> TTS(텍스트->음성).
``rag.pipeline`` 과 ``speech`` 모듈을 조립한다.
"""
from __future__ import annotations


class Orchestrator:
    def __init__(self) -> None:
        # TODO: STT, RAG pipeline, TTS 의존성 주입
        ...

    async def handle(self, *, audio: bytes | None = None, text: str | None = None) -> dict:
        """음성/텍스트 질의를 받아 답변·출처·음성 응답을 조립해 반환한다."""
        # 1) audio가 있으면 STT로 텍스트화
        # 2) 제품/기종 식별 -> 검색 범위 결정
        # 3) RAG 파이프라인 실행 -> 답변 + 출처
        # 4) TTS로 음성 합성
        raise NotImplementedError
