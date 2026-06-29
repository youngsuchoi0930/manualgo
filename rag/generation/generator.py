"""근거 기반 답변 생성 — 검색 청크 + 질문으로 프롬프트를 구성해 LLM 답변을 생성한다.

답변에는 반드시 출처(매뉴얼·페이지·섹션)를 함께 명시해 환각을 검증 가능하게 한다.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk


class AnswerGenerator:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini") -> None:
        self.provider = provider
        self.model = model

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> dict:
        """답변 텍스트와 출처 목록을 반환한다."""
        raise NotImplementedError
