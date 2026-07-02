"""온라인 추론 파이프라인 — 질문 텍스트를 받아 근거 기반 답변과 출처를 반환한다.

질문 임베딩 -> Top-k 검색·재순위 -> 컨텍스트 구성 -> 답변 생성.
검색 단계(retriever)를 교체해 Naive/Hybrid/Agentic 3단계를 동일 인터페이스로 비교한다.
"""
from __future__ import annotations

from rag.generation.generator import AnswerGenerator
from rag.retrieval.base import Retriever


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator | None = None) -> None:
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()

    def run(self, question: str, top_k: int = 5, manual_ids: list[str] | None = None) -> dict:
        contexts = self.retriever.retrieve(question, top_k=top_k, manual_ids=manual_ids)
        return self.generator.generate(question, contexts)
