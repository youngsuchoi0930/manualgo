"""온라인 추론 파이프라인 — 질문 텍스트를 받아 근거 기반 답변과 출처를 반환한다.

질문 임베딩 -> Top-k 검색·재순위 -> 컨텍스트 구성 -> 답변 생성.
검색 단계(retriever)를 교체해 Naive/Hybrid/Agentic 3단계를 동일 인터페이스로 비교한다.

응답에 단계별 소요(ms)를 timings로 담는다. 음성 UX는 총 3초가 목표인데 어느 단계가
시간을 먹는지 모르면 엉뚱한 곳을 최적화하게 된다(실측: 검색 ~0.9s vs 생성 ~5.5s).
"""
from __future__ import annotations

import time

from rag.generation.generator import AnswerGenerator
from rag.retrieval.base import Retriever


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator | None = None) -> None:
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()

    def run(self, question: str, top_k: int = 5, manual_ids: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        contexts = self.retriever.retrieve(question, top_k=top_k, manual_ids=manual_ids)
        t1 = time.perf_counter()
        out = self.generator.generate(question, contexts)
        t2 = time.perf_counter()
        if isinstance(out, dict):
            out["timings"] = {
                "retrieval_ms": round((t1 - t0) * 1000),
                "generation_ms": round((t2 - t1) * 1000),
                "total_ms": round((t2 - t0) * 1000),
            }
        return out
