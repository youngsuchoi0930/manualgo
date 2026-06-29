"""평가셋 자동 생성 — 매뉴얼 본문으로 LLM이 "질문 <-> 정답 페이지/섹션" 쌍을 만든다.

질문 유형을 의도적으로 분산한다: 에러코드형 · 사용법형 · 사양·수치형 · 문제해결형.
생성 결과는 ``data/eval/`` 에 저장하고 일부 수기 검수로 보정한다.

실행: ``python -m evaluation.generate_evalset``
"""
from __future__ import annotations

QUESTION_TYPES = ["error_code", "how_to", "spec_numeric", "troubleshooting"]


def generate_evalset(chunks_dir: str = "data/processed/chunks", out_dir: str = "data/eval") -> None:
    raise NotImplementedError


if __name__ == "__main__":
    generate_evalset()
