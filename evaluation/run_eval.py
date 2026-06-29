"""단계별 성능 비교 — Naive -> Hybrid+Reranker -> Agentic 의 Recall@k·MRR를 측정·비교한다.

평가셋(data/eval)으로 각 검색기를 돌려 지표를 집계하고 결과를 evaluation/results 에 저장한다.

실행: ``python -m evaluation.run_eval``
"""
from __future__ import annotations


def run_eval(evalset_path: str = "data/eval", results_dir: str = "evaluation/results") -> None:
    # TODO: 각 단계 retriever 로 평가셋 실행 -> metrics 집계 -> 표/그래프 저장
    raise NotImplementedError


if __name__ == "__main__":
    run_eval()
