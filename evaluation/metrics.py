"""검색 정확도 지표 — Recall@k, MRR."""
from __future__ import annotations


def recall_at_k(predicted_pages: list[int], gold_page: int, k: int) -> float:
    """정답 페이지가 상위 k개 안에 있으면 1.0, 아니면 0.0."""
    return 1.0 if gold_page in predicted_pages[:k] else 0.0


def reciprocal_rank(predicted_pages: list[int], gold_page: int) -> float:
    """정답이 r번째면 1/r, 없으면 0."""
    for i, p in enumerate(predicted_pages, start=1):
        if p == gold_page:
            return 1.0 / i
    return 0.0
