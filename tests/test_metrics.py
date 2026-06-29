"""평가 지표 단위 테스트."""
from evaluation.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_hit():
    assert recall_at_k([3, 1, 7], gold_page=1, k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k([3, 1, 7], gold_page=9, k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([3, 1, 7], gold_page=1) == 0.5
    assert reciprocal_rank([3, 1, 7], gold_page=9) == 0.0
