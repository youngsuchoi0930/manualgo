"""실패 클러스터링 (차별 포인트) — 시스템이 엉뚱한 페이지를 찾는 질문을 유형별로 분류한다.

질문유형(에러코드/사용법/사양·수치/문제해결)별 정확도를 비교해
'어떤 유형에서 정확도가 급락하는지' 같은 약점을 정량적으로 드러낸다.

실행: ``python -m evaluation.failure_clustering``
"""
from __future__ import annotations


def cluster_failures(results_dir: str = "evaluation/results") -> None:
    # TODO: 실패 케이스 임베딩 -> KMeans/유형별 집계 -> 리포트 생성
    raise NotImplementedError


if __name__ == "__main__":
    cluster_failures()
