"""오프라인 인덱싱 엔트리포인트 (사전 1회 실행).

매뉴얼 PDF -> 파싱 -> 청킹 -> 임베딩 -> 벡터 DB 저장 + BM25 인덱스 구축.

실행: ``python -m rag.indexing.build_index``
"""
from __future__ import annotations


def build_index(manuals_dir: str = "data/raw/manuals", index_dir: str = "data/index") -> None:
    """디렉토리의 모든 매뉴얼 PDF를 인덱싱한다."""
    raise NotImplementedError


if __name__ == "__main__":
    build_index()
