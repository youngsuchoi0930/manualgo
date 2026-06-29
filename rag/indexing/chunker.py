"""청킹 — 파싱된 페이지를 검색 단위 청크로 분할하고 페이지·섹션 메타데이터를 부착한다."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    manual_id: str
    page: int
    section: str | None
    text: str


def chunk_pages(pages: list, *, max_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    """페이지 리스트를 토큰 기준으로 청킹한다 (정답 페이지 추적용 메타데이터 유지)."""
    raise NotImplementedError
