"""청킹 — 파싱된 페이지를 검색 단위 청크로 분할하고 페이지·섹션 메타데이터를 부착한다.

매뉴얼은 평가가 '정답 페이지' 단위라 페이지 추적이 중요하다. 그래서 페이지 경계를
넘지 않고, 페이지 안에서만 길이 기준으로 나눈다(대부분은 페이지당 1청크).
OCR 텍스트라 토크나이저 의존을 피하고 문자 길이 + 줄/공백 경계로 자른다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    manual_id: str
    page: int
    section: str | None
    text: str


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    """길이 기준으로 자르되, 경계 부근의 줄바꿈/공백에서 끊는다."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text[start:end]
            brk = max(window.rfind("\n"), window.rfind(" "))
            if brk > max_chars * 0.5:  # 경계 후반부에서만 끊기
                end = start + brk
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_pages(pages: list, *, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    """ParsedPage 리스트를 Chunk 리스트로 변환한다 (페이지 메타데이터 보존)."""
    chunks: list[Chunk] = []
    for p in pages:
        text = (p.text or "").strip()
        if not text:
            continue
        for idx, piece in enumerate(_split(text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{p.manual_id}_p{p.page}_c{idx}",
                    manual_id=p.manual_id,
                    page=p.page,
                    section=p.section,
                    text=piece,
                )
            )
    return chunks
