"""PDF 파싱 — 매뉴얼 PDF에서 페이지별 텍스트와 메타데이터를 추출한다.

오프라인 인덱싱 파이프라인의 1단계. PyMuPDF(fitz) 기반.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedPage:
    manual_id: str
    page: int
    text: str
    section: str | None = None


def parse_pdf(path: str, manual_id: str) -> list[ParsedPage]:
    """PDF를 페이지 단위로 파싱한다."""
    raise NotImplementedError
