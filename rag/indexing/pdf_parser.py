"""PDF 파싱 — 매뉴얼 PDF에서 페이지별 텍스트를 추출한다.

이 프로젝트가 다루는 제조사 매뉴얼은 레거시 한국어 인코딩(KSCpc-EUC-UCS2C 등)을 써서
글꼴에 유니코드 매핑이 없다. 그래서 일반 텍스트 추출은 깨진 글자만 나온다.
→ 페이지를 이미지로 렌더한 뒤 한국어 OCR로 텍스트를 얻는다.

OCR 엔진은 ``OcrEngine`` 프로토콜로 추상화되어 교체 가능하다.
기본 구현은 EasyOCR(torch/CUDA). 나중에 PaddleOCR·비전 LLM 등으로 갈아끼울 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import fitz  # PyMuPDF


@dataclass
class ParsedPage:
    manual_id: str
    page: int            # 1-based 페이지 번호 (평가의 '정답 페이지' 기준)
    text: str
    section: str | None = None


class OcrEngine(Protocol):
    def image_to_text(self, png_bytes: bytes) -> str:
        """페이지 PNG 바이트를 받아 텍스트를 반환한다."""
        ...


class EasyOcrEngine:
    """EasyOCR 기반 한국어 OCR. torch/CUDA를 사용한다.

    첫 실행 시 한국어 모델(~100MB)을 자동 다운로드한다.
    """

    def __init__(self, langs: tuple[str, ...] = ("ko", "en"), gpu: bool = True) -> None:
        import easyocr  # 지연 임포트: 무거운 의존성을 모듈 import 시점에 강제하지 않음

        self._reader = easyocr.Reader(list(langs), gpu=gpu)

    def image_to_text(self, png_bytes: bytes) -> str:
        import io

        import numpy as np
        from PIL import Image

        img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        # detail=0 → 텍스트만 반환, paragraph=True → 인접한 줄을 문단으로 묶음
        lines = self._reader.readtext(img, detail=0, paragraph=True)
        return "\n".join(lines)


def render_page_png(page: "fitz.Page", dpi: int = 200) -> bytes:
    """PDF 페이지를 지정 DPI의 PNG 바이트로 렌더한다."""
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    return page.get_pixmap(matrix=matrix).tobytes("png")


def parse_pdf(
    path: str,
    manual_id: str,
    ocr: OcrEngine | None = None,
    dpi: int = 200,
) -> list[ParsedPage]:
    """PDF를 페이지 단위로 OCR해 ``ParsedPage`` 리스트를 반환한다."""
    ocr = ocr or EasyOcrEngine()
    doc = fitz.open(path)
    pages: list[ParsedPage] = []
    for i, page in enumerate(doc):
        png = render_page_png(page, dpi=dpi)
        text = ocr.image_to_text(png).strip()
        pages.append(ParsedPage(manual_id=manual_id, page=i + 1, text=text))
    return pages
