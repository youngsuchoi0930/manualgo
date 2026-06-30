"""PDF 파싱 — 매뉴얼 PDF에서 페이지별 텍스트를 추출한다.

이 프로젝트가 다루는 제조사 매뉴얼은 레거시 한국어 인코딩(KSCpc-EUC-UCS2C 등)을 써서
글꼴에 유니코드 매핑이 없다. 그래서 일반 텍스트 추출은 깨진 글자만 나온다.
→ 페이지를 이미지로 렌더한 뒤 한국어 OCR로 텍스트를 얻는다.

OCR 엔진은 ``OcrEngine`` 프로토콜로 추상화되어 교체 가능하다.
기본 구현은 EasyOCR(torch/CUDA). 나중에 PaddleOCR·비전 LLM 등으로 갈아끼울 수 있다.
"""
from __future__ import annotations

import os
import time
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


class GeminiOcrEngine:
    """Gemini 멀티모달로 페이지 이미지를 OCR한다 (torch 불필요, 한국어 품질 우수).

    GOOGLE_API_KEY 환경변수에서 키를 읽는다. 무료 티어 rate limit을 고려해 실패 시 backoff 재시도.
    """

    PROMPT = (
        "이 가전제품 매뉴얼 페이지 이미지의 모든 텍스트를 위에서 아래로 순서대로 원문 그대로 "
        "한국어로 추출하라. 표는 행 단위로 읽기 쉽게 풀어쓰고, 숫자·기호·에러코드·온도(°C)는 "
        "정확히 옮겨라. 설명·요약·머리말 없이 페이지에 적힌 텍스트만 출력하라."
    )

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from google import genai

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다 (.env 확인).")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=api_key)
        self._last_call = 0.0  # RPM 보호용 throttle

    MIN_INTERVAL = 7.0  # 무료 티어 ~10 RPM 대비 호출 간 최소 간격(초)

    def image_to_text(self, png_bytes: bytes) -> str:
        from google.genai import types

        part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
        last_err: Exception | None = None
        for attempt in range(6):
            wait = self.MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=[part, self.PROMPT],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                self._last_call = time.monotonic()
                return (resp.text or "").strip()
            except Exception as e:
                self._last_call = time.monotonic()
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(30)  # 분당 한도 리셋 대기
                elif "503" in msg or "UNAVAILABLE" in msg or "overload" in msg:
                    time.sleep(min(60, 5 * (2**attempt)))  # 과부하 → 지수 백오프
                else:
                    break  # 키/모델명 등 비일시적 오류 → 즉시 중단
        raise RuntimeError(f"Gemini OCR 실패: {last_err}")


class TesseractOcrEngine:
    """로컬 Tesseract OCR (torch·API·rate limit 없음). 한국어+영어.

    사전 설치:
      1) Tesseract 바이너리 (Windows: UB Mannheim 빌드, 설치 시 'Korean' 언어 데이터 포함)
      2) pip install pytesseract
    tesseract.exe가 PATH에 없으면 .env에 TESSERACT_CMD로 경로를 지정한다.
    """

    def __init__(self, lang: str = "kor+eng", cmd: str | None = None) -> None:
        import pytesseract

        cmd = cmd or os.environ.get("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self._pt = pytesseract
        self.lang = lang

    def image_to_text(self, png_bytes: bytes) -> str:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        return self._pt.image_to_string(img, lang=self.lang).strip()


def render_page_png(page: "fitz.Page", dpi: int = 200) -> bytes:
    """PDF 페이지를 지정 DPI의 PNG 바이트로 렌더한다."""
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    return page.get_pixmap(matrix=matrix).tobytes("png")


# 정상 한국어 본문에 흔한 토큰들 — 텍스트 레이어가 살아있는지 판정에 사용
_KR_MARKERS = ("니다", "사용", "주의", "하세요", "설명", "전원", "버튼", "기능", "세요", "있")


def _kr_score(text: str) -> int:
    return sum(text.count(m) for m in _KR_MARKERS)


def _doc_text_usable(doc: "fitz.Document", sample_pages: int = 6, threshold: int = 10) -> bool:
    """앞 몇 쪽을 추출해 '정상 텍스트 레이어'인지 판정. 깨진 인코딩이면 단어가 거의 안 잡힌다."""
    n = min(sample_pages, doc.page_count)
    text = "".join(doc[i].get_text("text") for i in range(n))
    return len(text) >= 50 and _kr_score(text) >= threshold


def parse_pdf(
    path: str,
    manual_id: str,
    ocr: OcrEngine | None = None,
    dpi: int = 300,
    skip_errors: bool = False,
    min_page_chars: int = 30,
) -> list[ParsedPage]:
    """PDF를 페이지 단위로 파싱한다 — **텍스트 추출 우선, 깨졌거나 이미지면 OCR 폴백**.

    - 텍스트 레이어가 정상인 PDF: pymupdf로 직접 추출(빠르고 정확, OCR 노이즈 없음)
    - 레이어가 깨진(레거시 인코딩) PDF: 문서 전체를 OCR
    - 정상 PDF 안의 이미지-only 쪽: 그 쪽만 OCR 폴백
    skip_errors=True면 OCR 폴백 실패 시 빈 텍스트로 두고 계속한다.
    """
    doc = fitz.open(path)
    use_text = _doc_text_usable(doc)
    print(f"        ({'텍스트 레이어 추출' if use_text else '깨진 레이어 → OCR'})", flush=True)

    ocr_engine: OcrEngine | None = ocr  # 필요할 때만 생성
    pages: list[ParsedPage] = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip() if use_text else ""
        if len(text) < min_page_chars:  # 깨진 문서 또는 텍스트 문서 내 이미지 쪽 → OCR 폴백
            try:
                if ocr_engine is None:
                    ocr_engine = TesseractOcrEngine()
                ocr_text = ocr_engine.image_to_text(render_page_png(page, dpi=dpi)).strip()
                if len(ocr_text) > len(text):
                    text = ocr_text
            except Exception as e:
                if not skip_errors:
                    raise
                print(f"  [!] {i + 1}쪽 OCR 폴백 실패 → 건너뜀: {e}", flush=True)
        pages.append(ParsedPage(manual_id=manual_id, page=i + 1, text=text))
    return pages
