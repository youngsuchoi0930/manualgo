"""질의 라우터.

POST /ask     : 텍스트 질문(+선택 매뉴얼) -> 스코핑된 Hybrid RAG -> 근거 기반 답변 + 출처.
GET  /manuals : 제품 선택 UI용 매뉴얼 목록.
STT/TTS는 브라우저(Web Speech API)가 담당하므로 서버는 텍스트만 주고받는다.
"""
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.schemas.ask import AskRequest, AskResponse, TtsRequest
from backend.services.orchestrator import Orchestrator

router = APIRouter(tags=["ask"])
_orchestrator = Orchestrator()  # 프로세스당 1개 (무거운 리소스는 내부에서 지연 로드)
_tts = None  # AzureTTS 지연 싱글턴


@router.get("/manuals")
async def manuals() -> list[str]:
    return _orchestrator.list_manuals()


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="질문 텍스트가 비었습니다.")
    try:
        result = _orchestrator.handle(text=text, manual_ids=req.manual_ids)
    except Exception as e:  # Gemini 쿼터/네트워크 등 — 원인을 그대로 노출
        raise HTTPException(status_code=502, detail=f"답변 생성 실패: {e}") from e
    return AskResponse(**result)


_MANUALS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "manuals"
_page_cache: dict[tuple[str, int], bytes] = {}  # (manual_id, page) -> PNG (최근 64쪽)
_ID_RE = re.compile(r"^[a-z0-9._-]+$", re.I)


@router.get("/page/{manual_id}/{page}")
async def page_image(manual_id: str, page: int) -> Response:
    """출처 매뉴얼 페이지를 PNG로 렌더해 반환한다 — '출처 페이지 보기' 미리보기용."""
    if not _ID_RE.match(manual_id):  # 경로 조작 방지
        raise HTTPException(status_code=404, detail="잘못된 매뉴얼 id")
    key = (manual_id, page)
    if key not in _page_cache:
        pdf = _MANUALS_DIR / f"{manual_id}.pdf"
        if not pdf.exists():
            raise HTTPException(status_code=404, detail="매뉴얼 없음")
        import fitz

        doc = fitz.open(str(pdf))
        if not 1 <= page <= doc.page_count:
            raise HTTPException(status_code=404, detail=f"페이지 범위 밖 (1~{doc.page_count})")
        from rag.indexing.pdf_parser import render_page_png

        if len(_page_cache) >= 64:
            _page_cache.pop(next(iter(_page_cache)))
        _page_cache[key] = render_page_png(doc[page - 1], dpi=110)
    return Response(content=_page_cache[key], media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.post("/tts")
async def tts(req: TtsRequest) -> Response:
    """답변 텍스트를 Azure TTS로 합성해 MP3를 반환한다. 실패 시 502 → 프론트가 브라우저 TTS로 폴백."""
    global _tts
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="합성할 텍스트가 비었습니다.")
    try:
        if _tts is None:
            from speech.tts import AzureTTS

            _tts = AzureTTS()
        audio = _tts.synthesize(text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS 실패: {e}") from e
    return Response(content=audio, media_type="audio/mpeg")
