"""질의 라우터.

POST /ask     : 텍스트 질문(+선택 매뉴얼) -> 스코핑된 Hybrid RAG -> 근거 기반 답변 + 출처.
GET  /manuals : 제품 선택 UI용 매뉴얼 목록.
STT/TTS는 브라우저(Web Speech API)가 담당하므로 서버는 텍스트만 주고받는다.
"""
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
