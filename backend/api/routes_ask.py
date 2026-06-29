"""음성/텍스트 질의 라우터.

POST /ask : 음성(또는 텍스트) 질문 -> STT -> RAG -> LLM 답변 -> TTS -> 응답.
답변에는 출처 매뉴얼(페이지·섹션)이 함께 포함된다.
"""
from fastapi import APIRouter, UploadFile

router = APIRouter(tags=["ask"])


@router.post("/ask")
async def ask(audio: UploadFile | None = None, text: str | None = None) -> dict:
    """음성 파일 또는 텍스트 질문을 받아 근거 기반 답변과 출처를 반환한다."""
    # TODO: backend.services.orchestrator 로 위임
    raise NotImplementedError
