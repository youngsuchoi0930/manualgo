"""질의/응답 Pydantic 스키마."""
from pydantic import BaseModel


class Source(BaseModel):
    manual_id: str
    page: int
    section: str | None = None
    score: float


class AskResponse(BaseModel):
    question: str           # STT로 인식된 질문 텍스트
    answer: str             # 근거 기반 생성 답변
    sources: list[Source]   # 출처 매뉴얼 위치
    audio_url: str | None = None   # TTS 결과 음성
