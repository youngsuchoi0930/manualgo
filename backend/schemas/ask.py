"""질의/응답 Pydantic 스키마."""
from pydantic import BaseModel


class AskRequest(BaseModel):
    text: str                              # 질문 텍스트 (STT는 브라우저에서 처리)
    manual_ids: list[str] | None = None    # 스코핑: 모델 1개 or 카테고리(여러 개), 없으면 전체


class TtsRequest(BaseModel):
    text: str                              # 합성할 답변 텍스트


class Source(BaseModel):
    manual_id: str
    page: int
    section: str | None = None
    score: float


class AskResponse(BaseModel):
    question: str                  # 질문 텍스트
    headline: str | None = None    # 핵심 답 한 구절 (예: "최대 24시간")
    answer: str                    # 부연 설명 (근거 기반)
    sources: list[Source]          # 출처 매뉴얼 위치
