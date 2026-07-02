"""질의/응답 Pydantic 스키마."""
from pydantic import BaseModel


class AskRequest(BaseModel):
    text: str                              # 질문 텍스트 (STT는 브라우저에서 처리)
    manual_ids: list[str] | None = None    # 스코핑: 모델 1개 or 카테고리(여러 개), 없으면 전체


class Source(BaseModel):
    manual_id: str
    page: int
    section: str | None = None
    score: float


class AskResponse(BaseModel):
    question: str           # 질문 텍스트
    answer: str             # 근거 기반 생성 답변
    sources: list[Source]   # 출처 매뉴얼 위치
