"""근거 기반 답변 생성 — 검색 청크 + 질문으로 프롬프트를 구성해 Gemini 답변을 생성한다.

답변은 검색된 '근거'에만 기반하도록 강제하고, 출처(매뉴얼·페이지)를 함께 반환해
환각 여부를 사용자가 검증할 수 있게 한다. API 키는 GOOGLE_API_KEY 환경변수에서 읽는다.
"""
from __future__ import annotations

import os

from rag.retrieval.base import RetrievedChunk

SYSTEM_INSTRUCTION = (
    "당신은 가전제품 매뉴얼 음성 도우미입니다. 아래 '근거'에 실제로 적힌 내용만 사용해 "
    "한국어로 짧고 명확하게 답하세요. 근거에 없는 내용은 지어내지 말고 headline을 '매뉴얼에서 찾지 못했어요'로 하세요. "
    "출력은 JSON 하나: {\"headline\": \"핵심 답 한 구절(15자 이내, 예: '최대 24시간')\", "
    "\"answer\": \"부연 설명 1~3문장. 끝에 (출처: N쪽) 표기\"} "
    "참고: 근거는 OCR로 추출되어 오타가 있을 수 있으니 문맥으로 보정해 이해하세요."
)


def _build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    blocks = [f"[출처 {c.manual_id} {c.page}쪽]\n{c.text}" for c in contexts]
    context_text = "\n\n".join(blocks) if blocks else "(검색된 근거 없음)"
    return f"근거:\n{context_text}\n\n질문: {question}\n\n답변:"


class AnswerGenerator:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from google import genai

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다 (.env를 확인하세요).")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=api_key)

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> dict:
        """답변(headline=핵심 답 한 구절 + answer=설명)과 출처 목록을 반환한다."""
        import json

        from google.genai import types

        prompt = _build_prompt(question, contexts)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        headline, answer = None, (resp.text or "").strip()
        try:  # JSON 실패 시 원문을 answer로 폴백
            data = json.loads(resp.text)
            headline = (data.get("headline") or "").strip() or None
            answer = (data.get("answer") or "").strip() or answer
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return {
            "question": question,
            "headline": headline,
            "answer": answer,
            "sources": [
                {
                    "manual_id": c.manual_id,
                    "page": c.page,
                    "section": c.section,
                    "score": c.score,
                }
                for c in contexts
            ],
        }
