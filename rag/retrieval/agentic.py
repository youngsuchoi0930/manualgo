"""3단계 Agentic — 질문에서 제품을 자동 식별해 스코핑 후 Hybrid 검색, 저신뢰 시 재질의.

사용자가 매뉴얼을 고르지 않아도(글로벌) 질문 텍스트에서 카테고리를 추정해 검색을 좁힌다.
- 제품 식별: 한국어 키워드 기반 분류 (torch·API 불필요 → 쿼터 0)
- 스코핑: 식별된 카테고리의 매뉴얼로 Hybrid 검색
- 재질의: 스코핑 결과가 빈약하면 글로벌로 broaden
호출자가 manual_ids를 지정하면(프론트 칩 선택) 그것을 존중한다.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk

# 질문 텍스트 → 카테고리 키워드
QUESTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "washer": ("세탁", "빨래", "헹굼", "탈수", "통돌이", "드럼", "세제", "섬유유연제"),
    "dishwasher": ("식기", "그릇", "식기세척", "노즐"),
    "microwave": ("전자레인지", "렌지", "데우", "해동", "조리", "오븐", "찌개"),
    "purifier": ("정수기", "정수", "필터", "냉수", "온수", "물맛"),
    "vacuum": ("청소기", "흡입", "먼지통", "먼지"),
    "fridge": ("냉장고", "냉동", "냉장", "성에", "탈취", "제빙"),
    "dehumidifier": ("제습", "습도", "실외기", "응축수"),
}
# 파일명 접두어가 washer지만 실제로는 식기세척기인 매뉴얼
_DISHWASHER_IDS = {"lg-washer-d1220mf", "lg-washer-mfl47377718"}


def manual_category(mid: str) -> str:
    if mid in _DISHWASHER_IDS:
        return "dishwasher"
    m = (mid or "").lower()
    if "washer" in m or "sew" in m:
        return "washer"
    if "microwave" in m:
        return "microwave"
    if "waterpurifier" in m:
        return "purifier"
    if "vacuum" in m:
        return "vacuum"
    if "fridge" in m:
        return "fridge"
    if "dehumidifier" in m:
        return "dehumidifier"
    return "etc"


def classify_question(query: str) -> str | None:
    """질문 텍스트에서 매칭 키워드가 가장 많은 카테고리를 반환. 매칭 없으면 None.

    각 카테고리의 첫 키워드(제품명: 청소기·정수기 등)에 가중치를 둬, 부수 키워드(필터 등)
    공유로 인한 오분류를 줄인다.
    """
    q = query or ""
    scores = {}
    for cat, kws in QUESTION_KEYWORDS.items():
        s = sum((2 if i == 0 else 1) * q.count(kw) for i, kw in enumerate(kws))
        if s > 0:
            scores[cat] = s
    return max(scores, key=scores.get) if scores else None


class AgenticRetriever:
    def __init__(self, hybrid=None) -> None:
        from rag.retrieval.hybrid import HybridRetriever

        self.hybrid = hybrid or HybridRetriever()
        by_cat: dict[str, set[str]] = {}
        for m in self.hybrid.bm25.metas:
            mid = (m or {}).get("manual_id")
            if mid:
                by_cat.setdefault(manual_category(mid), set()).add(mid)
        self.by_cat = {c: sorted(ids) for c, ids in by_cat.items()}

    def classify(self, query: str) -> str | None:
        cat = classify_question(query)
        return cat if cat in self.by_cat else None

    def retrieve(self, query: str, top_k: int = 5, manual_ids=None) -> list[RetrievedChunk]:
        scope = manual_ids  # 호출자가 지정한 스코프(칩 선택)를 우선
        if scope is None:
            cat = self.classify(query)
            if cat:
                scope = self.by_cat[cat]
        results = self.hybrid.retrieve(query, top_k=top_k, manual_ids=scope)
        if scope is not None and len(results) < top_k:  # 재질의: 빈약하면 글로벌로 broaden
            results = self.hybrid.retrieve(query, top_k=top_k, manual_ids=None)
        return results
