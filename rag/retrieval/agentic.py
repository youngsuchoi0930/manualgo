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
    "washer": ("세탁", "빨래", "헹굼", "탈수", "통돌이", "세제", "섬유유연제"),
    "dryer": ("건조기", "건조 코스", "먼지 필터", "응축", "터보건조"),
    "dishwasher": ("식기", "그릇", "식기세척", "예비세척", "린스"),
    "styler": ("의류관리기", "스타일러", "구김", "살균 코스", "옷걸이", "무빙행어"),
    "fridge": ("냉장고", "냉동", "냉장", "성에", "탈취", "제빙", "김치냉장고", "야채실"),
    "microwave": ("전자레인지", "렌지", "데우", "해동", "찌개", "광파"),
    "oven": ("인덕션", "전기레인지", "하이라이트", "화구", "쿡탑", "오븐"),
    "purifier": ("정수기", "정수", "냉수", "온수", "물맛", "코크", "얼음"),
    "airpurifier": ("공기청정기", "공기청정", "미세먼지", "청정도", "탈취필터"),
    "humidifier": ("가습기", "가습", "수조", "분무"),
    "dehumidifier": ("제습기", "제습", "습도", "응축수", "물통"),
    "aircon": ("에어컨", "냉방", "실외기", "운전 선택", "정전보상", "제상", "열대야"),
    "vacuum": ("청소기", "흡입", "먼지통", "브러시", "헤파", "침구"),
    "tv": ("티비", "티브이", "텔레비전", "화면", "채널", "리모컨", "방송", "HDMI", "볼륨"),
    "audio": ("사운드바", "홈시어터", "스피커", "블루레이", "우퍼", "프로젝터", "빔"),
    "massagechair": ("안마의자", "안마", "마사지", "등받이", "다리부"),
    "bidet": ("비데", "노즐 세정", "세정", "온수 세정", "변좌"),
    "ricecooker": ("밥솥", "취사", "보온", "내솥", "압력", "만능찜"),
}

# 파일명 접두어가 실제 제품과 다른 매뉴얼 (내용 확인으로 밝혀진 예외)
_ID_OVERRIDES = {
    # MFL473777xx 계열은 LG 식기세척기 문서군인데 파일명이 washer로 붙었다
    "lg-washer-d1220mf": "dishwasher",
    "lg-washer-mfl47377718": "dishwasher",
    # 파일명은 dehumidifier지만 본문 2쪽이 "사용설명서 에어컨 / 벽걸이형"
    "lg-dehumidifier-snc063": "aircon",
}

# manual_id에 포함된 문자열 → 카테고리. **순서가 중요**하다:
# airpurifier가 purifier보다, kimchi가 fridge보다, waterpurifier가 purifier보다 먼저 와야 한다.
_ID_PATTERNS: tuple[tuple[str, str], ...] = (
    ("airpurifier", "airpurifier"),
    ("waterpurifier", "purifier"),
    ("kimchifridge", "fridge"),
    ("dishwasher", "dishwasher"),
    ("ricecooker", "ricecooker"),
    ("massagechair", "massagechair"),
    ("dehumidifier", "dehumidifier"),
    ("humidifier", "humidifier"),
    ("hometheater", "audio"),
    ("soundbar", "audio"),
    ("projector", "audio"),
    ("induction", "oven"),
    ("cooktop", "oven"),
    ("microwave", "microwave"),
    ("styler", "styler"),
    ("dryer", "dryer"),
    ("bidet", "bidet"),
    ("vacuum", "vacuum"),
    ("fridge", "fridge"),
    ("aircon", "aircon"),
    ("purifier", "purifier"),
    ("washer", "washer"),
    ("oven", "oven"),
    ("tv", "tv"),
    ("sew", "washer"),
)


def manual_category(mid: str) -> str:
    """manual_id에서 제품 카테고리를 판정한다 (예외 → 패턴 순서대로)."""
    if mid in _ID_OVERRIDES:
        return _ID_OVERRIDES[mid]
    m = (mid or "").lower()
    for pat, cat in _ID_PATTERNS:
        if pat in m:
            return cat
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
