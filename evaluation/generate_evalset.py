"""평가셋 자동 생성 — 매뉴얼 본문으로 LLM(Gemini)이 "질문 ↔ (매뉴얼, 정답 페이지)" 쌍을 만든다.

질문 유형을 분산한다: 에러코드형 · 사용법형 · 사양·수치형 · 문제해결형.
이미 인덱싱된(=OCR된) 텍스트를 Chroma에서 읽으므로 재-OCR/torch 불필요.
매뉴얼이 여럿이라 페이지를 (manual_id, page)로 식별한다. 본문이 충실한 쪽을 매뉴얼별로
샘플링해 generate_content 호출을 적게(배치) 쓴다. 결과는 data/eval/evalset.jsonl.

실행: python -m evaluation.generate_evalset
"""
from __future__ import annotations

import json
import os
from pathlib import Path

QUESTION_TYPES = ["error_code", "how_to", "spec_numeric", "troubleshooting"]

PROMPT_HEADER = """다음은 여러 가전 매뉴얼의 '쪽'들이다. 각 블록 머리에 [id=<매뉴얼> page=<쪽>]이 붙어 있다.
OCR로 추출되어 오타가 있을 수 있으니 문맥으로 이해하라.

각 블록마다, 그 블록 내용만으로 답이 분명한 한국어 질문을 1개 만들어라(사용자가 음성으로 물어볼 법하게). 규칙:
- id와 page는 그 질문이 나온 블록 값을 그대로 적는다.
- type은 다음 중 하나: {types}
  (error_code=에러코드, how_to=사용법, spec_numeric=사양·수치, troubleshooting=문제해결)
- 전체적으로 유형을 골고루. 내용이 빈약해 좋은 질문이 안 나오는 블록은 건너뛴다.
- 보증·부품 보유기간·소비자 보상 같은 '법정 고지' 내용으로는 질문을 만들지 마라
  (매뉴얼마다 거의 동일해 어느 제품인지 변별이 안 됨). 제품 고유의 사용법·기능·에러·사양 위주로.
- 출력은 JSON 배열만:
[{{"id": "lg-washer-d1220mf", "page": 9, "question": "...", "type": "how_to"}}]

=== 블록들 ===
{blocks}
"""


def _load_pages(index_dir: str) -> dict[tuple[str, int], str]:
    """Chroma에서 (manual_id, page) -> 합쳐진 본문 텍스트."""
    import chromadb

    from rag.vectorstore.store import COLLECTION

    client = chromadb.PersistentClient(path=index_dir)
    col = client.get_or_create_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    pages: dict[tuple[str, int], list[str]] = {}
    for doc, md in zip(data["documents"], data["metadatas"]):
        md = md or {}
        key = (str(md.get("manual_id", "")), int(md.get("page", 0)))
        pages.setdefault(key, []).append(doc or "")
    return {k: "\n".join(t).strip() for k, t in pages.items()}


# 매뉴얼 공통 법정고지(보증/부품/보상) — 거의 동일해 변별력 낮음 → 평가셋에서 제외
_BOILERPLATE = ("보증", "보유기간", "부품 보유", "소비자", "보상", "분쟁", "무상", "유상", "품질보증", "서비스 기간")


def _is_boilerplate(text: str) -> bool:
    return sum(1 for k in _BOILERPLATE if k in text) >= 2


def _sample(pages: dict[tuple[str, int], str], per_manual: int, min_chars: int) -> list[tuple[str, int, str]]:
    """매뉴얼별로 제품 고유 페이지를 페이지 순으로 고르게 per_manual개 고른다(보일러플레이트 제외)."""
    by_manual: dict[str, list[tuple[int, str]]] = {}
    for (mid, page), text in pages.items():
        if len(text) >= min_chars and not _is_boilerplate(text):
            by_manual.setdefault(mid, []).append((page, text))
    sampled: list[tuple[str, int, str]] = []
    for mid, items in by_manual.items():
        items.sort(key=lambda x: x[0])  # 페이지 순
        n = len(items)
        if n <= per_manual:
            picks = items
        else:
            step = n / per_manual
            picks = [items[int(i * step)] for i in range(per_manual)]  # 앞·중간·뒤 고르게 분산
        for page, text in picks:
            sampled.append((mid, page, text))
    return sampled


def generate_evalset(
    index_dir: str = "data/index",
    out_path: str = "data/eval/evalset.jsonl",
    per_manual: int = 4,
    min_chars: int = 200,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    from google import genai
    from google.genai import types

    pages = _load_pages(index_dir)
    if not pages:
        print("[!] 인덱스가 비었습니다. 먼저 build_index 로 인덱싱하세요.")
        return

    sampled = _sample(pages, per_manual=per_manual, min_chars=min_chars)
    valid = {(mid, page) for mid, page, _ in sampled}
    blocks = "\n\n".join(f"[id={mid} page={page}]\n{text}" for mid, page, text in sampled)
    prompt = PROMPT_HEADER.format(types=" / ".join(QUESTION_TYPES), blocks=blocks)

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"[*] {len(valid)}개 쪽(매뉴얼 {len({m for m,_ in valid})}개)으로 평가셋 생성 중 (model={model})...", flush=True)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
    )

    try:
        items = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        print("[!] JSON 파싱 실패. 모델 원응답:\n", (resp.text or "")[:500])
        return

    clean = []
    for it in items:
        mid = str(it.get("id", ""))
        page = it.get("page")
        q = (it.get("question") or "").strip()
        typ = it.get("type")
        if q and isinstance(page, int) and (mid, page) in valid and typ in QUESTION_TYPES:
            clean.append({"manual_id": mid, "page": page, "question": q, "type": typ})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in clean:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for it in clean:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
    print(f"[완료] {len(clean)}개 질문 저장 → {out_path}", flush=True)
    print("       유형 분포:", by_type, flush=True)
    print("       (일부 수기 검수를 권장합니다.)", flush=True)


if __name__ == "__main__":
    generate_evalset()
