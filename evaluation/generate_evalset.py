"""평가셋 자동 생성 — 매뉴얼 본문으로 LLM(Gemini)이 "질문 ↔ 정답 페이지" 쌍을 만든다.

질문 유형을 분산한다: 에러코드형 · 사용법형 · 사양·수치형 · 문제해결형.
이미 인덱싱된(=OCR된) 텍스트를 Chroma에서 읽어 쓰므로 재-OCR/torch가 필요 없다.
rate limit 회피를 위해 전체를 한 번의 호출로 생성한다. 결과는 data/eval/evalset.jsonl.

실행: python -m evaluation.generate_evalset
"""
from __future__ import annotations

import json
import os
from pathlib import Path

QUESTION_TYPES = ["error_code", "how_to", "spec_numeric", "troubleshooting"]

PROMPT_TEMPLATE = """다음은 가전(세탁기) 매뉴얼의 쪽별 내용이다. OCR로 추출되어 오타가 있을 수 있으니 문맥으로 이해하라.

사용자가 실제로 음성으로 물어볼 법한 한국어 질문을 만들어라. 규칙:
- 각 질문의 답이 '실제로 있는 쪽 번호'를 page에 정확히 단다.
- type은 다음 중 하나: {types}
  (error_code=에러코드, how_to=사용법, spec_numeric=사양·수치, troubleshooting=문제해결)
- 쪽당 1~2개, 전체적으로 유형을 골고루 분산한다.
- 그 쪽 내용만으로 답이 분명한 질문만. 애매하면 만들지 않는다.
- 출력은 JSON 배열만. 형식:
[{{"question": "세탁기 UE 에러는 무슨 뜻이야?", "page": 9, "type": "error_code"}}]

=== 매뉴얼 내용 ===
{manual}
"""


def _load_page_texts(index_dir: str) -> dict[int, str]:
    import chromadb

    from rag.vectorstore.store import COLLECTION

    client = chromadb.PersistentClient(path=index_dir)
    col = client.get_or_create_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    pages: dict[int, list[str]] = {}
    for doc, md in zip(data["documents"], data["metadatas"]):
        page = int((md or {}).get("page", 0))
        pages.setdefault(page, []).append(doc or "")
    return {p: "\n".join(t).strip() for p, t in sorted(pages.items())}


def generate_evalset(index_dir: str = "data/index", out_path: str = "data/eval/evalset.jsonl") -> None:
    from dotenv import load_dotenv

    load_dotenv()
    from google import genai
    from google.genai import types

    page_texts = _load_page_texts(index_dir)
    if not page_texts:
        print("[!] 인덱스가 비었습니다. 먼저 build_index/reindex로 인덱싱하세요.")
        return

    manual = "\n\n".join(f"=== {p}쪽 ===\n{txt}" for p, txt in page_texts.items() if txt)
    prompt = PROMPT_TEMPLATE.format(types=" / ".join(QUESTION_TYPES), manual=manual)

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"[*] {len(page_texts)}개 쪽으로 평가셋 생성 중 (model={model})...", flush=True)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
    )

    try:
        items = json.loads(resp.text)
    except json.JSONDecodeError:
        print("[!] JSON 파싱 실패. 모델 원응답:\n", (resp.text or "")[:500])
        return

    # 검증/정제
    valid_pages = set(page_texts)
    clean = []
    for it in items:
        q = (it.get("question") or "").strip()
        page = it.get("page")
        typ = it.get("type")
        if q and isinstance(page, int) and page in valid_pages and typ in QUESTION_TYPES:
            clean.append({"question": q, "page": page, "type": typ})

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
