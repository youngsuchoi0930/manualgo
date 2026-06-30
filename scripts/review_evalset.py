"""평가셋 빠른 검수 — 각 질문 옆에 '정답 페이지 본문'을 보여줘 눈으로 확인한다.

정답 페이지에서 실제로 답이 나오는지/모호하지 않은지 스폿체크용. 임베딩·LLM 불필요(쿼터 0).
실행: python scripts/review_evalset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    import chromadb

    from rag.vectorstore.store import COLLECTION

    evalset_path = ROOT / "data" / "eval" / "evalset.jsonl"
    evalset = [json.loads(line) for line in evalset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    col = chromadb.PersistentClient(path=str(ROOT / "data" / "index")).get_or_create_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    page_text: dict[tuple, list[str]] = {}
    for doc, md in zip(data["documents"], data["metadatas"]):
        md = md or {}
        page_text.setdefault((md.get("manual_id"), md.get("page")), []).append(doc or "")
    page_text = {k: " ".join(v) for k, v in page_text.items()}

    for i, it in enumerate(evalset, 1):
        txt = page_text.get((it["manual_id"], it["page"]), "(본문 없음)")[:220].replace("\n", " ")
        print(f"[{i}] ({it['type']}) {it['question']}")
        print(f"    → 정답 {it['manual_id']} p{it['page']}: {txt}\n")

    print(f"총 {len(evalset)}문항. 각 질문의 답이 '정답 페이지 본문'에 실제로 있는지 확인하세요.")


if __name__ == "__main__":
    main()
