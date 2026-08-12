"""실사용자 평가셋 640 완성 — 재작성 결과(200+440)를 병합하고 조작을 재검증한다.

원본 evalset.jsonl과 **같은 순서·같은 정답**으로 배열해, 기존 640문항 결과 파일과
그대로 짝지은(paired) 비교가 되게 한다. 재작성이 누락된 문항은 원문을 유지하되 개수를 보고한다.

실행: python scripts/merge_natural_evalset.py <재작성1.json> [<재작성2.json> ...]
출력: data/eval/evalset_natural640.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("data/eval/evalset.jsonl")
OUT = Path("data/eval/evalset_natural640.jsonl")


def bigrams(s: str) -> set[str]:
    t = "".join(ch for ch in s if not ch.isspace())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def coverage(q: str, page: str) -> float:
    qa = bigrams(q)
    return len(qa & bigrams(page)) / len(qa) if qa else 0.0


def main() -> None:
    natural: dict[int, str] = {}
    for arg in sys.argv[1:]:
        payload = json.loads(Path(arg).read_text(encoding="utf-8"))
        items = payload.get("items", payload) if isinstance(payload, dict) else payload
        for it in items:
            if it.get("natural"):
                natural[int(it["qid"])] = it["natural"].strip()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [i for i in range(len(rows)) if i not in natural]
    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            q = natural.get(i, r["question"])  # 누락 시 원문 유지 (개수는 아래 보고)
            f.write(json.dumps({"manual_id": r["manual_id"], "page": int(r["page"]),
                                "question": q, "type": r["type"], "qid": i},
                               ensure_ascii=False) + "\n")
    print(f"[완료] {len(rows)}문항 → {OUT}  (재작성 {len(natural)} · 원문 유지 {len(missing)})")
    if missing:
        print("  누락 qid:", missing[:20], "..." if len(missing) > 20 else "")

    # 조작 재검증 — 정답 페이지 어휘 겹침 (전체)
    import chromadb

    from rag.indexing.backend import collection_name

    col = chromadb.PersistentClient(path="data/index").get_or_create_collection(collection_name())
    d = col.get(include=["documents", "metadatas"])
    pages: dict[tuple[str, int], list[str]] = {}
    for doc, md in zip(d["documents"], d["metadatas"]):
        md = md or {}
        pages.setdefault((str(md.get("manual_id")), int(md.get("page", 0))), []).append(doc or "")
    page_text = {k: " ".join(v) for k, v in pages.items()}

    dc, nc = [], []
    for i, r in enumerate(rows):
        page = page_text.get((r["manual_id"], int(r["page"])))
        if not page or i not in natural:
            continue
        dc.append(coverage(r["question"], page))
        nc.append(coverage(natural[i], page))
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    lowered = sum(1 for a, b in zip(dc, nc) if b < a)
    print(f"[검증 n={len(dc)}] 정답 페이지 어휘 겹침: 문서 {m(dc):.3f} → 실사용자 {m(nc):.3f} "
          f"(겹침 감소 {lowered}/{len(dc)})")


if __name__ == "__main__":
    main()
