"""어휘 편향 실험 — 짝지은 평가셋 2종을 만들고 '조작이 실제로 일어났는지' 검증한다.

입력: 재작성 결과 JSON(qid, natural) + data/eval/paraphrase_sample.jsonl
출력:
  data/eval/evalset_doc.jsonl      — 원래(문서 유래) 표현, 표본 200문항
  data/eval/evalset_natural.jsonl  — 같은 정답, 실사용자 말투
두 파일은 **같은 순서·같은 정답**이라 paired 비교가 성립한다.

검증: 질문이 '정답 페이지 본문'과 얼마나 겹치는지를 문자 바이그램 자카드로 잰다.
문서 유래 질문이 더 많이 겹쳐야(=편향이 존재해야) 실험이 성립한다. 안 줄었다면
재작성이 실패한 것이므로 평가를 돌릴 필요가 없다.

실행: python scripts/build_natural_evalset.py <재작성결과.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SAMPLE = Path("data/eval/paraphrase_sample.jsonl")
OUT_DOC = Path("data/eval/evalset_doc.jsonl")
OUT_NAT = Path("data/eval/evalset_natural.jsonl")


def bigrams(s: str) -> set[str]:
    t = "".join(ch for ch in s if not ch.isspace())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def coverage(q: str, page: str) -> float:
    """질문 바이그램 중 정답 페이지에 존재하는 비율 — '페이지 어휘를 얼마나 베꼈나'."""
    qa, pa = bigrams(q), bigrams(page)
    return len(qa & pa) / len(qa) if qa else 0.0


def _page_texts() -> dict[tuple[str, int], str]:
    import chromadb

    from rag.indexing.backend import collection_name

    col = chromadb.PersistentClient(path="data/index").get_or_create_collection(collection_name())
    d = col.get(include=["documents", "metadatas"])
    pages: dict[tuple[str, int], list[str]] = {}
    for doc, md in zip(d["documents"], d["metadatas"]):
        md = md or {}
        pages.setdefault((str(md.get("manual_id")), int(md.get("page", 0))), []).append(doc or "")
    return {k: " ".join(v) for k, v in pages.items()}


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python scripts/build_natural_evalset.py <재작성결과.json>")
        return
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    natural = {int(it["qid"]): it["natural"].strip() for it in items if it.get("natural")}

    sample = [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    paired = [r for r in sample if r["qid"] in natural]
    missing = [r["qid"] for r in sample if r["qid"] not in natural]
    if missing:
        print(f"[!] 재작성 누락 {len(missing)}건 → 짝이 맞는 {len(paired)}건만 사용")

    keys = ("manual_id", "page", "question", "type", "qid")
    with open(OUT_DOC, "w", encoding="utf-8") as fd, open(OUT_NAT, "w", encoding="utf-8") as fn:
        for r in paired:
            fd.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
            nat = dict({k: r[k] for k in keys}, question=natural[r["qid"]])
            fn.write(json.dumps(nat, ensure_ascii=False) + "\n")
    print(f"[완료] {len(paired)}문항 × 2 → {OUT_DOC.name} / {OUT_NAT.name}")

    # ── 조작 검증: 정답 페이지 본문과의 어휘 겹침 ──
    pages = _page_texts()
    dc, nc, dj, nj, miss = [], [], [], [], 0
    for r in paired:
        page = pages.get((r["manual_id"], int(r["page"])))
        if not page:
            miss += 1
            continue
        pa = bigrams(page)
        dc.append(coverage(r["question"], page))
        nc.append(coverage(natural[r["qid"]], page))
        dj.append(jaccard(bigrams(r["question"]), pa))
        nj.append(jaccard(bigrams(natural[r["qid"]]), pa))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n[검증] 질문 ↔ 정답 페이지 어휘 겹침 (n={len(dc)}{f', 페이지 없음 {miss}' if miss else ''})")
    print(f"  질문 바이그램이 정답 페이지에 있는 비율")
    print(f"    문서 유래 : {mean(dc):.3f}")
    print(f"    실사용자   : {mean(nc):.3f}   (Δ {mean(nc) - mean(dc):+.3f})")
    drop = [d - n for d, n in zip(dc, nc)]
    lowered = sum(1 for x in drop if x > 0)
    print(f"  겹침이 줄어든 문항: {lowered}/{len(drop)} ({lowered / max(1, len(drop)):.0%})")
    if mean(nc) >= mean(dc):
        print("  [!] 겹침이 줄지 않았습니다 — 재작성이 실패했을 수 있습니다(실험 무효).")
    else:
        print("  → 조작 성공: 실사용자 표현이 정답 페이지 어휘를 덜 씁니다.")


if __name__ == "__main__":
    main()
