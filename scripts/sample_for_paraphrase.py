"""평가셋 어휘 편향 실험 — 재작성할 표본을 층화 추출한다.

현재 평가셋은 LLM이 '정답 페이지를 보고' 만들었다. 그래서 질문이 그 페이지의 단어를
그대로 쓰고, 어휘 매칭에 유리하게 편향돼 있을 수 있다(=점수가 낙관적).
이를 재려면 **정답(manual_id, page)은 그대로 두고 표현만 실사용자 말투로 바꾼**
짝지은 평가셋이 필요하다. 그러면 두 점수의 차이가 곧 어휘 편향의 크기다.

유형·매뉴얼에 고르게 퍼지도록 층화 추출한다. 시드 고정.
실행: python scripts/sample_for_paraphrase.py [N]
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 11
SRC = Path("data/eval/evalset.jsonl")
OUT = Path("data/eval/paraphrase_sample.jsonl")


def main() -> None:
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, r in enumerate(rows):
        r["qid"] = i  # 원본 순서를 id로 고정 — 짝 맞추기에 쓴다

    # 유형별 목표 개수를 전체 분포에 비례해 배분
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    rng = random.Random(SEED)
    picked: list[dict] = []
    for typ, items in sorted(by_type.items()):
        quota = max(1, round(n_target * len(items) / len(rows)))
        # 매뉴얼이 겹치지 않게 라운드로빈 — 한 매뉴얼에 쏠리지 않도록
        by_manual: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_manual[it["manual_id"]].append(it)
        for lst in by_manual.values():
            rng.shuffle(lst)
        order = sorted(by_manual)
        rng.shuffle(order)
        take: list[dict] = []
        while len(take) < quota and any(by_manual[m] for m in order):
            for m in order:
                if by_manual[m] and len(take) < quota:
                    take.append(by_manual[m].pop())
        picked.extend(take)

    picked.sort(key=lambda r: r["qid"])
    with open(OUT, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[완료] {len(picked)}문항 추출 → {OUT}")
    tc: dict[str, int] = defaultdict(int)
    for r in picked:
        tc[r["type"]] += 1
    print("  유형:", dict(tc))
    print(f"  매뉴얼 {len({r['manual_id'] for r in picked})}종")


if __name__ == "__main__":
    main()
