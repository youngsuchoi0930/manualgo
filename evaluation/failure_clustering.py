"""실패 클러스터링 (차별 포인트) — 검색기가 여전히 틀리는 질문을 유형·매뉴얼별로 묶어 약점을 진단한다.

run_eval 결과(JSON)와 평가셋을 인덱스로 조인해:
- 검색기(naive/bm25/hybrid)별 유형 R@1 비교
- 유형별 '완전 실패(정답이 top-5에 없음)' / 'top-1 놓침' 집계
- 매뉴얼별 실패, 실패 예시(질문 + 정답 vs 가져온 것)
를 리포트한다. 임베딩/LLM 불필요(stdlib만) → 쿼터 0.

실행: python -m evaluation.failure_clustering [naive|bm25|hybrid]
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_results(results_dir: str, name: str) -> dict | None:
    p = Path(results_dir) / f"{name}_eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def cluster_failures(
    name: str = "hybrid",
    evalset_path: str = "data/eval/evalset.jsonl",
    results_dir: str = "evaluation/results",
) -> None:
    res = _load_results(results_dir, name)
    if not res:
        print(f"[!] 결과 없음: {results_dir}/{name}_eval.json (먼저 'run_eval {name}' 실행)")
        return
    evalset = _load_jsonl(evalset_path)
    rows = res["rows"]

    items = []
    for ev, row in zip(evalset, rows):
        items.append(
            {
                "question": ev.get("question", ""),
                "type": row.get("type", "?"),
                "gold": tuple(row["gold"]),
                "pred": [tuple(p) for p in row["pred"]],
                "r1": row.get("r@1", 0.0),
                "r5": row.get("r@5", 0.0),
            }
        )

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"=== 실패 분석: {name} ({len(items)}문항) ===\n")

    # 0) 검색기 비교 (있는 것만) — 유형별 R@1
    avail = {nm: _load_results(results_dir, nm) for nm in ("naive", "bm25", "hybrid")}
    avail = {k: v for k, v in avail.items() if v}
    if len(avail) > 1:
        out("[검색기 비교] 유형별 R@1")
        types = sorted({t for v in avail.values() for t in v["by_type"]})
        out("  " + f"{'유형':16}" + "".join(f"{nm:>9}" for nm in avail))
        for t in ["(전체)"] + types:
            cells = []
            for v in avail.values():
                val = v["overall"]["r@1"] if t == "(전체)" else v["by_type"].get(t, {}).get("r@1", 0.0)
                cells.append(f"{val:>9.3f}")
            out(f"  {t:16}" + "".join(cells))
        out("")

    # 1) 유형별 실패
    out("[유형별] n / 완전실패(R@5=0) / top1놓침(R@1=0)")
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for it in items:
        b = by_type[it["type"]]
        b[0] += 1
        b[1] += it["r5"] == 0
        b[2] += it["r1"] == 0
    for t, (n, hf, m1) in sorted(by_type.items()):
        out(f"  {t:16} {n:3}   완전실패 {hf} ({_pct(hf, n)})   top1놓침 {m1} ({_pct(m1, n)})")

    # 2) 매뉴얼별 완전실패
    out("\n[매뉴얼별] 완전실패(R@5=0)")
    by_man: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for it in items:
        m = it["gold"][0]
        by_man[m][0] += 1
        by_man[m][1] += it["r5"] == 0
    for m, (n, hf) in sorted(by_man.items(), key=lambda x: -x[1][1]):
        if hf:
            out(f"  {m:34} {hf}/{n}")

    # 3) 완전실패 예시
    out("\n[완전실패 예시] 정답이 top-5에 아예 없음")
    hard = [it for it in items if it["r5"] == 0]
    for it in hard:
        out(f"  · [{it['type']}] {it['question']}")
        out(f"      정답 {it['gold'][0]} p{it['gold'][1]} | 가져온 것 {it['pred'][:3]}")
    if not hard:
        out("  (없음)")

    # 4) top-1만 놓친 예시 (정답은 top-5엔 있음)
    out("\n[top-1 놓침 예시] 정답이 top-5엔 있지만 1위가 아님 (최대 8개)")
    near = [it for it in items if it["r1"] == 0 and it["r5"] > 0]
    for it in near[:8]:
        out(f"  · [{it['type']}] {it['question']}  (정답 {it['gold'][0]} p{it['gold'][1]})")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    md = Path(results_dir) / f"failure_analysis_{name}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[저장] {md}")


if __name__ == "__main__":
    import sys

    cluster_failures(name=sys.argv[1] if len(sys.argv) > 1 else "hybrid")
