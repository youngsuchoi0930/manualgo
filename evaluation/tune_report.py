"""재튜닝 판정 — 튜닝 절반으로 구성을 고르고, 홀드아웃 절반으로만 유의성을 확정한다.

같은 평가셋으로 '구성 선택'과 '최종 보고'를 다 하면 선택 과정 자체가 낙관 편향을 만든다
(여러 구성 중 우연히 좋은 것을 골라놓고 그 점수를 보고하게 됨). 그래서 qid 짝수(320)로
구성을 고르고, 홀수(320)에서 기준 구성 대비 paired bootstrap CI·McNemar로 확정한다.

실행: python -m evaluation.tune_report <기준_stem> <후보_stem> [<후보_stem> ...]
  예: python -m evaluation.tune_report \
        rerank_manual__pool10__evalset_natural640__onnx \
        rerank_manual__pool10__bw0.5__evalset_natural640__onnx ...
"""
from __future__ import annotations

import sys

import numpy as np

from evaluation.compare import _load_doc, _mcnemar_exact, _paired_bootstrap

METRICS = ("r@1", "r@3", "r@5", "mrr")


def _halves(rows: list[dict]) -> tuple[list[int], list[int]]:
    """(튜닝 인덱스, 홀드아웃 인덱스) — 행 순서=qid 순서이므로 인덱스 짝/홀로 나눈다."""
    tune = [i for i in range(len(rows)) if i % 2 == 0]
    hold = [i for i in range(len(rows)) if i % 2 == 1]
    return tune, hold


def _mean(rows: list[dict], idx: list[int], m: str) -> float:
    return float(np.mean([rows[i][m] for i in idx])) if idx else 0.0


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    base_stem, cand_stems = sys.argv[1], sys.argv[2:]

    docs = {s: _load_doc(s) for s in [base_stem, *cand_stems]}
    n = len(docs[base_stem]["rows"])
    for s, d in docs.items():
        if len(d["rows"]) != n:
            raise SystemExit(f"[!] {s}: 행 수 불일치 ({len(d['rows'])} vs {n})")
        if any(a["gold"] != b["gold"] for a, b in zip(d["rows"], docs[base_stem]["rows"])):
            raise SystemExit(f"[!] {s}: gold 불일치 — 같은 평가셋이 아님")

    tune, hold = _halves(docs[base_stem]["rows"])
    print(f"[분할] 전체 {n} = 튜닝 {len(tune)} + 홀드아웃 {len(hold)}  (qid 짝/홀)")

    # ── 1) 튜닝 절반에서 구성 비교 ──
    print(f"\n[1] 튜닝 절반 성적 (구성 선택용 — 이 수치로 보고하지 말 것)")
    hdr = f"{'구성':<58}" + "".join(f"{m:>8}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    scored = []
    for s in [base_stem, *cand_stems]:
        rows = docs[s]["rows"]
        vals = {m: _mean(rows, tune, m) for m in METRICS}
        scored.append((s, vals))
        mark = " (기준)" if s == base_stem else ""
        print(f"{s[:56]:<58}" + "".join(f"{vals[m]:>8.3f}" for m in METRICS) + mark)

    cands_only = [x for x in scored if x[0] != base_stem]
    winner = max(cands_only, key=lambda x: (x[1]["r@1"], x[1]["mrr"]))[0] if cands_only else base_stem
    print(f"\n→ 튜닝 절반 승자: {winner}")

    # ── 2) 홀드아웃 절반에서 승자 vs 기준 확정 ──
    print(f"\n[2] 홀드아웃 확정: {base_stem}  vs  {winner}  (n={len(hold)})")
    if winner == base_stem:
        print("  승자가 기준과 동일 — 현재 구성 유지가 결론.")
        return
    rng = np.random.default_rng(0)
    brows, trows = docs[base_stem]["rows"], docs[winner]["rows"]
    for m in METRICS:
        bv = np.array([brows[i][m] for i in hold], dtype=float)
        tv = np.array([trows[i][m] for i in hold], dtype=float)
        d, lo, hi = _paired_bootstrap(tv - bv, rng)
        sig = "✓유의" if (lo > 0 or hi < 0) else "·"
        print(f"    {m:<5} {bv.mean():.3f} → {tv.mean():.3f}  Δ{d:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")
    b1 = np.array([brows[i]["r@1"] for i in hold])
    t1 = np.array([trows[i]["r@1"] for i in hold])
    ob, ot, p = _mcnemar_exact(b1, t1)
    print(f"    McNemar R@1: 기준만 {ob} / 승자만 {ot} → p={p:.4f} {'(유의)' if p < 0.05 else '(미검출)'}")
    print("\n  · 홀드아웃에서 유의하면 채택, 아니면 기준 유지가 정직한 결론이다.")


if __name__ == "__main__":
    main()
