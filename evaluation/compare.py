"""A/B 유의성 검정 — 두 검색기 결과의 지표 차이가 통계적으로 진짜인지 본다.

같은 평가셋(동일 순서)에서 나온 두 결과 JSON을 질문 단위로 페어링해,
지표별 Δ(처리−기준)의 95% 신뢰구간을 **페어드 부트스트랩**으로 구한다.
CI가 0을 넘지 않으면 유의(개선이 노이즈가 아님). R@1(이진)은 **McNemar 정확검정**도 함께 본다.
시드를 고정해 재현 가능하다.

52문항처럼 표본이 작을 땐 "+9.6pp"만으론 부족하다 — CI로 불확실성을 같이 봐야 한다.

실행: python -m evaluation.compare hybrid_manual rerank_manual
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

METRICS = ("r@1", "r@3", "r@5", "mrr")
RESULTS_DIR = "evaluation/results"
B = 10000  # 부트스트랩 재표본 수
SEED = 0   # 재현성


def _load_doc(stem: str) -> dict:
    """결과 JSON을 읽는다. 새 이름(<stem>__<backend>_eval.json)과 구 이름을 모두 받는다."""
    d = Path(RESULTS_DIR)
    p = d / f"{stem}_eval.json"
    if not p.exists():
        cands = sorted(d.glob(f"{stem}__*_eval.json"))
        if len(cands) > 1:
            raise SystemExit(
                f"[!] '{stem}'에 여러 백엔드 결과가 있습니다: {[c.name for c in cands]}\n"
                f"    stem에 백엔드를 붙여 지정하세요 (예: {cands[0].name.replace('_eval.json', '')})"
            )
        if not cands:
            raise FileNotFoundError(p)
        p = cands[0]
    return json.load(open(p, encoding="utf-8"))


def _load(stem: str) -> list[dict]:
    return _load_doc(stem)["rows"]


def _paired_bootstrap(diffs: np.ndarray, rng) -> tuple[float, float, float]:
    """질문별 차이 벡터를 재표본해 평균 Δ의 점추정·95% CI를 구한다."""
    n = len(diffs)
    idx = rng.integers(0, n, size=(B, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def _mcnemar_exact(base: np.ndarray, treat: np.ndarray) -> tuple[int, int, float]:
    """R@1 이진 결과의 불일치쌍으로 McNemar 정확(이항) p값. b=기준만 정답, c=처리만 정답."""
    b = int(np.sum((base == 1) & (treat == 0)))
    c = int(np.sum((base == 0) & (treat == 1)))
    nd = b + c
    if nd == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(nd, i) for i in range(k + 1)) * (0.5 ** nd)
    return b, c, min(1.0, 2.0 * tail)


def compare(base_stem: str, treat_stem: str) -> None:
    bdoc, tdoc = _load_doc(base_stem), _load_doc(treat_stem)
    # 서로 다른 임베딩 백엔드/평가셋을 비교하면 A/B가 아니라 사과-오렌지가 된다
    for field, label in (("backend", "임베딩 백엔드"), ("evalset", "평가셋")):
        b, t = bdoc.get(field), tdoc.get(field)
        if b and t and b != t:
            print(f"[!] {label}가 다릅니다: {base_stem}={b} vs {treat_stem}={t} — 비교 중단.")
            return
    base, treat = bdoc["rows"], tdoc["rows"]
    if len(base) != len(treat):
        print(f"[!] 표본 크기 불일치: {base_stem}={len(base)} vs {treat_stem}={len(treat)} — 같은 평가셋이 아님")
        return
    # 동일 순서·동일 정답인지 확인 (페어링 안전장치)
    mism = sum(1 for a, b in zip(base, treat) if a["gold"] != b["gold"])
    if mism:
        print(f"[!] 정답(gold) {mism}개 불일치 — 두 결과가 같은 평가셋/순서가 아님. 중단.")
        return

    rng = np.random.default_rng(SEED)
    n = len(base)
    print(f"\n[A/B] 기준={base_stem}  vs  처리={treat_stem}   (n={n}, paired bootstrap B={B}, seed={SEED})\n")
    hdr = f"{'지표':<8}{'기준':>8}{'처리':>8}{'Δ':>8}{'95% CI':>18}{'유의':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in METRICS:
        bvec = np.array([r[m] for r in base], dtype=float)
        tvec = np.array([r[m] for r in treat], dtype=float)
        mean_d, lo, hi = _paired_bootstrap(tvec - bvec, rng)
        sig = "✓" if (lo > 0 or hi < 0) else "·"
        print(f"{m:<8}{bvec.mean():>8.3f}{tvec.mean():>8.3f}{mean_d:>+8.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>18}{sig:>6}")

    # R@1 McNemar 정확검정 (이진 지표 전용)
    b1 = np.array([r["r@1"] for r in base])
    t1 = np.array([r["r@1"] for r in treat])
    only_b, only_t, p = _mcnemar_exact(b1, t1)
    verdict = "유의(p<0.05)" if p < 0.05 else "미검출"
    print(f"\n[McNemar·R@1] 기준만 정답 {only_b} / 처리만 정답 {only_t} / 나머지 동일 → p={p:.4f} ({verdict})")
    print("  · CI가 0을 넘지 않거나 p<0.05면, 그 개선은 표본 노이즈로 설명되지 않는다.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python -m evaluation.compare <기준_stem> <처리_stem>")
        print("  예: python -m evaluation.compare hybrid_manual rerank_manual")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
