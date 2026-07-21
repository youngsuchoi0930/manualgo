"""통합 리포트 — 6개 검색기 구성의 지표표 + 3쌍 A/B 유의성 + 필요 표본 크기 추정.

compare.py의 페어드 부트스트랩/McNemar를 재사용해 한 번에 정리한다:
  (1) 6구성 한눈 표 (n, R@1/3/5, MRR)
  (2) manual / category / auto(agentic) 3쌍의 Δ·95% CI·McNemar p
  (3) 필요 표본 크기: 관측된 효과크기(불일치율 p_d, 우세비 ψ)를 그대로 두고
      R@1 McNemar가 80% 검출력에 도달하는 n*를 시뮬레이션으로 추정.
      ※ 관측 효과 기준의 '계획용' 추정이지 사후 검정력 주장이 아니다. χ² 근사 사용.

실행: python -m evaluation.report
"""
from __future__ import annotations

import math

import numpy as np

from evaluation.compare import _load, _mcnemar_exact, _paired_bootstrap

# (라벨, 기준_stem, 처리_stem)
PAIRS = [
    ("manual  (오라클: 정확한 매뉴얼)", "hybrid_manual", "rerank_manual"),
    ("category(오라클: 제품군)", "hybrid_category", "rerank_category"),
    ("auto    (agentic 자동 스코핑)", "agentic", "agentic-rerank"),
]
METRICS = ("r@1", "r@3", "r@5", "mrr")
CHI2_CRIT = 3.841  # χ²_1, α=0.05


def _means(stem: str) -> dict:
    rows = _load(stem)
    return {"n": len(rows), **{m: float(np.mean([r[m] for r in rows])) for m in METRICS}}


def _sample_size_for_power(p_disc: float, psi: float, target: float = 0.80,
                           alpha_crit: float = CHI2_CRIT, sims: int = 4000,
                           n_max: int = 1500, seed: int = 0) -> tuple[int | None, float]:
    """관측된 불일치율 p_disc·우세비 ψ를 고정하고, McNemar(χ² 근사)가 target 검출력에
    도달하는 최소 n을 찾는다. (n*, 현재는 참고용으로 반환하지 않음)."""
    rng = np.random.default_rng(seed)

    def power(n: int) -> float:
        nd = rng.binomial(n, p_disc, size=sims)
        c = np.where(nd > 0, rng.binomial(np.maximum(nd, 1), psi), 0)
        b = nd - c
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(nd > 0, (b - c) ** 2 / np.maximum(nd, 1), 0.0)
        return float(np.mean(chi2 > alpha_crit))

    for n in range(20, n_max + 1, 10):
        if power(n) >= target:
            return n, power(n)
    return None, power(n_max)


def report() -> None:
    print("=" * 64)
    print(" manualgo · 리랭커 A/B 통합 리포트")
    print("=" * 64)

    # (1) 6구성 한눈 표
    print("\n[1] 구성별 지표")
    hdr = f"{'구성':<18}{'n':>5}" + "".join(f"{m:>8}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    seen: set[str] = set()
    for _, base, treat in PAIRS:
        for stem in (base, treat):
            if stem in seen:
                continue
            seen.add(stem)
            try:
                s = _means(stem)
            except FileNotFoundError:
                print(f"{stem:<18}{'(결과 없음)':>5}")
                continue
            print(f"{stem:<18}{s['n']:>5}" + "".join(f"{s[m]:>8.3f}" for m in METRICS))

    # (2) 3쌍 A/B
    print("\n[2] A/B 유의성 (paired bootstrap 95% CI · McNemar R@1)")
    for label, base, treat in PAIRS:
        try:
            brows, trows = _load(base), _load(treat)
        except FileNotFoundError:
            print(f"\n  · {label}: 결과 파일 없음 — 건너뜀")
            continue
        if len(brows) != len(trows):
            print(f"\n  · {label}: n 불일치({len(brows)} vs {len(trows)}) — 같은 평가셋 아님")
            continue
        rng = np.random.default_rng(0)
        print(f"\n  {label}   (n={len(brows)})")
        for m in METRICS:
            bvec = np.array([r[m] for r in brows], dtype=float)
            tvec = np.array([r[m] for r in trows], dtype=float)
            d, lo, hi = _paired_bootstrap(tvec - bvec, rng)
            sig = "✓유의" if (lo > 0 or hi < 0) else "·"
            print(f"    {m:<5} {bvec.mean():.3f} → {tvec.mean():.3f}  Δ{d:+.3f}  "
                  f"CI[{lo:+.3f},{hi:+.3f}]  {sig}")
        b1 = np.array([r["r@1"] for r in brows])
        t1 = np.array([r["r@1"] for r in trows])
        ob, ot, p = _mcnemar_exact(b1, t1)
        print(f"    McNemar R@1: 기준만 {ob} / 처리만 {ot} → p={p:.4f}"
              f"  {'(유의)' if p < 0.05 else '(미검출)'}")

    # (3) 필요 표본 크기 (관측 효과크기 기준, R@1 기준쌍)
    print("\n[3] 필요 표본 크기 — R@1에서 이 효과를 80% 검출력으로 잡으려면")
    print("    (관측된 불일치율·우세비 고정, χ² 근사 시뮬레이션 · 계획용 추정)")
    for label, base, treat in PAIRS:
        try:
            b1 = np.array([r["r@1"] for r in _load(base)])
            t1 = np.array([r["r@1"] for r in _load(treat)])
        except FileNotFoundError:
            continue
        n_obs = len(b1)
        ob = int(np.sum((b1 == 1) & (t1 == 0)))
        ot = int(np.sum((b1 == 0) & (t1 == 1)))
        nd = ob + ot
        if nd == 0:
            print(f"    {label}: 불일치쌍 0 — 추정 불가")
            continue
        p_disc = nd / n_obs
        psi = ot / nd  # 불일치쌍 중 처리(리랭커) 우세 비율
        n_star, _ = _sample_size_for_power(p_disc, psi)
        tag = f"~{n_star}문항" if n_star else ">1500문항"
        print(f"    {label}: 불일치 {nd}쌍(처리우세 {ot}/{nd}={psi:.0%}), "
              f"불일치율 {p_disc:.0%} → 필요 {tag}")


if __name__ == "__main__":
    report()
