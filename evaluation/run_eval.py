"""베이스라인 채점 — 평가셋으로 검색기를 돌려 Recall@k·MRR을 측정한다 (유형별 포함).

정답은 (manual_id, page) 쌍으로 판정한다 — 매뉴얼이 여럿이라 page 번호만으론 모호하다.
지금은 Naive 검색기. 이후 Hybrid+Reranker / Agentic을 같은 평가셋에 돌려 단계별로 비교한다.
결과는 콘솔 표 + evaluation/results/<name>_eval.json 으로 저장한다.

실행: python -m evaluation.run_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import recall_at_k, reciprocal_rank

KS = (1, 3, 5)
RETRIEVE_K = 10  # 청크를 넉넉히 뽑아 unique (매뉴얼,페이지) 순위를 만든다


def _load_evalset(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _unique_keys(chunks) -> list[tuple[str, int]]:
    """검색된 청크에서 (manual_id, page)를 순서 유지·중복 제거해 반환."""
    keys: list[tuple[str, int]] = []
    for c in chunks:
        k = (c.manual_id, c.page)
        if k not in keys:
            keys.append(k)
    return keys


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _make_retriever(name: str):
    if name == "bm25":
        from rag.retrieval.bm25 import BM25Retriever

        return BM25Retriever()
    if name == "hybrid":
        from rag.retrieval.hybrid import HybridRetriever

        return HybridRetriever()
    if name == "agentic":
        from rag.retrieval.agentic import AgenticRetriever

        return AgenticRetriever()
    # RERANK_POOL로 리랭커에 넘길 후보 수를 조절한다 — 라이브 지연이 후보 수에 거의 비례해서,
    # "후보를 줄이면 정확도를 얼마나 잃는가"를 같은 평가셋으로 재야 트레이드오프를 정할 수 있다.
    import os

    pool = int(os.environ.get("RERANK_POOL") or 20)
    if name in ("rerank", "hybrid-rerank"):
        from rag.retrieval.hybrid import HybridRetriever
        from rag.retrieval.reranker import Reranker

        return HybridRetriever(reranker=Reranker(), rerank_pool=pool)
    if name == "agentic-rerank":
        from rag.retrieval.agentic import AgenticRetriever
        from rag.retrieval.hybrid import HybridRetriever
        from rag.retrieval.reranker import Reranker

        return AgenticRetriever(hybrid=HybridRetriever(reranker=Reranker(), rerank_pool=pool))
    from rag.retrieval.naive import NaiveRetriever

    return NaiveRetriever()


def _category_of(mid: str) -> str:
    """카테고리 판정은 agentic.manual_category 하나만 쓴다.

    (예전엔 여기 별도 키워드 표가 있어 식기세척기·에어컨 오버라이드가 빠졌고,
     category 스코핑과 Agentic 자동분류가 서로 다른 그룹을 만들었다.)
    """
    from rag.retrieval.agentic import manual_category

    return manual_category(mid)


def _all_manual_ids(index_dir: str = "data/index") -> list[str]:
    import chromadb

    from rag.indexing.backend import collection_name

    col = chromadb.PersistentClient(path=index_dir).get_or_create_collection(collection_name())
    metas = col.get(include=["metadatas"])["metadatas"]
    return sorted({(m or {}).get("manual_id") for m in metas if m and m.get("manual_id")})


def run_eval(
    evalset_path: str = "data/eval/evalset.jsonl",
    results_dir: str = "evaluation/results",
    name: str = "naive",
    scope: str | None = None,  # None | "manual" | "category" — oracle 스코핑
) -> None:
    evalset = _load_evalset(evalset_path)
    if not evalset:
        print(f"[!] 평가셋이 비었습니다: {evalset_path} (먼저 generate_evalset 실행)")
        return
    if "manual_id" not in evalset[0]:
        print("[!] 평가셋에 manual_id가 없습니다(구버전). generate_evalset로 다시 만드세요.")
        return

    retriever = _make_retriever(name)
    cat_groups: dict[str, list[str]] = {}
    if scope == "category":
        for mid in _all_manual_ids():
            cat_groups.setdefault(_category_of(mid), []).append(mid)

    rows = []
    label = name + (f" (scope={scope})" if scope else "")
    print(f"[*] '{label}' 검색기로 {len(evalset)}개 질문 평가 중...", flush=True)
    for item in evalset:
        q = item["question"]
        gold = (item["manual_id"], int(item["page"]))
        typ = item.get("type", "?")
        if scope == "manual":
            scope_ids = [item["manual_id"]]
        elif scope == "category":
            scope_ids = cat_groups.get(_category_of(item["manual_id"]))
        else:
            scope_ids = None
        keys = _unique_keys(retriever.retrieve(q, top_k=RETRIEVE_K, manual_ids=scope_ids))
        row = {"type": typ, "gold": gold, "pred": keys, "mrr": reciprocal_rank(keys, gold)}
        for k in KS:
            row[f"r@{k}"] = recall_at_k(keys, gold, k)
        rows.append(row)

    metric_keys = [f"r@{k}" for k in KS] + ["mrr"]

    def summarize(subset: list[dict]) -> dict:
        return {"n": len(subset), **{m: round(_mean([r[m] for r in subset]), 3) for m in metric_keys}}

    overall = summarize(rows)
    by_type = {t: summarize([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}

    # 콘솔 출력
    hdr = f"{'유형':<16}{'n':>4}" + "".join(f"{m:>8}" for m in metric_keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    print(f"{'(전체)':<16}{overall['n']:>4}" + "".join(f"{overall[m]:>8.3f}" for m in metric_keys))
    for t, s in by_type.items():
        print(f"{t:<16}{s['n']:>4}" + "".join(f"{s[m]:>8.3f}" for m in metric_keys))

    # 결과 파일명에 임베딩 백엔드를 넣는다 — 넣지 않으면 임베더가 다른 실행이 서로를 덮어쓰고,
    # compare.py가 서로 다른 벡터공간의 결과를 짝지어 비교해도 알아챌 수 없다.
    from rag.indexing.backend import backend_name, collection_name

    backend = backend_name()
    # 리랭커 후보 수가 기본과 다르면 파일명에 남긴다 (pool 변형끼리 덮어쓰지 않도록)
    import os as _os

    pool_env = _os.environ.get("RERANK_POOL")
    tag = f"__pool{pool_env}" if pool_env and "rerank" in name else ""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(results_dir) / f"{name}{('_' + scope) if scope else ''}{tag}__{backend}_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"name": name, "scope": scope, "backend": backend,
                   "rerank_pool": int(pool_env) if pool_env else None,
                   "collection": collection_name(), "evalset": evalset_path,
                   "overall": overall, "by_type": by_type, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[완료] 저장 → {out}", flush=True)


if __name__ == "__main__":
    import sys

    # 사용법: python -m evaluation.run_eval [naive|bm25|hybrid] [manual|category]
    run_eval(
        name=sys.argv[1] if len(sys.argv) > 1 else "naive",
        scope=sys.argv[2] if len(sys.argv) > 2 else None,
    )
