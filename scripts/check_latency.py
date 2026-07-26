"""라이브 경로 지연 측정 — 캐시 없는 신규 질의로 단계별 소요를 잰다.

음성 UX는 총 3초 안쪽이 목표다. 어디에 시간이 가는지(임베딩 / BM25+벡터검색 / 재정렬)
분해해서 봐야 리랭커를 라이브에 넣을지 판단할 수 있다.
캐시를 끄고(EMBED_CACHE 무시) 매번 새 질문을 써서 낙관적 측정을 피한다.

실행: conda run -n manual python scripts/check_latency.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 서로 다른 신규 질문 — 캐시 히트로 지연이 과소평가되는 것을 막는다
QUERIES = [
    "세탁기에서 물이 새면 어떻게 확인하나요?",
    "냉장고 제빙기가 얼음을 안 만들면 뭘 봐야 하죠?",
    "에어컨을 오래 안 쓰다가 다시 켤 때 주의할 점은?",
    "전자레인지 내부를 청소하는 방법 알려줘",
    "정수기 필터 교체 주기가 어떻게 되나요?",
    "청소기 먼지통을 비우는 방법은?",
]


def _stage_times(retriever, q: str) -> dict:
    t0 = time.perf_counter()
    qv = retriever.dense.embedder.encode([q], task_type="RETRIEVAL_QUERY")[0]
    t1 = time.perf_counter()
    retriever.dense.store.search(qv, top_k=20)
    t2 = time.perf_counter()
    retriever.bm25.retrieve(q, top_k=20)
    t3 = time.perf_counter()
    return {"임베딩": t1 - t0, "벡터검색": t2 - t1, "BM25": t3 - t2}


def main() -> None:
    import os

    os.environ["EMBED_CACHE"] = "0"  # 캐시 끄기 — 라이브 신규 질의 조건

    from rag.retrieval.hybrid import HybridRetriever
    from rag.retrieval.reranker import Reranker

    print("[구성] 임베더 provider / 리랭커 provider 는 환경변수로 정함\n", flush=True)

    base = HybridRetriever()
    print(f"임베더: {base.dense.embedder.provider} · {base.dense.embedder.weights}", flush=True)
    rr = Reranker()
    print(f"리랭커: {rr.provider} · {rr.weights}\n", flush=True)
    pool = int(os.environ.get("RERANK_POOL") or 20)
    print(f"리랭커 후보 수(rerank_pool) = {pool}\n", flush=True)
    withrr = HybridRetriever(reranker=rr, bm25=base.bm25, dense=base.dense, rerank_pool=pool)

    for label, r in (("리랭커 없음", base), ("리랭커 포함", withrr)):
        r.retrieve(QUERIES[0], top_k=5)  # 워밍업
        tot, stages = [], {"임베딩": 0.0, "벡터검색": 0.0, "BM25": 0.0}
        for q in QUERIES:
            st = _stage_times(r, q)
            for k in stages:
                stages[k] += st[k]
            t = time.perf_counter()
            r.retrieve(q, top_k=5)
            tot.append(time.perf_counter() - t)
        n = len(QUERIES)
        avg = sum(tot) / n
        parts = " · ".join(f"{k} {v / n * 1000:.0f}ms" for k, v in stages.items())
        rest = avg - sum(stages.values()) / n
        print(f"[{label}] 총 {avg * 1000:.0f}ms  (min {min(tot) * 1000:.0f} / max {max(tot) * 1000:.0f})")
        print(f"    분해: {parts} · 나머지(재정렬 등) {rest * 1000:.0f}ms")

    print("\n※ 답변 생성(Gemini)은 별도로 ~1.5s가 더 붙는다 — 음성 UX 목표는 총 3s 이내.")


if __name__ == "__main__":
    main()
