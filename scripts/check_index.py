"""인덱스 진단 — 저장소에 데이터가 있는지, 검색이 결과를 내는지 단계별로 확인한다.

실행: ``python scripts/check_index.py``
"""
from __future__ import annotations


def main() -> None:
    from rag.vectorstore.store import ChromaStore

    store = ChromaStore()
    print("[1] store.count() =", store.count())

    # 저장된 청크 일부 들여다보기
    peek = store._col.peek(3)
    print("[2] peek ids   :", peek.get("ids"))
    print("    peek pages :", [(m or {}).get("page") for m in (peek.get("metadatas") or [])])
    docs = peek.get("documents") or []
    if docs:
        print("    peek doc[0]:", (docs[0] or "")[:80])

    # 임베딩 + 검색 직접 호출
    from rag.indexing.embedder import Embedder

    emb = Embedder()
    qv = emb.encode(["예약은 최대 몇 시간까지 돼?"])[0]
    print("[3] query vec dim =", len(qv))

    hits = store.search(qv, top_k=3)
    print("[4] hits =", len(hits))
    for h in hits:
        print(f"    {h['page']}p score={h['score']:.3f} | {(h['text'] or '')[:60]}")


if __name__ == "__main__":
    main()
