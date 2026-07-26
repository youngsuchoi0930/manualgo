"""인덱스 진단 — 저장소에 데이터가 있는지, 검색이 결과를 내는지 단계별로 확인한다.

실행: ``python scripts/check_index.py``
"""
from __future__ import annotations


def main() -> None:
    from rag.indexing.backend import backend_name, collection_name
    from rag.vectorstore.store import ChromaStore

    store = ChromaStore()
    print(f"[0] 백엔드={backend_name()} · 컬렉션={collection_name()}")
    print("[1] store.count() =", store.count())

    # 저장된 청크 일부 들여다보기
    peek = store._col.peek(3)
    print("[2] peek ids   :", peek.get("ids"))
    print("    peek pages :", [(m or {}).get("page") for m in (peek.get("metadatas") or [])])
    docs = peek.get("documents") or []
    if docs:
        print("    peek doc[0]:", (docs[0] or "")[:80])

    # 임베딩 + 검색 직접 호출 — 임베더는 컬렉션과 짝이 맞아야 한다(백엔드가 정함).
    # Embedder를 직접 쓰면 onnx 컬렉션(1024d)에 768d 질의를 던져 정상 인덱스를 고장으로 오진한다.
    from rag.indexing.backend import make_embedder

    emb = make_embedder()
    qv = emb.encode(["예약은 최대 몇 시간까지 돼?"])[0]
    print("[3] query vec dim =", len(qv))

    hits = store.search(qv, top_k=3)
    print("[4] hits =", len(hits))
    for h in hits:
        print(f"    {h['page']}p score={h['score']:.3f} | {(h['text'] or '')[:60]}")


if __name__ == "__main__":
    main()
