"""기존 Chroma에 저장된 OCR 텍스트를 Gemini 임베딩으로 다시 인덱싱한다 (재-OCR 없음).

torch/sentence-transformers를 전혀 쓰지 않으므로 Windows 네이티브 크래시를 피한다.
기존 "manuals"(BGE-M3 1024d) 컬렉션의 문서를 읽어 → Gemini로 임베딩 → "manuals_gemini"(768d)에 저장.

실행: python scripts/reindex_gemini.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    import chromadb

    from rag.indexing.embedder import Embedder
    from rag.vectorstore.store import COLLECTION  # "manuals_gemini"

    client = chromadb.PersistentClient(path="data/index")

    old = client.get_or_create_collection("manuals")  # 기존 OCR 텍스트가 들어있는 컬렉션
    data = old.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    if not ids:
        print("[!] 'manuals' 컬렉션이 비었습니다. 먼저 build_index로 OCR/인덱싱이 필요합니다.")
        return
    print(f"[*] 기존 청크 {len(ids)}개를 Gemini로 재임베딩...", flush=True)

    vectors = Embedder().encode(docs, task_type="RETRIEVAL_DOCUMENT")
    print(f"    임베딩 완료 — dim={len(vectors[0])}", flush=True)

    # 차원이 달라 기존과 섞이면 안 되므로 새 컬렉션을 깨끗이 재생성
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    new = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    new.add(ids=ids, embeddings=vectors, metadatas=metas, documents=docs)
    print(f"[완료] '{COLLECTION}' 컬렉션에 {new.count()}개 저장 (torch 미사용)", flush=True)


if __name__ == "__main__":
    main()
