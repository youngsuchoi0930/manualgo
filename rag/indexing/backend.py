"""임베딩 백엔드 선택 — 임베더와 컬렉션 이름을 **항상 짝으로** 고정한다.

임베딩 모델이 다르면 벡터 공간도 다르다. 임베더만 바꾸고 컬렉션을 그대로 쓰면
검색이 조용히 엉뚱한 결과를 낸다(에러도 안 난다). 그래서 모델↔컬렉션 대응을
여기 한 곳에만 둔다.

  EMBED_BACKEND=onnx    → 로컬 BGE-M3 ONNX (1024d, 쿼터 0)      · manuals_bgem3_onnx
  EMBED_BACKEND=gemini  → Gemini API (768d, 무료 1000건/일)     · manuals_gemini

기본값은 gemini(기존 인덱스)다. onnx 인덱스를 다 쌓고 검증한 뒤 기본값을 바꾼다.
"""
from __future__ import annotations

import os

_COLLECTIONS = {
    "gemini": "manuals_gemini",
    "onnx": "manuals_bgem3_onnx",
}
DEFAULT = "gemini"


def backend_name() -> str:
    name = (os.environ.get("EMBED_BACKEND") or DEFAULT).strip().lower()
    if name not in _COLLECTIONS:
        raise ValueError(f"EMBED_BACKEND는 {sorted(_COLLECTIONS)} 중 하나여야 합니다 (받은 값: {name!r})")
    return name


def collection_name() -> str:
    """현재 백엔드에 대응하는 Chroma 컬렉션 이름."""
    return _COLLECTIONS[backend_name()]


def make_embedder():
    """현재 백엔드의 임베더 인스턴스. 둘 다 encode(texts, task_type=...) 인터페이스."""
    if backend_name() == "onnx":
        from rag.indexing.onnx_embedder import OnnxEmbedder

        return OnnxEmbedder()
    from rag.indexing.embedder import Embedder

    return Embedder()
