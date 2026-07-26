"""벡터 DB 래퍼 — 공통 인터페이스(Protocol)와 ChromaDB 구현.

ChromaDB를 cosine 공간으로 영속화한다. 임베딩은 외부(Embedder)에서 계산해 주입한다.
나중에 FAISS 구현을 같은 인터페이스로 추가할 수 있다.
"""
from __future__ import annotations

from typing import Protocol

# 컬렉션 이름은 임베딩 백엔드가 정한다 (모델별 벡터 공간이 달라 섞으면 안 됨).
# 하위 호환용 별칭: 예전 코드가 store.COLLECTION을 참조한다.
from rag.indexing.backend import collection_name  # noqa: E402

COLLECTION = collection_name()


class VectorStore(Protocol):
    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None: ...
    def search(self, query_vector: list[float], top_k: int) -> list[dict]: ...


class ChromaStore:
    """ChromaDB 영속 저장소 (cosine)."""

    def __init__(self, persist_dir: str = "data/index", collection: str | None = None) -> None:
        import chromadb

        self._name = collection or collection_name()  # 호출 시점의 백엔드를 따른다
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """컬렉션을 비우고 새로 만든다 (폴더 전체로 재구축할 때 사용)."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        """레코드를 추가한다. Chroma는 add() 1회당 상한(≈5,461)이 있어 알아서 나눠 넣는다."""
        try:
            limit = int(self._client.get_max_batch_size())
        except Exception:
            limit = 5000
        limit = max(1, limit)
        for i in range(0, len(ids), limit):
            j = i + limit
            self._col.add(
                ids=ids[i:j],
                embeddings=vectors[i:j],
                metadatas=metadatas[i:j],
                documents=documents[i:j],
            )

    def stamp_model(self, model: str, dim: int, weights: str | None = None) -> None:
        """이 컬렉션을 만든 임베딩 모델·차원·가중치 변형을 컬렉션 메타데이터에 남긴다.

        modify()에 hnsw:* 설정을 다시 넘기면 Chroma가 '거리함수 변경 불가'로 거부하므로
        우리 키만 남긴다(거리함수는 생성 시 정해지고 그대로 유지된다).
        """
        md = {k: v for k, v in (self._col.metadata or {}).items() if not k.startswith("hnsw:")}
        md.update({"embed_model": str(model), "embed_dim": int(dim)})
        if weights:
            md["embed_weights"] = str(weights)
        self._col.modify(metadata=md)

    def check_model(self, model: str, dim: int, weights: str | None = None) -> str | None:
        """기록된 임베딩 구성과 지금 쓰려는 것이 다르면 경고 문구를 반환한다 (같으면 None).

        컬렉션 이름은 백엔드 단위라, 같은 백엔드에서 모델만 바꾸면(예: GEMINI_EMBED_MODEL,
        또는 int8→fp32 가중치 교체) 이름은 그대로다. 그때 벡터공간이 어긋나는 것을 여기서 잡는다.
        양자화가 다르면 같은 모델이어도 벡터가 미세하게 달라 검색 품질이 조용히 떨어진다.
        """
        md = self._col.metadata or {}
        was_model, was_dim = md.get("embed_model"), md.get("embed_dim")
        if was_model is None and was_dim is None:
            return None  # 예전 인덱스 — 기록이 없음
        if str(was_model) != str(model) or int(was_dim or 0) != int(dim):
            return (f"컬렉션 '{self._name}'은 {was_model}({was_dim}d)로 만들어졌는데 "
                    f"지금 {model}({dim}d)를 쓰려 합니다 — 재인덱싱이 필요합니다")
        was_w = md.get("embed_weights")
        if weights and was_w and str(was_w) != str(weights):
            return (f"컬렉션 '{self._name}'은 {was_w}로 임베딩됐는데 지금 {weights}를 씁니다 "
                    f"— 양자화가 달라 검색이 미세하게 어긋납니다(재인덱싱 권장)")
        return None

    def replace_all(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> int:
        """전체 재구축을 **안전하게** 한다 — 임시 컬렉션에 다 넣은 뒤 이름을 교체한다.

        reset() 후 add()가 실패하면 컬렉션이 빈 채로 남아 기존 인덱스를 잃는다.
        그래서 쓰기가 모두 성공한 다음에만 기존 것을 지우고 임시본을 승격시킨다.
        """
        tmp_name = f"{self._name}__building"
        try:
            self._client.delete_collection(tmp_name)  # 이전 시도의 잔여물 정리
        except Exception:
            pass
        tmp = self._client.get_or_create_collection(name=tmp_name, metadata={"hnsw:space": "cosine"})
        saved = self._col
        self._col = tmp
        try:
            self.add(ids=ids, vectors=vectors, metadatas=metadatas, documents=documents)
        except Exception:
            self._col = saved  # 기존 컬렉션은 손대지 않았다
            try:
                self._client.delete_collection(tmp_name)
            except Exception:
                pass
            raise
        # 여기까지 왔으면 신규 데이터가 전부 들어갔다 → 이제서야 교체
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        tmp.modify(name=self._name)
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
        return self._col.count()

    def search(self, query_vector: list[float], top_k: int = 5, manual_ids=None) -> list[dict]:
        kwargs = {}
        if manual_ids:
            kwargs["where"] = {"manual_id": {"$in": list(manual_ids)}}
        res = self._col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
            **kwargs,
        )
        hits: list[dict] = []
        for i in range(len(res["ids"][0])):
            md = res["metadatas"][0][i] or {}
            hits.append(
                {
                    "chunk_id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "score": 1.0 - res["distances"][0][i],  # cosine 거리 → 유사도
                    "manual_id": md.get("manual_id"),
                    "page": md.get("page"),
                    "section": md.get("section") or None,
                }
            )
        return hits

    def count(self) -> int:
        return self._col.count()
