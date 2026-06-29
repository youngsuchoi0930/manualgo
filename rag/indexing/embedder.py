"""임베딩 — 한국어 특화 모델로 청크/질문을 벡터화한다 (예: BGE-M3)."""
from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        # TODO: sentence-transformers / FlagEmbedding 로 모델 로드
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
