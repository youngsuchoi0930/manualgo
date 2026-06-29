"""임베딩 — Gemini 임베딩 API로 텍스트를 벡터화한다 (로컬 torch 불필요).

Windows에서 sentence-transformers/torch import가 무음 크래시하는 문제를 우회하기 위해
임베딩을 Gemini API로 처리한다. 검색 품질을 위해 문서/질의를 task_type으로 구분하고,
cosine 검색을 위해 L2 정규화한다. 키는 GOOGLE_API_KEY 환경변수에서 읽는다.

(로컬 BGE-M3 + Reranker 파이프라인은 W4–5에서 Windows 네이티브 문제를 해결하면 복귀시킨다.)
"""
from __future__ import annotations

import math
import os

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIM = 768


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class Embedder:
    def __init__(self, model: str | None = None, dim: int = DEFAULT_DIM, api_key: str | None = None) -> None:
        from google import genai

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다 (.env 확인).")
        self.model = model or os.environ.get("GEMINI_EMBED_MODEL", DEFAULT_MODEL)
        self.dim = dim
        self._client = genai.Client(api_key=api_key)

    def encode(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",  # 질의는 "RETRIEVAL_QUERY"
        batch_size: int = 50,
    ) -> list[list[float]]:
        """텍스트 리스트를 정규화된 임베딩 벡터 리스트로 변환한다."""
        from google.genai import types

        texts = list(texts)
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dim,
                ),
            )
            out.extend(_l2_normalize(e.values) for e in resp.embeddings)
        return out
