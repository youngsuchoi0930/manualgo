"""임베딩 — Gemini 임베딩 API로 텍스트를 벡터화한다 (로컬 torch 불필요).

Windows에서 sentence-transformers/torch import가 무음 크래시하는 문제를 우회하기 위해
임베딩을 Gemini API로 처리한다. 검색 품질을 위해 문서/질의를 task_type으로 구분하고,
cosine 검색을 위해 L2 정규화한다. 키는 GOOGLE_API_KEY 환경변수에서 읽는다.

선택적 디스크 캐시(EMBED_CACHE=1 또는 cache=True): 같은 (모델,차원,task_type,텍스트)는
한 번만 API를 부른다. 여러 검색기가 같은 평가셋 쿼리를 반복 임베딩할 때 무상 쿼터(1000/day)를
아끼고, 라이브에서 반복 질문을 즉시 처리한다. 키별 개별 파일이라 다중 프로세스에도 안전하다.

(로컬 BGE-M3 + Reranker 파이프라인은 W4–5에서 Windows 네이티브 문제를 해결하면 복귀시킨다.)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIM = 768
CACHE_DIR = os.environ.get("EMBED_CACHE_DIR", "data/cache/emb")


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        dim: int = DEFAULT_DIM,
        api_key: str | None = None,
        cache: bool | None = None,  # None이면 EMBED_CACHE 환경변수(=1)로 결정
    ) -> None:
        from google import genai

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다 (.env 확인).")
        self.model = model or os.environ.get("GEMINI_EMBED_MODEL", DEFAULT_MODEL)
        self.dim = dim
        self.cache = (os.environ.get("EMBED_CACHE") == "1") if cache is None else cache
        self._client = genai.Client(api_key=api_key)

    def _cache_key(self, task_type: str, text: str) -> str:
        raw = f"{self.model}|{self.dim}|{task_type}|{text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _cache_get(self, key: str):
        p = os.path.join(CACHE_DIR, key + ".json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _cache_put(self, key: str, vec: list[float]) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, key + ".json"), "w", encoding="utf-8") as f:
            json.dump(vec, f)

    def encode(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",  # 질의는 "RETRIEVAL_QUERY"
        batch_size: int = 50,
    ) -> list[list[float]]:
        """텍스트 리스트를 정규화된 임베딩 벡터 리스트로 변환한다. 캐시 켜지면 히트는 API를 건너뛴다."""
        texts = list(texts)
        if not self.cache:
            return self._encode_raw(texts, task_type, batch_size)

        results: list[list[float] | None] = [None] * len(texts)
        miss_idx, miss_txt = [], []
        for i, t in enumerate(texts):
            v = self._cache_get(self._cache_key(task_type, t))
            if v is not None:
                results[i] = v
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if miss_txt:
            got = self._encode_raw(miss_txt, task_type, batch_size)
            for j, i in enumerate(miss_idx):
                results[i] = got[j]
                self._cache_put(self._cache_key(task_type, texts[i]), got[j])
        return results  # type: ignore[return-value]

    def _encode_raw(self, texts: list[str], task_type: str, batch_size: int) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._embed_with_retry(types, batch, task_type)
            out.extend(_l2_normalize(e.values) for e in resp.embeddings)
        return out

    def _embed_with_retry(self, types, batch: list[str], task_type: str, attempts: int = 5):
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dim,
                    ),
                )
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(20)  # 분당 한도 리셋 대기
                elif "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(min(60, 5 * (2**attempt)))
                else:
                    raise
        raise RuntimeError(f"임베딩 실패(재시도 소진): {last_err}")
