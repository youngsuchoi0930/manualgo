"""Cross-Encoder Reranker — ONNX(onnxruntime)로 실행해 torch 크래시를 우회한다.

BGE-reranker-v2-m3(다국어, 한국어 강함)의 사전 export ONNX(onnx-community)를 받아
(query, chunk) 쌍을 함께 스코어링해 Hybrid 상위 후보를 재정렬한다.
- torch/sentence-transformers 불필요 (이 Windows 머신에서 그 스택은 import만 해도 크래시).
- 토큰화는 tokenizers(Rust) 직접 사용, 추론은 onnxruntime(CPU). 모델은 첫 실행 시 HF에서 캐시.
"""
from __future__ import annotations

from rag.retrieval.base import RetrievedChunk

_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"


def _download_onnx(repo: str, prefer_quantized: bool):
    from huggingface_hub import hf_hub_download

    names = (
        ["onnx/model_quantized.onnx", "onnx/model_int8.onnx", "onnx/model.onnx"]
        if prefer_quantized
        else ["onnx/model.onnx", "onnx/model_quantized.onnx"]
    )
    last = None
    for f in names:
        try:
            return hf_hub_download(repo, f)
        except Exception as e:
            last = e
    raise RuntimeError(f"ONNX 모델 파일을 찾지 못했습니다: {last}")


class Reranker:
    """ONNX 크로스인코더 리랭커. quantized=True면 int8(CPU 빠름)."""

    def __init__(self, repo: str = _REPO, quantized: bool = True, max_length: int = 384) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model_name = repo
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        tok.enable_truncation(max_length=max_length)
        self._tok = tok
        self._sess = ort.InferenceSession(
            _download_onnx(repo, quantized), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def scores(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        encs = [self._tok.encode(query, p) for p in passages]  # (query, passage) 쌍 인코딩
        maxlen = max((len(e.ids) for e in encs), default=1)
        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        attn = np.zeros((len(encs), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            attn[i, : len(e.attention_mask)] = e.attention_mask
        feeds = {"input_ids": ids, "attention_mask": attn}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        logits = self._sess.run(None, feeds)[0]
        return np.asarray(logits).reshape(-1).astype(float).tolist()

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []
        sc = self.scores(query, [c.text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: sc[i], reverse=True)[:top_n]
        return [
            RetrievedChunk(
                chunk_id=candidates[i].chunk_id,
                manual_id=candidates[i].manual_id,
                page=candidates[i].page,
                section=candidates[i].section,
                text=candidates[i].text,
                score=float(sc[i]),
            )
            for i in order
        ]
