"""로컬 임베딩 — BGE-M3를 ONNX(onnxruntime)로 실행해 API 쿼터를 없앤다.

배경: Gemini 무료 임베딩은 **하루 1000건**(배치를 써도 텍스트 1건=1건으로 과금)이라
6,375청크 코퍼스를 인덱싱하는 데 6일이 걸린다. 리랭커에서 ONNX가 이 머신의
torch 무음 크래시를 우회한다는 걸 확인했으므로, 임베딩도 같은 경로로 로컬화한다.
→ 쿼터 0, 코퍼스 크기 제약 없음, 오프라인 재현 가능.

BGE-M3 dense 규약: CLS 토큰(last_hidden_state[:, 0])을 L2 정규화한 1024차원 벡터.
질의/문서 프리픽스가 필요 없다(대칭 모델). 그래서 Embedder와 달리 task_type은
호환을 위해 받기만 하고 무시한다 — 같은 텍스트는 항상 같은 벡터가 나온다.
"""
from __future__ import annotations

import os

_REPO = "onnx-community/bge-m3-ONNX"
DIM = 1024


def _open_session(repo: str, quantized: bool, gpu: bool, opts, provs):
    """후보 가중치를 순서대로 **받아서 실제로 열어보고** 성공한 것을 쓴다.

    다운로드 성공만 보고 넘기면 안 된다 — 저장소에 손상된 변형이 섞여 있다
    (예: bge-m3의 model_fp16.onnx는 opset 정보가 없어 로드가 실패한다).
    GPU면 fp16, CPU면 int8을 우선한다(int8은 GPU에서 미지원 연산자가 많다).
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    from rag.onnx_runtime import weight_candidates

    names = weight_candidates(gpu, "embed") if quantized else ["onnx/model.onnx"]
    errors: list[str] = []
    for f in names:
        try:
            path = hf_hub_download(repo, f)
        except Exception as e:
            errors.append(f"{f}: 다운로드 실패 {type(e).__name__}")
            continue
        # fp32 model.onnx은 가중치가 외부(model.onnx_data)로 분리된 경우가 있다
        for side in (f + "_data", f + ".data"):
            try:
                hf_hub_download(repo, side)
            except Exception:
                pass
        try:
            sess = ort.InferenceSession(path, sess_options=opts, providers=provs)
        except Exception as e:
            errors.append(f"{f}: 로드 실패 {str(e)[:80]}")
            continue
        return sess, os.path.basename(path)
    raise RuntimeError("사용 가능한 ONNX 임베딩 가중치가 없습니다:\n  " + "\n  ".join(errors))


class OnnxEmbedder:
    """BGE-M3 ONNX 임베더. Embedder와 같은 encode() 인터페이스(드롭인 교체용)."""

    def __init__(
        self,
        repo: str = _REPO,
        quantized: bool = True,
        max_length: int = 512,
        threads: int | None = None,
        cache: bool | None = None,  # None이면 EMBED_CACHE 환경변수(=1)로 결정
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model = repo
        self.dim = DIM
        self.max_length = max_length
        self._cache = (os.environ.get("EMBED_CACHE") == "1") if cache is None else cache
        self._cache_dir = os.environ.get("EMBED_CACHE_DIR_ONNX", "data/cache/emb_onnx")

        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        tok.enable_truncation(max_length=max_length)
        tok.enable_padding()  # 배치 내 최장 길이에 맞춰 패딩
        self._tok = tok

        from rag.onnx_runtime import on_gpu, providers

        provs = providers("embed")
        gpu = on_gpu(provs)
        opts = ort.SessionOptions()
        if not gpu:
            opts.intra_op_num_threads = threads or (os.cpu_count() or 4)
        # int8/fp16 등 실제로 로드된 파일명을 남긴다 — 캐시 키에 넣어 양자화 혼합 방지
        self._sess, self.weights = _open_session(repo, quantized, gpu, opts, provs)
        self.provider = self._sess.get_providers()[0]
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._out_names = [o.name for o in self._sess.get_outputs()]

    def encode(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",  # 호환용 — BGE-M3는 대칭이라 사용하지 않음
        batch_size: int = 8,
    ) -> list[list[float]]:
        """텍스트를 임베딩한다. EMBED_CACHE=1이면 디스크 캐시를 쓴다.

        로컬이라 쿼터는 없지만 6,000청크 인덱싱이 ~1시간이라, 중단 후 재실행이
        캐시 히트로 즉시 끝나도록 캐시를 둔다. 키에 모델·차원이 들어가 백엔드끼리 섞이지 않는다.
        """
        texts = list(texts)
        if not self._cache:
            return self._encode_raw(texts, batch_size)

        results: list[list[float] | None] = [None] * len(texts)
        miss_idx, miss_txt = [], []
        for i, t in enumerate(texts):
            v = self._cache_get(self._cache_key(t))
            if v is not None:
                results[i] = v
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if miss_txt:
            got = self._encode_raw(miss_txt, batch_size)
            for j, i in enumerate(miss_idx):
                results[i] = got[j]
                self._cache_put(self._cache_key(texts[i]), got[j])
        return results  # type: ignore[return-value]

    def _cache_key(self, text: str) -> str:
        import hashlib

        # 가중치 종류(int8/fp16)까지 키에 넣는다 — 양자화가 다르면 벡터도 미세하게 달라서
        # 한 인덱스에 섞이면 안 된다(조용한 품질 저하).
        raw = f"onnx|{self.model}|{self.weights}|{self.dim}|{self.max_length}|{text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _cache_get(self, key: str):
        import json

        p = os.path.join(self._cache_dir, key + ".json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _cache_put(self, key: str, vec: list[float]) -> None:
        import json

        os.makedirs(self._cache_dir, exist_ok=True)
        with open(os.path.join(self._cache_dir, key + ".json"), "w", encoding="utf-8") as f:
            json.dump(vec, f)

    def _encode_raw(self, texts: list[str], batch_size: int) -> list[list[float]]:
        import numpy as np

        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            encs = self._tok.encode_batch(batch)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            attn = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": attn}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            res = self._sess.run(None, feeds)
            vecs = self._dense(res, attn)
            out.extend(vecs.tolist())
        return out

    def _dense(self, outputs: list, attn) -> "object":
        """모델 출력에서 dense 벡터를 뽑아 L2 정규화한다.

        BGE-M3는 CLS 풀링이다. 출력이 (B, T, H)면 CLS를, 이미 (B, H)면 그대로 쓴다.
        """
        import numpy as np

        arr = None
        for o in outputs:  # 3D(last_hidden_state) 우선, 없으면 2D(sentence_embedding)
            a = np.asarray(o)
            if a.ndim == 3:
                arr = a[:, 0, :]
                break
            if a.ndim == 2 and arr is None:
                arr = a
        if arr is None:
            raise RuntimeError(f"dense 출력을 찾지 못했습니다: {[np.asarray(o).shape for o in outputs]}")
        arr = arr.astype(np.float32)
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norm, 1e-12)
