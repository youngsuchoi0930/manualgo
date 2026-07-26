"""ONNX 실행 provider 선택 — CPU / CUDA / DirectML을 한 곳에서 정한다.

임베더와 리랭커가 같은 규칙을 쓰도록 공통화한다. 설치된 onnxruntime이 CPU 전용이면
자동으로 CPU로 떨어지므로, GPU 런타임이 없어도 코드는 그대로 동작한다.

  ONNX_PROVIDER=auto(기본) → CUDA > DirectML > CPU 순으로 가능한 것 사용
  ONNX_PROVIDER=cpu|cuda|dml → 강제 지정
  ONNX_PROVIDER_EMBED / ONNX_PROVIDER_RERANK → 컴포넌트별 덮어쓰기(VRAM 분배용)

가중치 선택도 provider에 맞춘다: GPU에서는 int8 양자화가 미지원/저속인 연산자가 많아
fp16이 유리하고, CPU에서는 int8이 가장 빠르다.
"""
from __future__ import annotations

import os

_PREF = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def providers(kind: str | None = None) -> list[str]:
    """이 환경에서 실제로 쓸 provider 목록 (앞이 우선). 항상 CPU를 마지막 폴백으로 둔다.

    kind("embed"/"rerank")를 주면 ONNX_PROVIDER_EMBED / ONNX_PROVIDER_RERANK를 먼저 본다.
    임베더와 리랭커를 동시에 GPU에 올리면 6GB VRAM에 여유가 거의 없어(실측 5.8/6.1GB),
    한쪽만 GPU로 두고 다른 쪽을 CPU로 내릴 수 있어야 한다.
    """
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    want = os.environ.get(f"ONNX_PROVIDER_{kind.upper()}") if kind else None
    want = (want or os.environ.get("ONNX_PROVIDER") or "auto").strip().lower()
    if want != "auto":
        if want not in _PREF:
            raise ValueError(f"ONNX_PROVIDER는 auto/{'/'.join(_PREF)} 중 하나여야 합니다 (받은 값: {want!r})")
        name = _PREF[want]
        if name not in have:
            raise RuntimeError(
                f"{name}를 쓸 수 없습니다. 설치된 onnxruntime provider: {sorted(have)}\n"
                f"  CUDA: pip install onnxruntime-gpu / DirectML: pip install onnxruntime-directml"
            )
        return [name] if name == "CPUExecutionProvider" else [name, "CPUExecutionProvider"]

    for key in ("cuda", "dml"):
        if _PREF[key] in have:
            return [_PREF[key], "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def on_gpu(provs: list[str] | None = None) -> bool:
    provs = provs or providers()
    return bool(provs) and provs[0] != "CPUExecutionProvider"


def weight_candidates(gpu: bool, kind: str | None = None) -> list[str]:
    """provider에 맞는 ONNX 가중치 파일 우선순위 (onnx-community 저장소 관례 기준).

    ONNX_WEIGHTS(_EMBED/_RERANK)로 특정 파일을 고정할 수 있다. 인덱스를 fp32로 만들었다면
    질의 임베딩도 fp32여야 한다 — provider를 CPU로 내리면 기본값이 int8로 바뀌어
    양자화가 어긋나므로(캐시 키도 달라짐) 그때 고정이 필요하다.

    GPU에서 fp32(model.onnx)를 먼저 쓴다 — 실측 근거:
      이 머신(GTX 1660 SUPER, Turing TU116)은 **텐서코어가 없어** fp16 이득이 없고,
      DirectML의 fp16 경로가 오히려 느렸다. bge-m3 임베더 실측 24.7청크/초(fp32) vs
      리랭커 fp16 3.0seq/초. int8은 GPU에서 미지원 연산자가 많아 마지막.
    텐서코어가 있는 GPU(RTX 계열)로 옮기면 fp16을 먼저 시도하도록 되돌릴 가치가 있다.
    """
    if gpu:
        order = ["onnx/model.onnx", "onnx/model_fp16.onnx", "onnx/model_quantized.onnx"]
    else:
        order = ["onnx/model_int8.onnx", "onnx/model_quantized.onnx", "onnx/model.onnx"]

    pin = os.environ.get(f"ONNX_WEIGHTS_{kind.upper()}") if kind else None
    pin = (pin or os.environ.get("ONNX_WEIGHTS") or "").strip()
    if pin:
        if not pin.startswith("onnx/"):
            pin = f"onnx/{pin}"
        order = [pin] + [f for f in order if f != pin]  # 고정본을 먼저, 실패 시 기본 순서로 폴백
    return order
