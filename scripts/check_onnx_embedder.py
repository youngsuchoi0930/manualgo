"""로컬 ONNX 임베더 점검 — 차원·판별력·속도를 인덱싱 전에 확인한다.

(1) 출력 차원, (2) 한국어 질문↔정답/무관 문장 유사도 순위, (3) 초당 청크 처리량.
첫 실행 시 ONNX 모델(bge-m3 int8, ~600MB)을 HF에서 캐시한다.
실행: conda run -n manual python scripts/check_onnx_embedder.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERY = "세탁기 예약은 최대 몇 시간까지 돼?"
PASSAGES = [
    "예약 버튼을 누르면 최대 24시간까지 예약 가능합니다",       # 정답
    "예약 세탁을 설정하려면 전원을 켠 후 예약 버튼을 누르세요",  # 관련
    "정수기 필터는 6개월마다 교체하세요",                        # 무관
    "전자레인지로 계란을 가열하면 터질 수 있습니다",              # 무관
]


def main() -> None:
    from rag.indexing.onnx_embedder import OnnxEmbedder

    t0 = time.perf_counter()
    emb = OnnxEmbedder()
    load = time.perf_counter() - t0
    print(f"[로드] {load:.1f}s  (model={emb.model}, weights={emb.weights}, "
          f"provider={emb.provider}, max_len={emb.max_length})")

    vecs = emb.encode([QUERY] + PASSAGES)
    dim = len(vecs[0])
    q, ps = vecs[0], vecs[1:]
    sims = [sum(a * b for a, b in zip(q, p)) for p in ps]  # 정규화됨 → 내적=cosine

    print(f"[차원] {dim}")
    print("[판별] 질문:", QUERY)
    order = sorted(range(len(ps)), key=lambda i: -sims[i])
    for rank, i in enumerate(order, 1):
        print(f"   {rank}위 {sims[i]:+.3f}  {PASSAGES[i][:44]}")
    ok = order[0] in (0, 1) and min(sims[0], sims[1]) > max(sims[2], sims[3])
    print(f"[판정] {'OK — 관련 문장이 무관 문장보다 상위' if ok else 'FAIL — 순위가 뒤섞임'}")

    # 처리량: 실제 청크 길이(~500자)에 가까운 더미로 측정
    dummy = ["세탁기 사용 설명. 예약 헹굼 탈수 코스 온도 조절 동작 일시정지 " * 8] * 16
    emb.encode(dummy[:4])  # 워밍업
    t = time.perf_counter()
    emb.encode(dummy)
    dt = time.perf_counter() - t
    rate = len(dummy) / dt
    print(f"[속도] {rate:.1f} 청크/초 → 6,375청크 ≈ {6375 / rate / 60:.0f}분")


if __name__ == "__main__":
    main()
