"""전체 파이프라인 진단/실행 — 검색 → Gemini 근거 기반 답변.

각 단계를 flush하며 찍고, 에러는 전체 트레이스백으로 드러낸다.
실행: python scripts/ask.py  또는  python scripts/ask.py "질문"
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 어느 위치/셸에서 실행하든 rag를 찾도록 프로젝트 루트를 import 경로에 추가하고,
# torch/chromadb import 전에 OpenMP 충돌 회피 플래그를 켠다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    # .env 명시적 로드 (rag import에서도 로드되지만, 키 확인을 먼저 하려고 직접 호출)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print("python-dotenv 미설치", flush=True)

    question = sys.argv[1] if len(sys.argv) > 1 else "예약은 최대 몇 시간까지 돼?"
    print(f"[질문] {question}", flush=True)

    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        print(f"[키] 로드됨 — 끝 4자리 …{key[-4:]} (길이 {len(key)})", flush=True)
    else:
        print("[키] ✗ GOOGLE_API_KEY 없음 → .env 파일명/위치/형식 확인 필요", flush=True)

    print("[1] 검색기 로드 + 검색...", flush=True)
    from rag.retrieval.naive import NaiveRetriever

    hits = NaiveRetriever().retrieve(question, top_k=3)
    print(f"    hits={len(hits)}", flush=True)
    for c in hits:
        print(f"    {c.page}p score={c.score:.3f} | {(c.text or '')[:50]}", flush=True)

    print("[2] Gemini 답변 생성...", flush=True)
    try:
        from rag.generation.generator import AnswerGenerator

        out = AnswerGenerator().generate(question, hits)
        print("\n[답변]", out["answer"], flush=True)
        print("[출처]", [(s["manual_id"], s["page"]) for s in out["sources"]], flush=True)
    except Exception:
        print("[2] ✗ 생성 단계 에러:", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
