"""rag 패키지.

import 시점에:
1) Windows OpenMP 충돌 회피 플래그 설정 (torch ↔ onnxruntime/chromadb)
2) 프로젝트 루트의 .env 로드 (GOOGLE_API_KEY 등)
"""
import os

# torch와 onnxruntime(chromadb 기본 임베딩 함수) 등이 각자 OpenMP(libiomp) 런타임을
# 로드하면서 Windows에서 프로세스가 트레이스백 없이 조용히 죽는 문제를 막는다.
# 반드시 torch/chromadb import 전에 설정되어야 하므로 패키지 최상단에 둔다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
