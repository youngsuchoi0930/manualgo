"""FastAPI 진입점 — 추론 API + 프론트엔드 정적 서빙.

실행: ``uvicorn backend.main:app --reload``
접속: http://localhost:8000  (프론트) / http://localhost:8000/docs (API 문서)
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()  # AZURE_SPEECH_* / GOOGLE_API_KEY 등 — 라우터 import 전에 로드

from backend.api import routes_ask, routes_health  # noqa: E402

app = FastAPI(title="매뉴얼 음성 도우미", version="0.1.0")

app.include_router(routes_health.router)
app.include_router(routes_ask.router)

# 프론트엔드 정적 서빙 (같은 오리진 → CORS 불필요). 라우터 뒤에 마운트해야 API가 우선된다.
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
