"""FastAPI 진입점 — 실시간 추론 파이프라인의 HTTP 엔드포인트를 노출한다.

실행: ``uvicorn backend.main:app --reload``
"""
from fastapi import FastAPI

from backend.api import routes_ask, routes_health

app = FastAPI(title="매뉴얼 음성 도우미", version="0.1.0")

app.include_router(routes_health.router)
app.include_router(routes_ask.router)
