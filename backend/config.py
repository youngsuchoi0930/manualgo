"""앱 환경설정 — .env 값을 읽어 타입 안전한 설정 객체로 노출한다."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # 임베딩 / Reranker
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # 벡터 DB / 검색
    vector_store: str = "chroma"
    vector_index_dir: str = "data/index"
    retriever_top_k: int = 5
    hybrid_bm25_weight: float = 0.5
    rerank_top_n: int = 20

    # STT / TTS
    stt_provider: str = "whisper"
    whisper_model: str = "large-v3"
    tts_provider: str = "clova"


@lru_cache
def get_settings() -> Settings:
    return Settings()
