"""STT — 음성을 텍스트로 변환한다 (Whisper 로컬 GPU 또는 CLOVA Speech)."""
from __future__ import annotations

from typing import Protocol


class STT(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class WhisperSTT:
    def __init__(self, model: str = "large-v3") -> None:
        self.model = model

    def transcribe(self, audio: bytes) -> str:
        raise NotImplementedError
