"""TTS — 답변 텍스트를 음성으로 합성한다 (CLOVA Voice 또는 XTTS 계열)."""
from __future__ import annotations

from typing import Protocol


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class ClovaTTS:
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError
