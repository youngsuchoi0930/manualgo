"""TTS — 답변 텍스트를 음성으로 합성한다. 기본: Azure Speech (REST).

SDK 대신 REST API를 사용한다 — 의존성이 requests뿐이라 가볍고,
이 프로젝트의 Windows 환경에서 네이티브 SDK 리스크가 없다.
키/리전은 .env의 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 에서 읽는다.
무료 티어(F0): 월 50만 자.
"""
from __future__ import annotations

import html
import os
from typing import Protocol


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class AzureTTS:
    """Azure Speech REST TTS. 한국어 뉴럴 보이스, MP3 반환."""

    def __init__(
        self,
        voice: str = "ko-KR-SunHiNeural",
        output_format: str = "audio-24khz-48kbitrate-mono-mp3",
        key: str | None = None,
        region: str | None = None,
    ) -> None:
        self.key = key or os.environ.get("AZURE_SPEECH_KEY")
        self.region = region or os.environ.get("AZURE_SPEECH_REGION")
        if not self.key or not self.region:
            raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION이 설정되지 않았습니다 (.env 확인).")
        self.voice = voice
        self.output_format = output_format
        self._url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def synthesize(self, text: str) -> bytes:
        import requests

        ssml = (
            f"<speak version='1.0' xml:lang='ko-KR'>"
            f"<voice name='{self.voice}'>{html.escape(text)}</voice>"
            f"</speak>"
        )
        resp = requests.post(
            self._url,
            headers={
                "Ocp-Apim-Subscription-Key": self.key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": self.output_format,
                "User-Agent": "manualgo",
            },
            data=ssml.encode("utf-8"),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
