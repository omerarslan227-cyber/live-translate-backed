from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float | None


class SpeechToText:
    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    async def transcribe(self, audio_bytes: bytes, *, source_lang: str | None) -> TranscriptionResult:
        if not audio_bytes:
            return TranscriptionResult(text="", language=None, duration=None)

        model = await self._get_model()
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            def run() -> TranscriptionResult:
                segments, info = model.transcribe(
                    tmp_path,
                    language=_whisper_language(source_lang),
                    vad_filter=True,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.62,
                    compression_ratio_threshold=2.4,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                return TranscriptionResult(
                    text=text,
                    language=getattr(info, "language", None),
                    duration=getattr(info, "duration", None),
                )

            return await asyncio.to_thread(run)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _get_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    WhisperModel,
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
        return self._model


def _whisper_language(value: str | None) -> str | None:
    if not value:
        return None
    lang = value.upper().split("-")[0]
    return {
        "TR": "tr",
        "RU": "ru",
        "UK": "uk",
        "EN": "en",
        "KA": "ka",
    }.get(lang)
