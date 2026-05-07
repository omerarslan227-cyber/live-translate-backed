from __future__ import annotations

import base64
import binascii
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .rooms import RoomManager
from .speech import SpeechToText
from .translation import Translator

settings = get_settings()
rooms = RoomManager()
speech = SpeechToText(
    settings.whisper_model_size,
    settings.whisper_device,
    settings.whisper_compute_type,
)
translator = Translator(settings.deepl_api_key)

app = FastAPI(title="BridgeCall realtime backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "deeplConfigured": translator.configured,
        "whisperModel": settings.whisper_model_size,
    }


@app.websocket("/signal")
async def signal_socket(websocket: WebSocket) -> None:
    client = await rooms.connect(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "chat_message":
                try:
                    await _translate_chat_message(message)
                except Exception as exc:
                    await rooms.send_error(client, str(exc))
                    continue
            await rooms.handle(client, message)
    except WebSocketDisconnect:
        await rooms.disconnect(client)
    except Exception as exc:
        await rooms.send_error(client, str(exc))
        await rooms.disconnect(client)


@app.websocket("/translate")
async def translate_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            try:
                payload = await _translate_payload(message)
                await websocket.send_json(payload)
            except Exception as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
    except WebSocketDisconnect:
        return


async def _translate_payload(message: dict[str, Any]) -> dict[str, Any]:
    source_lang = _empty_to_none(message.get("sourceLang"))
    target_lang = _empty_to_none(message.get("targetLang"))
    audio_b64 = str(message.get("audio") or "")
    if not audio_b64:
        text = str(message.get("text") or "").strip()
        if not text:
            return {"stage": "partial", "original": "", "translated": ""}
        translated = await translator.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        return {"stage": "final", "original": text, "translated": translated.text}

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 audio") from exc

    max_bytes = settings.bridgecall_max_audio_seconds * 16000 * 2 + 128000
    if len(audio_bytes) > max_bytes:
        raise ValueError("Audio chunk is too large")

    transcription = await speech.transcribe(audio_bytes, source_lang=source_lang)
    original = _clean_text(transcription.text)
    if not original:
        return {"stage": "partial", "original": "", "translated": ""}

    translated = await translator.translate_text(
        original,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    stage = "final" if _looks_complete(original) else "partial"
    return {
        "stage": stage,
        "original": original,
        "translated": _clean_text(translated.text),
        "detectedSourceLang": transcription.language or translated.detected_source_lang,
    }


def _empty_to_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _looks_complete(value: str) -> bool:
    text = value.strip()
    return bool(text) and (text.endswith((".", "!", "?")) or len(text) >= 86)


async def _translate_chat_message(message: dict[str, Any]) -> None:
    text = str(message.get("text") or "").strip()
    if not text or message.get("translatedText"):
        return
    translated = await translator.translate_text(
        text,
        source_lang=_empty_to_none(message.get("sourceLang")),
        target_lang=_empty_to_none(message.get("targetLang")),
    )
    message["translatedText"] = translated.text
