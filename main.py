import os
import json
import base64
import tempfile
import time
import re
from collections import defaultdict, OrderedDict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import deepl
from faster_whisper import WhisperModel

app = FastAPI(title="Live Translate Backend")

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY", "")
PORT = int(os.getenv("PORT", "8000"))

translator = deepl.Translator(DEEPL_AUTH_KEY) if DEEPL_AUTH_KEY else None

# Model boyutu:
# base  = daha hızlı, biraz daha düşük doğruluk
# small = daha iyi doğruluk, yine iyi hız
# İlk önerim: small
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)

# room -> set[WebSocket]
signal_rooms: dict[str, set[WebSocket]] = defaultdict(set)

# -------------------------
# CACHE AYARLARI
# -------------------------
MAX_CACHE_ITEMS = 1000
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 saat
translation_cache = OrderedDict()


def normalize_source_lang(lang: str) -> str:
    if not lang:
        return "TR"
    lang = lang.upper()
    if lang == "EN-US":
        return "EN"
    return lang


def normalize_target_lang(lang: str) -> str:
    if not lang:
        return "RU"
    lang = lang.upper()
    if lang == "EN":
        return "EN-US"
    return lang


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?,:;]+$", "", text)
    return text


def make_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    normalized_text = normalize_text(text)
    return f"{source_lang}__{target_lang}__{normalized_text}"


def cleanup_expired_cache():
    now = time.time()
    expired_keys = []

    for key, value in translation_cache.items():
        if now - value["created_at"] > CACHE_TTL_SECONDS:
            expired_keys.append(key)

    for key in expired_keys:
        translation_cache.pop(key, None)


def get_cached_translation(cache_key: str):
    if cache_key in translation_cache:
        translation_cache.move_to_end(cache_key)
        return translation_cache[cache_key]["result"]
    return None


def set_cached_translation(cache_key: str, translated_text: str):
    if cache_key in translation_cache:
        translation_cache.pop(key, None)

    translation_cache[cache_key] = {
        "result": translated_text,
        "created_at": time.time()
    }

    while len(translation_cache) > MAX_CACHE_ITEMS:
        translation_cache.popitem(last=False)


def transcribe_audio_file(audio_path: str) -> str:
    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
        language=None,
    )

    text_parts = []
    for segment in segments:
        if segment.text:
            text_parts.append(segment.text.strip())

    return " ".join(text_parts).strip()


@app.get("/")
async def root():
    return JSONResponse(
        {
            "ok": True,
            "service": "live-translate-backend",
            "routes": ["/signal", "/translate", "/health", "/cache/stats"],
            "cache_enabled": True,
            "stt_engine": "faster-whisper",
            "model_size": MODEL_SIZE,
        }
    )


@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}


@app.get("/cache/stats")
async def cache_stats():
    cleanup_expired_cache()
    return {
        "cache_items": len(translation_cache),
        "max_cache_items": MAX_CACHE_ITEMS,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }


@app.websocket("/signal")
async def signal_socket(websocket: WebSocket):
    await websocket.accept()
    joined_room = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            msg_type = data.get("type")
            room = data.get("room")

            if not room:
                await websocket.send_text(json.dumps({"error": "room gerekli"}))
                continue

            if msg_type == "join":
                signal_rooms[room].add(websocket)
                joined_room = room

                for client in list(signal_rooms[room]):
                    if client != websocket:
                        try:
                            await client.send_text(json.dumps({
                                "type": "join",
                                "room": room,
                            }))
                        except Exception:
                            pass
                continue

            for client in list(signal_rooms[room]):
                if client != websocket:
                    try:
                        await client.send_text(raw)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    finally:
        if joined_room and websocket in signal_rooms[joined_room]:
            signal_rooms[joined_room].remove(websocket)
            if not signal_rooms[joined_room]:
                del signal_rooms[joined_room]


@app.websocket("/translate")
async def translate_socket(websocket: WebSocket):
    await websocket.accept()

    if translator is None:
        await websocket.send_text(json.dumps({
            "error": "DEEPL_AUTH_KEY ayarlı değil"
        }))

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            audio_b64 = data.get("audio")
            source_lang = normalize_source_lang(data.get("sourceLang", "TR"))
            target_lang = normalize_target_lang(data.get("targetLang", "RU"))

            if not audio_b64:
                await websocket.send_text(json.dumps({
                    "error": "audio alanı boş"
                }))
                continue

            if translator is None:
                await websocket.send_text(json.dumps({
                    "error": "DeepL bağlantısı hazır değil"
                }))
                continue

            temp_path = None

            try:
                audio_bytes = base64.b64decode(audio_b64)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    temp_path = f.name

                text = transcribe_audio_file(temp_path)

                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    temp_path = None

                if not text:
                    await websocket.send_text(json.dumps({
                        "error": "ses algılanamadı"
                    }))
                    continue

                cleanup_expired_cache()
                cache_key = make_cache_key(text, source_lang, target_lang)
                cached_translation = get_cached_translation(cache_key)

                if cached_translation is not None:
                    await websocket.send_text(json.dumps({
                        "original": text,
                        "translated": cached_translation,
                        "sourceLang": source_lang,
                        "targetLang": target_lang,
                        "cached": True
                    }))
                    continue

                translated = translator.translate_text(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                ).text

                set_cached_translation(cache_key, translated)

                await websocket.send_text(json.dumps({
                    "original": text,
                    "translated": translated,
                    "sourceLang": source_lang,
                    "targetLang": target_lang,
                    "cached": False
                }))

            except Exception as e:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                await websocket.send_text(json.dumps({
                    "error": str(e)
                }))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)