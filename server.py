import os
import json
import base64
import tempfile
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import whisper
import deepl

app = FastAPI(title="Live Translate Backend")

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY", "")
PORT = int(os.getenv("PORT", "8000"))

translator = deepl.Translator(DEEPL_AUTH_KEY) if DEEPL_AUTH_KEY else None
model = whisper.load_model("small")

# room -> set[WebSocket]
signal_rooms: dict[str, set[WebSocket]] = defaultdict(set)


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


@app.get("/")
async def root():
    return JSONResponse(
        {
            "ok": True,
            "service": "live-translate-backend",
            "routes": ["/signal", "/translate"],
        }
    )


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

                # odaya bilgi ver
                for client in list(signal_rooms[room]):
                    if client != websocket:
                        await client.send_text(json.dumps({
                            "type": "join",
                            "room": room,
                        }))
                continue

            # diğer tüm signaling mesajlarını odadaki diğer kullanıcılara ilet
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

            try:
                audio_bytes = base64.b64decode(audio_b64)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    temp_path = f.name

                result = model.transcribe(temp_path, fp16=False)
                text = result["text"].strip()

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                if not text:
                    await websocket.send_text(json.dumps({
                        "error": "ses algılanamadı"
                    }))
                    continue

                translated = translator.translate_text(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                ).text

                await websocket.send_text(json.dumps({
                    "original": text,
                    "translated": translated,
                    "sourceLang": source_lang,
                    "targetLang": target_lang,
                }))

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": str(e)
                }))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)