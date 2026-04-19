import os
import json
import base64
import tempfile
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import deepl
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

app = FastAPI(title="BridgeCall Backend")

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY", "")
PORT = int(os.getenv("PORT", "8000"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))

translator = deepl.Translator(DEEPL_AUTH_KEY) if DEEPL_AUTH_KEY else None
fallback_translator = GoogleTranslator
model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

client_info = {}
signal_rooms = {}


def normalize_source_lang(lang: str):
    if not lang:
        return "TR"
    lang = lang.upper()
    if lang == "EN-US":
        return "EN"
    return lang


def normalize_target_lang(lang: str):
    if not lang:
        return "RU"
    lang = lang.upper()
    if lang == "EN":
        return "EN-US"
    return lang


def whisper_lang(lang: str) -> str:
    lang = normalize_source_lang(lang)
    mapping = {
        "TR": "tr",
        "RU": "ru",
        "UK": "uk",
        "EN": "en",
        "EN-US": "en",
    }
    return mapping.get(lang, "en")


def clean_transcript(text: str) -> str:
    return " ".join((text or "").split()).strip()


def room_payload(room_id: str):
    room = signal_rooms[room_id]
    return {
        "room": room_id,
        "ownerId": room["ownerId"],
        "capacity": room["capacity"],
        "memberCount": len(room["members"]),
        "privateCode": room["privateCode"],
    }


def google_lang(lang: str) -> str:
    lang = normalize_target_lang(lang)
    mapping = {
        "TR": "tr",
        "RU": "ru",
        "UK": "uk",
        "EN-US": "en",
        "EN": "en",
    }
    return mapping.get(lang, "en")


def translate_text_value(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return ""

    source_lang = normalize_source_lang(source_lang)
    target_lang = normalize_target_lang(target_lang)

    if translator is not None:
        return translator.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
        ).text

    return fallback_translator(
        source=google_lang(source_lang),
        target=google_lang(target_lang),
    ).translate(text)


async def safe_send(websocket: WebSocket, payload: dict):
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass


def remove_socket_from_pending(websocket: WebSocket):
    for room in signal_rooms.values():
        pending_to_remove = []
        for requester_id, pending_ws in room["pending"].items():
            if pending_ws == websocket:
                pending_to_remove.append(requester_id)
        for requester_id in pending_to_remove:
            room["pending"].pop(requester_id, None)


@app.get("/")
async def root():
    return JSONResponse(
        {
            "ok": True,
            "service": "bridgecall-backend",
            "routes": ["/signal", "/translate", "/health"],
        }
    )


@app.get("/health")
async def health():
    return {"ok": True, "rooms": len(signal_rooms), "clients": len(client_info), "whisper_model": WHISPER_MODEL_SIZE, "whisper_beam_size": WHISPER_BEAM_SIZE}


async def broadcast_room_state(room_id: str):
    if room_id not in signal_rooms:
        return
    room = signal_rooms[room_id]
    payload = {"type": "room_state", **room_payload(room_id)}
    for member in list(room["members"]):
        await safe_send(member, payload)


@app.websocket("/signal")
async def signal_socket(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid4())
    client_info[websocket] = {"clientId": client_id, "room": None}

    try:
        await safe_send(websocket, {"type": "welcome", "clientId": client_id})

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "create_room":
                room_id = data.get("room")
                capacity = int(data.get("capacity", 2))
                private_code = data.get("privateCode")

                if not room_id or not private_code:
                    await safe_send(websocket, {"type": "error", "message": "room ve privateCode gerekli"})
                    continue

                if room_id in signal_rooms:
                    await safe_send(websocket, {"type": "error", "message": "Bu oda zaten var"})
                    continue

                signal_rooms[room_id] = {
                    "owner": websocket,
                    "ownerId": client_id,
                    "members": {websocket},
                    "pending": {},
                    "capacity": capacity,
                    "privateCode": private_code,
                }
                client_info[websocket]["room"] = room_id

                await safe_send(websocket, {"type": "room_created", **room_payload(room_id)})
                await broadcast_room_state(room_id)
                continue

            if msg_type == "request_join":
                room_id = data.get("room")
                private_code = data.get("privateCode")

                if room_id not in signal_rooms:
                    await safe_send(websocket, {"type": "error", "message": "Oda bulunamadı"})
                    continue

                room = signal_rooms[room_id]

                if room["privateCode"] != private_code:
                    await safe_send(websocket, {"type": "error", "message": "Oda kodu yanlış"})
                    continue

                if len(room["members"]) >= room["capacity"]:
                    await safe_send(websocket, {"type": "error", "message": "Oda dolu"})
                    continue

                room["pending"][client_id] = websocket
                await safe_send(room["owner"], {"type": "join_request", "room": room_id, "requesterId": client_id})
                continue

            if msg_type == "join_decision":
                room_id = data.get("room")
                requester_id = data.get("requesterId")
                accept = data.get("accept", False)

                if room_id not in signal_rooms:
                    continue

                room = signal_rooms[room_id]
                if websocket != room["owner"]:
                    continue

                requester_ws = room["pending"].pop(requester_id, None)
                if requester_ws is None:
                    continue

                if accept:
                    if len(room["members"]) >= room["capacity"]:
                        await safe_send(requester_ws, {"type": "error", "message": "Oda doldu"})
                        continue

                    room["members"].add(requester_ws)
                    client_info[requester_ws]["room"] = room_id

                    await safe_send(requester_ws, {"type": "join_accepted", **room_payload(room_id)})
                    for member in list(room["members"]):
                        if member != requester_ws:
                            await safe_send(member, {"type": "member_joined", "room": room_id, "clientId": requester_id})
                    await broadcast_room_state(room_id)
                else:
                    await safe_send(requester_ws, {"type": "join_rejected", "room": room_id})
                continue

            if msg_type in ["leave_room", "leave"]:
                room_id = client_info.get(websocket, {}).get("room")
                cid = client_info.get(websocket, {}).get("clientId")

                if room_id and room_id in signal_rooms:
                    room = signal_rooms[room_id]
                    room["members"].discard(websocket)
                    remove_socket_from_pending(websocket)

                    if websocket == room["owner"]:
                        for member in list(room["members"]):
                            await safe_send(member, {"type": "room_closed", "room": room_id})
                            client_info[member]["room"] = None
                        del signal_rooms[room_id]
                    else:
                        for member in list(room["members"]):
                            await safe_send(member, {"type": "member_left", "room": room_id, "clientId": cid})
                        await broadcast_room_state(room_id)

                    client_info[websocket]["room"] = None

                await safe_send(websocket, {"type": "left_room"})
                continue

            if msg_type == "chat_message":
                room_id = data.get("room")
                if room_id not in signal_rooms:
                    continue

                room = signal_rooms[room_id]
                text = (data.get("text") or "").strip()
                source_lang = data.get("sourceLang", "TR")
                target_lang = data.get("targetLang", "RU")
                translated_text = ""

                if text:
                    try:
                        translated_text = translate_text_value(text, source_lang, target_lang)
                    except Exception as exc:
                        translated_text = f"Çeviri hatası: {exc}"

                payload = {
                    "type": "chat_message",
                    "room": room_id,
                    "text": text,
                    "translatedText": translated_text,
                    "senderId": client_id,
                }

                for member in list(room["members"]):
                    await safe_send(member, payload)
                continue

            if msg_type == "reaction":
                room_id = data.get("room")
                if room_id not in signal_rooms:
                    continue
                room = signal_rooms[room_id]
                payload = {
                    "type": "reaction",
                    "room": room_id,
                    "emoji": data.get("emoji"),
                    "senderId": client_id,
                }
                for member in list(room["members"]):
                    if member != websocket:
                        await safe_send(member, payload)
                continue

            if msg_type == "media_state":
                room_id = data.get("room")
                if room_id not in signal_rooms:
                    continue
                room = signal_rooms[room_id]
                payload = {
                    "type": "media_state",
                    "room": room_id,
                    "senderId": client_id,
                    "micOn": bool(data.get("micOn", True)),
                    "camOn": bool(data.get("camOn", True)),
                    "subtitlesOn": bool(data.get("subtitlesOn", True)),
                }
                for member in list(room["members"]):
                    if member != websocket:
                        await safe_send(member, payload)
                continue

            if msg_type in ["offer", "answer", "candidate"]:
                room_id = data.get("room")
                if room_id not in signal_rooms:
                    continue

                room = signal_rooms[room_id]
                for member in list(room["members"]):
                    if member != websocket:
                        await safe_send(member, data)
                continue

    except WebSocketDisconnect:
        room_id = client_info.get(websocket, {}).get("room")
        cid = client_info.get(websocket, {}).get("clientId")
        remove_socket_from_pending(websocket)

        if room_id and room_id in signal_rooms:
            room = signal_rooms[room_id]
            room["members"].discard(websocket)

            if websocket == room["owner"]:
                for member in list(room["members"]):
                    await safe_send(member, {"type": "room_closed", "room": room_id})
                    client_info[member]["room"] = None
                del signal_rooms[room_id]
            else:
                for member in list(room["members"]):
                    await safe_send(member, {"type": "member_left", "room": room_id, "clientId": cid})
                await broadcast_room_state(room_id)

        client_info.pop(websocket, None)


@app.websocket("/translate")
async def translate_socket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            audio_b64 = data.get("audio")
            source_lang = normalize_source_lang(data.get("sourceLang", "TR"))
            target_lang = normalize_target_lang(data.get("targetLang", "RU"))

            if not audio_b64:
                await safe_send(websocket, {"error": "audio alanı boş"})
                continue

            try:
                audio_bytes = base64.b64decode(audio_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    temp_path = f.name

                segments, _ = model.transcribe(
                    temp_path,
                    beam_size=WHISPER_BEAM_SIZE,
                    language=whisper_lang(source_lang),
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = clean_transcript(" ".join(segment.text for segment in segments))

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                if not text:
                    await safe_send(websocket, {"error": "ses algılanamadı"})
                    continue

                translated = translate_text_value(text, source_lang, target_lang)

                await safe_send(
                    websocket,
                    {
                        "original": text,
                        "translated": translated,
                        "stage": "partial",
                        "sourceLang": source_lang,
                        "targetLang": target_lang,
                    },
                )
            except Exception as e:
                await safe_send(websocket, {"error": str(e)})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
