import os
import json
import base64
import tempfile
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import deepl
from faster_whisper import WhisperModel

app = FastAPI(title="Live Translate Backend")

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY", "")
PORT = int(os.getenv("PORT", "8000"))

translator = deepl.Translator(DEEPL_AUTH_KEY) if DEEPL_AUTH_KEY else None
model = WhisperModel("small", device="cpu", compute_type="int8")

client_info = {}
signal_rooms = {}


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


def room_payload(room_id: str):
    room = signal_rooms[room_id]
    return {
        "room": room_id,
        "ownerId": room["ownerId"],
        "capacity": room["capacity"],
        "memberCount": len(room["members"]),
        "privateCode": room["privateCode"],
    }


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
            "service": "live-translate-backend",
            "routes": ["/signal", "/translate"],
        }
    )


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

                    client_info[websocket]["room"] = None

                await safe_send(websocket, {"type": "left_room"})
                continue

            if msg_type in ["offer", "answer", "candidate", "chat_message", "reaction"]:
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

        client_info.pop(websocket, None)


@app.websocket("/translate")
async def translate_socket(websocket: WebSocket):
    await websocket.accept()

    if translator is None:
        await safe_send(websocket, {"error": "DEEPL_AUTH_KEY ayarlı değil"})

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

            if translator is None:
                await safe_send(websocket, {"error": "DeepL bağlantısı hazır değil"})
                continue

            try:
                audio_bytes = base64.b64decode(audio_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    temp_path = f.name

                segments, _ = model.transcribe(temp_path, beam_size=1)
                text = " ".join(segment.text for segment in segments).strip()

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                if not text:
                    await safe_send(websocket, {"error": "ses algılanamadı"})
                    continue

                translated = translator.translate_text(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                ).text

                await safe_send(
                    websocket,
                    {
                        "original": text,
                        "translated": translated,
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
