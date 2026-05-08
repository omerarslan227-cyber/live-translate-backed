import asyncio
import base64
import json
import logging
import math
import os
import struct
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import deepl
import uvicorn
from deep_translator import GoogleTranslator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bridgecall.voice")

app = FastAPI(title="BridgeCall Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = int(os.getenv("PORT", "8000"))
DEEPL_AUTH_KEY = os.getenv("DEEPL_API_KEY") or os.getenv("DEEPL_AUTH_KEY", "")
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE") or os.getenv("WHISPER_MODEL", "small")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.68"))
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-0.65"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0"))
PARTIAL_TRANSCRIBE_INTERVAL_SECONDS = float(os.getenv("PARTIAL_TRANSCRIBE_INTERVAL_SECONDS", "0.9"))
ROLLING_TRANSCRIBE_WINDOW_SECONDS = float(os.getenv("ROLLING_TRANSCRIBE_WINDOW_SECONDS", "1.8"))
FINAL_SILENCE_SECONDS = float(os.getenv("FINAL_SILENCE_SECONDS", "0.75"))
MIN_TRANSCRIBE_SECONDS = float(os.getenv("MIN_TRANSCRIBE_SECONDS", "0.7"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "420"))
MAX_SESSION_AUDIO_SECONDS = float(os.getenv("MAX_SESSION_AUDIO_SECONDS", "6.0"))
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

translator = deepl.Translator(DEEPL_AUTH_KEY, server_url=DEEPL_API_URL or None) if DEEPL_AUTH_KEY else None
model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
translation_cache: dict[tuple[str, str, str], str] = {}

legacy_client_info: dict[WebSocket, dict[str, Any]] = {}
legacy_signal_rooms: dict[str, dict[str, Any]] = {}

LOW_VALUE_TRANSCRIPTS = {
    "",
    ".",
    "...",
    "thanks for watching",
    "thank you for watching",
    "altyazi m.k.",
    "abone olmayi unutmayin",
}

TRANSCRIPTION_PROMPTS = {
    "tr": "Live video call subtitle. Transcribe natural Turkish speech accurately and briefly.",
    "ru": "Live video call subtitle. Transcribe natural Russian speech accurately and briefly.",
    "uk": "Live video call subtitle. Transcribe natural Ukrainian speech accurately and briefly.",
    "en": "Live video call subtitle. Transcribe natural English speech accurately and briefly.",
}


@dataclass
class TranslationConfig:
    source_language: str = "TR"
    target_language: str = "RU"
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * SAMPLE_WIDTH_BYTES


class RoomManager:
    def __init__(self) -> None:
        self.signaling: dict[str, dict[str, WebSocket]] = {}
        self.translation: dict[str, dict[str, WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, kind: str, room: str, peer_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self._bucket(kind).setdefault(room, {})[peer_id] = websocket

    async def disconnect(self, kind: str, room: str, peer_id: str) -> None:
        async with self.lock:
            bucket = self._bucket(kind).get(room, {})
            bucket.pop(peer_id, None)
            if not bucket:
                self._bucket(kind).pop(room, None)

    async def broadcast(self, kind: str, room: str, sender: str, payload: dict[str, Any]) -> None:
        await self.send(kind, room, sender, payload, include_sender=False)

    async def send(
        self,
        kind: str,
        room: str,
        sender: str,
        payload: dict[str, Any],
        include_sender: bool = True,
    ) -> None:
        async with self.lock:
            peers = list(self._bucket(kind).get(room, {}).items())
        stale: list[str] = []
        for peer_id, websocket in peers:
            if not include_sender and peer_id == sender:
                continue
            try:
                await websocket.send_json(payload | {"peer_id": sender})
            except RuntimeError:
                stale.append(peer_id)
        for peer_id in stale:
            await self.disconnect(kind, room, peer_id)

    def _bucket(self, kind: str) -> dict[str, dict[str, WebSocket]]:
        return self.signaling if kind == "signaling" else self.translation


manager = RoomManager()


class AudioTranslationSession:
    def __init__(self, room: str, peer_id: str, config: TranslationConfig) -> None:
        self.room = room
        self.peer_id = peer_id
        self.config = config
        self.audio_buffer = bytearray()
        self.last_voice_at = 0.0
        self.last_partial_at = 0.0
        self.last_partial_text = ""
        self.in_speech = False
        self.generation = 0
        self.partial_task: asyncio.Task[None] | None = None
        self.final_task: asyncio.Task[None] | None = None

    def update_config(self, config: TranslationConfig) -> None:
        self.config = config
        self.audio_buffer.clear()
        self.last_voice_at = 0.0
        self.last_partial_at = 0.0
        self.last_partial_text = ""
        self.in_speech = False
        self.generation += 1

    async def add_audio(self, chunk: bytes) -> None:
        chunk = clean_pcm(chunk)
        if not chunk:
            return

        now = time.monotonic()
        rms = pcm_rms(chunk)
        if rms >= VAD_RMS_THRESHOLD:
            self.audio_buffer.extend(chunk)
            self._trim_buffer()
            self.in_speech = True
            self.last_voice_at = now
            await self._maybe_start_partial(now)
            return

        if self.in_speech and now - self.last_voice_at >= FINAL_SILENCE_SECONDS:
            await self._start_final()

    async def close(self) -> None:
        await self._start_final()

    async def _maybe_start_partial(self, now: float) -> None:
        min_bytes = int(self.config.bytes_per_second * MIN_TRANSCRIBE_SECONDS)
        if len(self.audio_buffer) < min_bytes:
            return
        if now - self.last_partial_at < PARTIAL_TRANSCRIBE_INTERVAL_SECONDS:
            return
        if self.partial_task is not None and not self.partial_task.done():
            return

        rolling_bytes = int(self.config.bytes_per_second * ROLLING_TRANSCRIBE_WINDOW_SECONDS)
        pcm = bytes(self.audio_buffer[-rolling_bytes:])
        self.last_partial_at = now
        self.partial_task = asyncio.create_task(
            self._process(pcm, is_final=False, generation=self.generation)
        )

    async def _start_final(self) -> None:
        if not self.audio_buffer:
            self.in_speech = False
            return
        if self.final_task is not None and not self.final_task.done():
            return

        pcm = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        self.in_speech = False
        self.last_partial_text = ""
        self.generation += 1
        self.final_task = asyncio.create_task(
            self._process(pcm, is_final=True, generation=self.generation)
        )

    async def _process(self, pcm: bytes, is_final: bool, generation: int) -> None:
        try:
            started = time.perf_counter()
            text = await transcribe_pcm(pcm, self.config)
            stt_ms = int((time.perf_counter() - started) * 1000)
            if not text:
                logger.info(
                    "modern_stt_no_speech room=%s peer=%s bytes=%s stt_ms=%s",
                    self.room,
                    self.peer_id,
                    len(pcm),
                    stt_ms,
                )
                return
            if not is_final and generation != self.generation:
                return
            if not is_final and text == self.last_partial_text:
                return
            if not is_final:
                self.last_partial_text = text
            translate_started = time.perf_counter()
            translated = translate_text_value(text, self.config.source_language, self.config.target_language)
            translate_ms = int((time.perf_counter() - translate_started) * 1000)
            total_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "modern_caption room=%s peer=%s final=%s bytes=%s stt_ms=%s translate_ms=%s total_ms=%s text_len=%s translated_len=%s",
                self.room,
                self.peer_id,
                is_final,
                len(pcm),
                stt_ms,
                translate_ms,
                total_ms,
                len(text),
                len(translated),
            )
            payload = {
                "type": "caption",
                "text": text,
                "translation": translated,
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "is_final": is_final,
                "timing_ms": {
                    "stt": stt_ms,
                    "translation": translate_ms,
                    "total": total_ms,
                },
            }
            await manager.send("translation", self.room, self.peer_id, payload)
        except Exception as exc:
            await manager.send("translation", self.room, self.peer_id, {"type": "error", "message": str(exc)})

    def _trim_buffer(self) -> None:
        max_bytes = int(self.config.bytes_per_second * MAX_SESSION_AUDIO_SECONDS)
        if len(self.audio_buffer) > max_bytes:
            del self.audio_buffer[: len(self.audio_buffer) - max_bytes]


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


def whisper_lang(lang: str) -> str:
    mapping = {
        "TR": "tr",
        "RU": "ru",
        "UK": "uk",
        "EN": "en",
        "EN-US": "en",
        "DE": "de",
        "FR": "fr",
        "ES": "es",
    }
    return mapping.get(normalize_source_lang(lang), "en")


def google_lang(lang: str) -> str:
    mapping = {
        "TR": "tr",
        "RU": "ru",
        "UK": "uk",
        "EN-US": "en",
        "EN": "en",
        "DE": "de",
        "FR": "fr",
        "ES": "es",
    }
    return mapping.get(normalize_target_lang(lang), "en")


def clean_transcript(text: str) -> str:
    cleaned = " ".join((text or "").replace("\n", " ").split()).strip()
    return cleaned.strip(" -–—")


def is_low_value_transcript(text: str) -> bool:
    normalized = clean_transcript(text).lower().strip(" .!?")
    return normalized in LOW_VALUE_TRANSCRIPTS or len(normalized) < 2


def transcription_prompt(source_lang: str, previous_text: str = "") -> str:
    lang = whisper_lang(source_lang)
    prompt = TRANSCRIPTION_PROMPTS.get(lang, TRANSCRIPTION_PROMPTS["en"])
    previous_text = clean_transcript(previous_text)
    if previous_text:
        return f"{prompt} Previous context: {previous_text[-220:]}"
    return prompt


def stable_segment_text(segments) -> tuple[str, float]:
    accepted: list[str] = []
    confidences: list[float] = []

    for segment in segments:
        text = clean_transcript(getattr(segment, "text", ""))
        if not text:
            continue

        no_speech_prob = float(getattr(segment, "no_speech_prob", 0) or 0)
        avg_logprob = float(getattr(segment, "avg_logprob", 0) or 0)
        if no_speech_prob > WHISPER_NO_SPEECH_THRESHOLD:
            continue
        if avg_logprob < WHISPER_LOG_PROB_THRESHOLD:
            continue

        accepted.append(text)
        confidences.append(max(0.0, min(1.0, 1.0 + avg_logprob)))

    text = clean_transcript(" ".join(accepted))
    confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    return text, confidence


def translate_text_value(text: str, source_lang: str, target_lang: str) -> str:
    text = clean_transcript(text)
    if not text:
        return ""

    source_lang = normalize_source_lang(source_lang)
    target_lang = normalize_target_lang(target_lang)
    cache_key = (text, source_lang, target_lang)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    if translator is not None:
        translated = translator.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
        ).text
    else:
        translated = GoogleTranslator(
            source=google_lang(source_lang),
            target=google_lang(target_lang),
        ).translate(text)

    if len(translation_cache) > 512:
        translation_cache.clear()
    translation_cache[cache_key] = translated
    return translated


async def safe_send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass


def room_payload(room_id: str) -> dict[str, Any]:
    room = legacy_signal_rooms[room_id]
    return {
        "room": room_id,
        "ownerId": room["ownerId"],
        "capacity": room["capacity"],
        "memberCount": len(room["members"]),
        "privateCode": room["privateCode"],
    }


async def broadcast_room_state(room_id: str) -> None:
    if room_id not in legacy_signal_rooms:
        return
    room = legacy_signal_rooms[room_id]
    payload = {"type": "room_state", **room_payload(room_id)}
    for member in list(room["members"]):
        await safe_send(member, payload)


def remove_socket_from_pending(websocket: WebSocket) -> None:
    for room in legacy_signal_rooms.values():
        pending_to_remove = [
            requester_id
            for requester_id, pending_ws in room["pending"].items()
            if pending_ws == websocket
        ]
        for requester_id in pending_to_remove:
            room["pending"].pop(requester_id, None)


def clean_pcm(pcm: bytes) -> bytes:
    if len(pcm) < SAMPLE_WIDTH_BYTES:
        return b""
    if len(pcm) % SAMPLE_WIDTH_BYTES:
        pcm = pcm[: -(len(pcm) % SAMPLE_WIDTH_BYTES)]
    samples = list(struct.unpack(f"<{len(pcm) // 2}h", pcm))
    if not samples:
        return b""
    avg = int(sum(samples) / len(samples))
    cleaned = [max(-32768, min(32767, sample - avg)) for sample in samples]
    return struct.pack(f"<{len(cleaned)}h", *cleaned)


def pcm_rms(pcm: bytes) -> int:
    if len(pcm) < SAMPLE_WIDTH_BYTES:
        return 0
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    if not samples:
        return 0
    return int(math.sqrt(sum(sample * sample for sample in samples) / len(samples)))


def write_wav(pcm: bytes, sample_rate: int, channels: int) -> str:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    file.close()
    with wave.open(file.name, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return file.name


async def transcribe_pcm(pcm: bytes, config: TranslationConfig) -> str:
    wav_path = write_wav(pcm, config.sample_rate, config.channels)
    try:
        return await transcribe_wav(wav_path, config.source_language)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


async def transcribe_wav(path: str, source_lang: str, previous_text: str = "") -> str:
    loop = asyncio.get_running_loop()

    def run_transcribe() -> str:
        segments, _ = model.transcribe(
            path,
            beam_size=WHISPER_BEAM_SIZE,
            language=whisper_lang(source_lang),
            vad_filter=WHISPER_VAD_FILTER,
            vad_parameters={
                "min_silence_duration_ms": 450,
                "speech_pad_ms": 280,
            },
            initial_prompt=transcription_prompt(source_lang, previous_text),
            temperature=WHISPER_TEMPERATURE,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            condition_on_previous_text=False,
        )
        text, _confidence = stable_segment_text(segments)
        if is_low_value_transcript(text):
            return ""
        return text

    return await loop.run_in_executor(None, run_transcribe)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "bridgecall-backend",
            "routes": [
                "/signal",
                "/translate",
                "/ws/signaling/{room}/{peer_id}",
                "/ws/translate/{room}/{peer_id}",
                "/health",
            ],
        }
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ok",
        "rooms": len(legacy_signal_rooms) + len(manager.signaling),
        "clients": len(legacy_client_info),
        "whisper_model": WHISPER_MODEL_SIZE,
        "whisper_beam_size": WHISPER_BEAM_SIZE,
        "whisper_vad_filter": WHISPER_VAD_FILTER,
        "whisper_no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        "whisper_log_prob_threshold": WHISPER_LOG_PROB_THRESHOLD,
        "whisper_compression_ratio_threshold": WHISPER_COMPRESSION_RATIO_THRESHOLD,
        "deepl_configured": bool(DEEPL_AUTH_KEY),
    }


@app.websocket("/ws/signaling/{room}/{peer_id}")
async def modern_signaling_socket(websocket: WebSocket, room: str, peer_id: str) -> None:
    await manager.connect("signaling", room, peer_id, websocket)
    await manager.broadcast("signaling", room, peer_id, {"type": "peer-ready"})
    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            await manager.broadcast("signaling", room, peer_id, payload)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect("signaling", room, peer_id)
        await manager.broadcast("signaling", room, peer_id, {"type": "peer-left"})


@app.websocket("/ws/translate/{room}/{peer_id}")
async def modern_translation_socket(websocket: WebSocket, room: str, peer_id: str) -> None:
    await manager.connect("translation", room, peer_id, websocket)
    config = TranslationConfig()
    session = AudioTranslationSession(room, peer_id, config)
    await websocket.send_json({"type": "status", "message": "Ceviri kanali hazir"})

    try:
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"]:
                payload = json.loads(message["text"])
                if payload.get("type") == "config":
                    config = TranslationConfig(
                        source_language=payload.get("source_language", config.source_language),
                        target_language=payload.get("target_language", config.target_language),
                        sample_rate=int(payload.get("sample_rate", config.sample_rate)),
                        channels=int(payload.get("channels", config.channels)),
                    )
                    session.update_config(config)
                elif payload.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            chunk = message.get("bytes")
            if chunk:
                await session.add_audio(chunk)
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        await manager.disconnect("translation", room, peer_id)


@app.websocket("/signal")
async def legacy_signal_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    client_id = str(uuid4())
    legacy_client_info[websocket] = {"clientId": client_id, "room": None}

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
                if room_id in legacy_signal_rooms:
                    await safe_send(websocket, {"type": "error", "message": "Bu oda zaten var"})
                    continue

                legacy_signal_rooms[room_id] = {
                    "owner": websocket,
                    "ownerId": client_id,
                    "members": {websocket},
                    "pending": {},
                    "capacity": capacity,
                    "privateCode": private_code,
                }
                legacy_client_info[websocket]["room"] = room_id
                await safe_send(websocket, {"type": "room_created", **room_payload(room_id)})
                await broadcast_room_state(room_id)
                continue

            if msg_type == "request_join":
                room_id = data.get("room")
                private_code = data.get("privateCode")
                if room_id not in legacy_signal_rooms:
                    await safe_send(websocket, {"type": "error", "message": "Oda bulunamadi"})
                    continue
                room = legacy_signal_rooms[room_id]
                if room["privateCode"] != private_code:
                    await safe_send(websocket, {"type": "error", "message": "Oda kodu yanlis"})
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
                if room_id not in legacy_signal_rooms:
                    continue
                room = legacy_signal_rooms[room_id]
                if websocket != room["owner"]:
                    continue
                requester_ws = room["pending"].pop(requester_id, None)
                if requester_ws is None:
                    continue
                if accept:
                    room["members"].add(requester_ws)
                    legacy_client_info[requester_ws]["room"] = room_id
                    await safe_send(requester_ws, {"type": "join_accepted", **room_payload(room_id)})
                    for member in list(room["members"]):
                        if member != requester_ws:
                            await safe_send(member, {"type": "member_joined", "room": room_id, "clientId": requester_id})
                    await broadcast_room_state(room_id)
                else:
                    await safe_send(requester_ws, {"type": "join_rejected", "room": room_id})
                continue

            if msg_type in ["leave_room", "leave"]:
                await leave_legacy_room(websocket)
                await safe_send(websocket, {"type": "left_room"})
                continue

            if msg_type == "chat_message":
                room_id = data.get("room")
                if room_id not in legacy_signal_rooms:
                    continue
                text = (data.get("text") or "").strip()
                translated_text = ""
                if text:
                    try:
                        translated_text = translate_text_value(
                            text,
                            data.get("sourceLang", "TR"),
                            data.get("targetLang", "RU"),
                        )
                    except Exception as exc:
                        translated_text = f"Ceviri hatasi: {exc}"
                payload = {
                    "type": "chat_message",
                    "room": room_id,
                    "text": text,
                    "translatedText": translated_text,
                    "senderId": client_id,
                }
                for member in list(legacy_signal_rooms[room_id]["members"]):
                    await safe_send(member, payload)
                continue

            if msg_type in ["reaction", "media_state", "offer", "answer", "candidate"]:
                room_id = data.get("room")
                if room_id not in legacy_signal_rooms:
                    continue
                for member in list(legacy_signal_rooms[room_id]["members"]):
                    if member != websocket:
                        await safe_send(member, data)
                continue
    except WebSocketDisconnect:
        await leave_legacy_room(websocket)
        legacy_client_info.pop(websocket, None)


async def leave_legacy_room(websocket: WebSocket) -> None:
    room_id = legacy_client_info.get(websocket, {}).get("room")
    cid = legacy_client_info.get(websocket, {}).get("clientId")
    remove_socket_from_pending(websocket)
    if not room_id or room_id not in legacy_signal_rooms:
        return
    room = legacy_signal_rooms[room_id]
    room["members"].discard(websocket)
    if websocket == room["owner"]:
        for member in list(room["members"]):
            await safe_send(member, {"type": "room_closed", "room": room_id})
            legacy_client_info[member]["room"] = None
        del legacy_signal_rooms[room_id]
    else:
        for member in list(room["members"]):
            await safe_send(member, {"type": "member_left", "room": room_id, "clientId": cid})
        await broadcast_room_state(room_id)
    legacy_client_info[websocket]["room"] = None


@app.websocket("/translate")
async def legacy_translate_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    previous_text = ""
    try:
        while True:
            request_started = time.perf_counter()
            raw = await websocket.receive_text()
            data = json.loads(raw)
            audio_b64 = data.get("audio")
            source_lang = normalize_source_lang(data.get("sourceLang", "TR"))
            target_lang = normalize_target_lang(data.get("targetLang", "RU"))
            client_previous_text = clean_transcript(data.get("previousText", ""))

            if not audio_b64:
                await safe_send(websocket, {"error": "audio alani bos"})
                continue

            try:
                decode_started = time.perf_counter()
                audio_bytes = base64.b64decode(audio_b64)
                decode_ms = int((time.perf_counter() - decode_started) * 1000)
                if len(audio_bytes) < 12000:
                    logger.info(
                        "legacy_audio_too_small bytes=%s source=%s target=%s",
                        len(audio_bytes),
                        source_lang,
                        target_lang,
                    )
                    await safe_send(
                        websocket,
                        {
                            "noSpeech": True,
                            "audioBytes": len(audio_bytes),
                            "format": "pcm16wav",
                            "sampleRate": 16000,
                            "timingMs": {"decode": decode_ms, "total": int((time.perf_counter() - request_started) * 1000)},
                        },
                    )
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as file:
                    file.write(audio_bytes)
                    temp_path = file.name
                try:
                    stt_started = time.perf_counter()
                    text = await transcribe_wav(temp_path, source_lang, previous_text or client_previous_text)
                    stt_ms = int((time.perf_counter() - stt_started) * 1000)
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                if not text:
                    total_ms = int((time.perf_counter() - request_started) * 1000)
                    logger.info(
                        "legacy_stt_no_speech bytes=%s source=%s target=%s stt_ms=%s total_ms=%s",
                        len(audio_bytes),
                        source_lang,
                        target_lang,
                        stt_ms,
                        total_ms,
                    )
                    await safe_send(
                        websocket,
                        {
                            "noSpeech": True,
                            "audioBytes": len(audio_bytes),
                            "format": "pcm16wav",
                            "sampleRate": 16000,
                            "timingMs": {"decode": decode_ms, "stt": stt_ms, "total": total_ms},
                        },
                    )
                    continue
                translate_started = time.perf_counter()
                translated = translate_text_value(text, source_lang, target_lang)
                translate_ms = int((time.perf_counter() - translate_started) * 1000)
                total_ms = int((time.perf_counter() - request_started) * 1000)
                previous_text = clean_transcript(f"{previous_text} {text}")[-500:]
                logger.info(
                    "legacy_caption bytes=%s source=%s target=%s stt_ms=%s translate_ms=%s total_ms=%s text_len=%s translated_len=%s",
                    len(audio_bytes),
                    source_lang,
                    target_lang,
                    stt_ms,
                    translate_ms,
                    total_ms,
                    len(text),
                    len(translated),
                )
                await safe_send(
                    websocket,
                    {
                        "stage": "final",
                        "original": text,
                        "translated": translated,
                        "sourceLang": source_lang,
                        "targetLang": target_lang,
                        "audioBytes": len(audio_bytes),
                        "format": "pcm16wav",
                        "sampleRate": 16000,
                        "timingMs": {
                            "decode": decode_ms,
                            "stt": stt_ms,
                            "translation": translate_ms,
                            "total": total_ms,
                        },
                    },
                )
            except Exception as exc:
                logger.exception("legacy_translate_error")
                await safe_send(websocket, {"error": str(exc)})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
