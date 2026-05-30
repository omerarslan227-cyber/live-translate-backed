import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import struct
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import deepl
import numpy as np
import uvicorn
from deep_translator import GoogleTranslator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from groq import Groq

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - optional production dependency
    redis = None

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
WHISPER_VAD_FILTER = False
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.68"))
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-0.65"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0"))
PARTIAL_TRANSCRIBE_INTERVAL_SECONDS = float(os.getenv("PARTIAL_TRANSCRIBE_INTERVAL_SECONDS", "0.65"))
ROLLING_TRANSCRIBE_WINDOW_SECONDS = float(os.getenv("ROLLING_TRANSCRIBE_WINDOW_SECONDS", "1.25"))
FINAL_SILENCE_SECONDS = float(os.getenv("FINAL_SILENCE_SECONDS", "0.45"))
MIN_TRANSCRIBE_SECONDS = float(os.getenv("MIN_TRANSCRIBE_SECONDS", "0.5"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "420"))
STT_HARD_SILENCE_RMS = int(os.getenv("STT_HARD_SILENCE_RMS", "120"))
MAX_SESSION_AUDIO_SECONDS = float(os.getenv("MAX_SESSION_AUDIO_SECONDS", "6.0"))
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
STT_TARGET_RMS = int(os.getenv("STT_TARGET_RMS", "1600"))
STT_MAX_GAIN = float(os.getenv("STT_MAX_GAIN", "6.0"))
STT_FAST_GAIN_THRESHOLD = float(os.getenv("STT_FAST_GAIN_THRESHOLD", "1.35"))
STT_TRIM_FRAME_MS = int(os.getenv("STT_TRIM_FRAME_MS", "20"))
STT_TRIM_PAD_MS = int(os.getenv("STT_TRIM_PAD_MS", "120"))
STT_VAD_MIN_SILENCE_MS = int(os.getenv("STT_VAD_MIN_SILENCE_MS", "220"))
STT_VAD_SPEECH_PAD_MS = int(os.getenv("STT_VAD_SPEECH_PAD_MS", "120"))
STT_MIN_AUDIO_SECONDS = float(os.getenv("STT_MIN_AUDIO_SECONDS", "0.2"))
STT_MIN_RMS = int(os.getenv("STT_MIN_RMS", "90"))
STT_MIN_ACTIVE_RATIO = float(os.getenv("STT_MIN_ACTIVE_RATIO", "0.08"))
STT_REPEAT_WINDOW_SECONDS = float(os.getenv("STT_REPEAT_WINDOW_SECONDS", "4.0"))
REALTIME_CHUNK_MIN_SECONDS = float(os.getenv("REALTIME_CHUNK_MIN_SECONDS", "0.2"))
REALTIME_CHUNK_TARGET_SECONDS = float(os.getenv("REALTIME_CHUNK_TARGET_SECONDS", "0.32"))
REALTIME_CHUNK_MAX_SECONDS = float(os.getenv("REALTIME_CHUNK_MAX_SECONDS", "0.4"))
REALTIME_MAX_BUFFER_SECONDS = float(os.getenv("REALTIME_MAX_BUFFER_SECONDS", "0.8"))
REALTIME_MAX_QUEUE_SIZE = int(os.getenv("REALTIME_MAX_QUEUE_SIZE", "2"))
REALTIME_STT_TIMEOUT_SECONDS = float(os.getenv("REALTIME_STT_TIMEOUT_SECONDS", "1.1"))
REALTIME_TRANSLATE_TIMEOUT_SECONDS = float(os.getenv("REALTIME_TRANSLATE_TIMEOUT_SECONDS", "2.0"))
REALTIME_RESULT_MAX_AGE_SECONDS = float(os.getenv("REALTIME_RESULT_MAX_AGE_SECONDS", "0.85"))
REDIS_URL = os.getenv("REDIS_URL", "")
ROOM_CODE_LENGTH = int(os.getenv("ROOM_CODE_LENGTH", "8"))
ROOM_TTL_SECONDS = int(os.getenv("ROOM_TTL_SECONDS", "7200"))
JOIN_ATTEMPT_LIMIT = int(os.getenv("JOIN_ATTEMPT_LIMIT", "5"))
JOIN_ATTEMPT_WINDOW_SECONDS = int(os.getenv("JOIN_ATTEMPT_WINDOW_SECONDS", "300"))
JOIN_COOLDOWN_SECONDS = int(os.getenv("JOIN_COOLDOWN_SECONDS", "60"))
ROOM_CODE_RE = re.compile(rf"^[A-Za-z0-9]{{{ROOM_CODE_LENGTH},}}$")

translator = deepl.Translator(DEEPL_AUTH_KEY, server_url=DEEPL_API_URL or None) if DEEPL_AUTH_KEY else None
WHISPER_MODEL = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@dataclass(frozen=True)
class WhisperRuntimeConfig:
    model_size: str
    beam_size: int
    vad_filter: bool
    compute_type: str
    partial_interval_seconds: float
    rolling_window_seconds: float
    final_silence_seconds: float
    min_transcribe_seconds: float
    vad_min_silence_ms: int
    vad_speech_pad_ms: int


WHISPER_RUNTIME = WhisperRuntimeConfig(
    model_size=WHISPER_MODEL_SIZE,
    beam_size=WHISPER_BEAM_SIZE,
    vad_filter=WHISPER_VAD_FILTER,
    compute_type=WHISPER_COMPUTE_TYPE,
    partial_interval_seconds=PARTIAL_TRANSCRIBE_INTERVAL_SECONDS,
    rolling_window_seconds=ROLLING_TRANSCRIBE_WINDOW_SECONDS,
    final_silence_seconds=FINAL_SILENCE_SECONDS,
    min_transcribe_seconds=MIN_TRANSCRIBE_SECONDS,
    vad_min_silence_ms=STT_VAD_MIN_SILENCE_MS,
    vad_speech_pad_ms=STT_VAD_SPEECH_PAD_MS,
)

translation_cache: dict[tuple[str, str, str], str] = {}

legacy_client_info: dict[WebSocket, dict[str, Any]] = {}
legacy_signal_rooms: dict[str, dict[str, Any]] = {}
legacy_join_attempts: dict[str, list[float]] = {}
legacy_join_cooldowns: dict[str, float] = {}
redis_client = (
    redis.from_url(REDIS_URL, decode_responses=True)
    if redis is not None and REDIS_URL
    else None
)

LOW_VALUE_TRANSCRIPTS = {
    "",
    ".",
    "...",
    "m.k",
    "m.k.",
    "mk",
    "agzi m.k.",
    "agzi m k",
    "thanks for watching",
    "thank you for watching",
    "altyazi m.k.",
    "altyazi m k",
    "altyazı m.k.",
    "ağzı m.k.",
    "abone olmayi unutmayin",
}

TRANSCRIPTION_PROMPTS = {
    "tr": "Transcribe this speech naturally and accurately in the selected source language.",
    "ru": "Transcribe this speech naturally and accurately in the selected source language.",
    "uk": "Transcribe this speech naturally and accurately in the selected source language.",
    "en": "Transcribe this speech naturally and accurately in the selected source language.",
    "de": "Transcribe this speech naturally and accurately in the selected source language.",
    "nl": "Transcribe this speech naturally and accurately in the selected source language.",
    "ar": "Transcribe this speech naturally and accurately in the selected source language.",
    "es": "Transcribe this speech naturally and accurately in the selected source language.",
    "zh": "Transcribe this speech naturally and accurately in the selected source language.",
    "ka": "Transcribe this speech naturally and accurately in the selected source language.",
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


@dataclass(frozen=True)
class AudioStats:
    duration_seconds: float = 0.0
    rms: int = 0
    silence_ratio: float = 1.0
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    created_at: float
    generation: int
    stats: AudioStats
    chunk_hash: str


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
        self.config = self._fixed_audio_config(config)
        self.audio_buffer = bytearray()
        self.chunk_queue: deque[AudioChunk] = deque()
        self.last_voice_at = 0.0
        self.last_caption_text = ""
        self.last_caption_at = 0.0
        self.last_chunk_hash = ""
        self.in_speech = False
        self.generation = 0
        self.processing_task: asyncio.Task[None] | None = None
        self.closed = False

    def update_config(self, config: TranslationConfig) -> None:
        self.config = self._fixed_audio_config(config)
        self.audio_buffer.clear()
        self.chunk_queue.clear()
        self.last_voice_at = 0.0
        self.last_caption_text = ""
        self.last_caption_at = 0.0
        self.last_chunk_hash = ""
        self.in_speech = False
        self.generation += 1

    def _fixed_audio_config(self, config: TranslationConfig) -> TranslationConfig:
        if config.sample_rate != SAMPLE_RATE or config.channels != CHANNELS:
            logger.info(
                "modern_audio_config_fixed room=%s peer=%s requested_sample_rate=%s requested_channels=%s fixed_sample_rate=%s fixed_channels=%s",
                self.room,
                self.peer_id,
                config.sample_rate,
                config.channels,
                SAMPLE_RATE,
                CHANNELS,
            )
        return TranslationConfig(
            source_language=config.source_language,
            target_language=config.target_language,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )

    async def add_audio(self, chunk: bytes) -> None:
        if self.closed:
            return
        chunk = clean_pcm(chunk)
        if not chunk:
            return

        now = time.monotonic()
        chunk_stats = pcm_audio_stats(chunk, self.config.sample_rate, self.config.channels)
        if chunk_stats.rms < STT_MIN_RMS or chunk_stats.silence_ratio >= 0.98:
            logger.info(
                "modern_audio_noise_skip room=%s peer=%s duration=%.3f rms=%s silence_ratio=%.3f queue_size=%s",
                self.room,
                self.peer_id,
                chunk_stats.duration_seconds,
                chunk_stats.rms,
                chunk_stats.silence_ratio,
                len(self.chunk_queue),
            )
            if self.in_speech and now - self.last_voice_at >= WHISPER_RUNTIME.final_silence_seconds:
                self._enqueue_buffer(now)
            return

        if chunk_stats.rms >= VAD_RMS_THRESHOLD and (1.0 - chunk_stats.silence_ratio) >= STT_MIN_ACTIVE_RATIO:
            self.audio_buffer.extend(chunk)
            self.in_speech = True
            self.last_voice_at = now
            self._trim_buffer()
            self._enqueue_ready_chunks(now)
            return

        if self.in_speech and now - self.last_voice_at >= WHISPER_RUNTIME.final_silence_seconds:
            self._enqueue_buffer(now)

    async def close(self) -> None:
        self.closed = True
        self.audio_buffer.clear()
        self.chunk_queue.clear()
        if self.processing_task is not None and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    def _enqueue_ready_chunks(self, now: float) -> None:
        target_bytes = int(self.config.bytes_per_second * REALTIME_CHUNK_TARGET_SECONDS)
        max_bytes = int(self.config.bytes_per_second * REALTIME_CHUNK_MAX_SECONDS)
        min_bytes = int(self.config.bytes_per_second * REALTIME_CHUNK_MIN_SECONDS)
        while len(self.audio_buffer) >= target_bytes:
            take = min(max_bytes, len(self.audio_buffer))
            if take < min_bytes:
                return
            pcm = bytes(self.audio_buffer[:take])
            del self.audio_buffer[:take]
            self._enqueue_chunk(pcm, now)

    def _enqueue_buffer(self, now: float) -> None:
        min_bytes = int(self.config.bytes_per_second * REALTIME_CHUNK_MIN_SECONDS)
        if len(self.audio_buffer) >= min_bytes:
            self._enqueue_chunk(bytes(self.audio_buffer), now)
        else:
            logger.info(
                "modern_audio_short_skip room=%s peer=%s duration=%.3f rms=%s silence_ratio=%.3f queue_size=%s",
                self.room,
                self.peer_id,
                len(self.audio_buffer) / self.config.bytes_per_second if self.config.bytes_per_second else 0,
                pcm_rms(bytes(self.audio_buffer)),
                pcm_silence_ratio(bytes(self.audio_buffer), self.config.sample_rate, self.config.channels),
                len(self.chunk_queue),
            )
        self.audio_buffer.clear()
        self.in_speech = False

    def _enqueue_chunk(self, pcm: bytes, now: float) -> None:
        stats = pcm_audio_stats(pcm, self.config.sample_rate, self.config.channels)
        if is_hard_silence_for_stt(stats):
            logger.info(
                "modern_hard_silence_skip room=%s peer=%s duration=%.3f rms=%s silence_ratio=%.3f queue_size=%s",
                self.room,
                self.peer_id,
                stats.duration_seconds,
                stats.rms,
                stats.silence_ratio,
                len(self.chunk_queue),
            )
            return

        chunk_hash = hashlib.sha1(pcm).hexdigest()
        if chunk_hash == self.last_chunk_hash or any(item.chunk_hash == chunk_hash for item in self.chunk_queue):
            logger.info(
                "modern_duplicate_chunk_skip room=%s peer=%s duration=%.3f rms=%s silence_ratio=%.3f queue_size=%s",
                self.room,
                self.peer_id,
                stats.duration_seconds,
                stats.rms,
                stats.silence_ratio,
                len(self.chunk_queue),
            )
            return

        while len(self.chunk_queue) >= REALTIME_MAX_QUEUE_SIZE:
            dropped = self.chunk_queue.popleft()
            logger.info(
                "modern_queue_drop room=%s peer=%s duration=%.3f rms=%s silence_ratio=%.3f queue_size=%s",
                self.room,
                self.peer_id,
                dropped.stats.duration_seconds,
                dropped.stats.rms,
                dropped.stats.silence_ratio,
                len(self.chunk_queue),
            )

        self.chunk_queue.append(
            AudioChunk(
                pcm=pcm,
                created_at=now,
                generation=self.generation,
                stats=stats,
                chunk_hash=chunk_hash,
            )
        )
        logger.info(
            "modern_chunk_queued room=%s peer=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s",
            self.room,
            self.peer_id,
            int(stats.duration_seconds * 1000),
            stats.rms,
            stats.silence_ratio,
            len(self.chunk_queue),
            0,
            0,
            0,
            False,
            self.config.source_language,
            self.config.target_language,
        )
        self._ensure_processor()

    def _ensure_processor(self) -> None:
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        while self.chunk_queue and not self.closed:
            chunk = self.chunk_queue.popleft()
            if time.monotonic() - chunk.created_at > REALTIME_RESULT_MAX_AGE_SECONDS:
                logger.info(
                    "modern_stale_chunk_discard room=%s peer=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s",
                    self.room,
                    self.peer_id,
                    int(chunk.stats.duration_seconds * 1000),
                    chunk.stats.rms,
                    chunk.stats.silence_ratio,
                    len(self.chunk_queue),
                    0,
                    True,
                    self.config.source_language,
                    self.config.target_language,
                )
                continue
            await self._process(chunk)

    async def _process(self, chunk: AudioChunk) -> None:
        try:
            started = time.perf_counter()
            try:
                text = await asyncio.wait_for(
                    transcribe_pcm(chunk.pcm, self.config),
                    timeout=REALTIME_STT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                stt_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "modern_stt_timeout_discard room=%s peer=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s",
                    self.room,
                    self.peer_id,
                    int(chunk.stats.duration_seconds * 1000),
                    chunk.stats.rms,
                    chunk.stats.silence_ratio,
                    len(self.chunk_queue),
                    stt_ms,
                    0,
                    stt_ms,
                    True,
                    self.config.source_language,
                    self.config.target_language,
                )
                return
            stt_ms = int((time.perf_counter() - started) * 1000)
            if not text:
                logger.info(
                    "modern_stt_no_speech room=%s peer=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s",
                    self.room,
                    self.peer_id,
                    int(chunk.stats.duration_seconds * 1000),
                    chunk.stats.rms,
                    chunk.stats.silence_ratio,
                    len(self.chunk_queue),
                    stt_ms,
                    0,
                    stt_ms,
                    False,
                    self.config.source_language,
                    self.config.target_language,
                )
                return
            if chunk.generation != self.generation:
                return
            now = time.monotonic()
            if transcript_key(text) == self.last_caption_text and now - self.last_caption_at < STT_REPEAT_WINDOW_SECONDS:
                logger.info(
                    "modern_stt_repeat_filtered room=%s peer=%s duration=%.3f rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s stt_text=%r",
                    self.room,
                    self.peer_id,
                    chunk.stats.duration_seconds,
                    chunk.stats.rms,
                    chunk.stats.silence_ratio,
                    len(self.chunk_queue),
                    stt_ms,
                    text[:80],
                )
                return
            self.last_caption_text = transcript_key(text)
            self.last_caption_at = now
            self.last_chunk_hash = chunk.chunk_hash
            if time.monotonic() - chunk.created_at > REALTIME_RESULT_MAX_AGE_SECONDS:
                total_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "modern_old_stt_result_discard room=%s peer=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s stt_text=%r",
                    self.room,
                    self.peer_id,
                    int(chunk.stats.duration_seconds * 1000),
                    chunk.stats.rms,
                    chunk.stats.silence_ratio,
                    len(self.chunk_queue),
                    stt_ms,
                    0,
                    total_ms,
                    True,
                    self.config.source_language,
                    self.config.target_language,
                    text[:80],
                )
                return

            translate_started = time.perf_counter()
            translated = await translate_text_fast(text, self.config.source_language, self.config.target_language)
            translate_ms = int((time.perf_counter() - translate_started) * 1000)
            total_ms = int((time.perf_counter() - started) * 1000)

            # FIX: Don't discard successful translations due to timeout
            if not translated:
                logger.info(
                    "modern_translation_failed room=%s peer=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s stt_text=%r",
                    self.room,
                    self.peer_id,
                    int(chunk.stats.duration_seconds * 1000),
                    chunk.stats.rms,
                    chunk.stats.silence_ratio,
                    len(self.chunk_queue),
                    stt_ms,
                    translate_ms,
                    total_ms,
                    total_ms > int(REALTIME_RESULT_MAX_AGE_SECONDS * 1000),
                    self.config.source_language,
                    self.config.target_language,
                    text[:80],
                )
                return

            if total_ms > 2000:
                logger.warning("slow_translation_sent room=%s peer=%s total_ms=%s", self.room, self.peer_id, total_ms)
            logger.info(
                "modern_caption room=%s peer=%s final=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s tts_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s text_len=%s translated_len=%s stt_text=%r",
                self.room,
                self.peer_id,
                True,
                int(chunk.stats.duration_seconds * 1000),
                chunk.stats.rms,
                chunk.stats.silence_ratio,
                len(self.chunk_queue),
                stt_ms,
                translate_ms,
                0,
                total_ms,
                False,
                self.config.source_language,
                self.config.target_language,
                len(text),
                len(translated),
                text[:80],
            )
            payload = {
                "type": "caption",
                "text": text,
                "translation": translated,
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "is_final": True,
                "timing_ms": {
                    "stt": stt_ms,
                    "translation": translate_ms,
                    "tts": 0,
                    "total": total_ms,
                },
            }
            await manager.send("translation", self.room, self.peer_id, payload)
        except Exception as exc:
            await manager.send("translation", self.room, self.peer_id, {"type": "error", "message": str(exc)})

    def _trim_buffer(self) -> None:
        max_bytes = int(self.config.bytes_per_second * REALTIME_MAX_BUFFER_SECONDS)
        if len(self.audio_buffer) > max_bytes:
            dropped_bytes = len(self.audio_buffer) - max_bytes
            del self.audio_buffer[: len(self.audio_buffer) - max_bytes]
            logger.info(
                "modern_buffer_trim room=%s peer=%s dropped_duration=%.3f buffer_duration=%.3f queue_size=%s",
                self.room,
                self.peer_id,
                dropped_bytes / self.config.bytes_per_second,
                len(self.audio_buffer) / self.config.bytes_per_second,
                len(self.chunk_queue),
            )


def normalize_source_lang(lang: str) -> str:
    """Normalize kaynak dili DeepL ve Whisper için"""
    if not lang:
        return "TR"

    lang = str(lang).upper().replace("_", "-")

    VALID_SOURCE_LANGS = {
        "AUTO": "AUTO",
        "EN": "EN", "EN-US": "EN", "EN-GB": "EN",
        "TR": "TR", "TR-TR": "TR",
        "DE": "DE", "NL": "NL", "FR": "FR", "ES": "ES",
        "RU": "RU", "UK": "UK", "ZH": "ZH", "KA": "KA", "AR": "AR",
        "IT": "IT", "PL": "PL", "JA": "JA", "KO": "KO",
        "PT": "PT", "PT-BR": "PT", "PT-PT": "PT",
        "CS": "CS", "SV": "SV", "DA": "DA", "FI": "FI",
        "EL": "EL", "HU": "HU", "NO": "NO", "RO": "RO",
        "SK": "SK", "SL": "SL",
    }

    return VALID_SOURCE_LANGS.get(lang, "TR")


def normalize_target_lang(lang: str) -> str:
    """Normalize hedef dili DeepL için"""
    if not lang:
        return "EN-US"

    lang = str(lang).upper().replace("_", "-")

    VALID_TARGET_LANGS = {
        "EN": "EN-US", "EN-US": "EN-US", "EN-GB": "EN-GB",
        "DE": "DE", "NL": "NL", "FR": "FR", "ES": "ES",
        "RU": "RU", "UK": "UK", "ZH": "ZH", "KA": "KA", "AR": "AR",
        "IT": "IT", "PL": "PL", "JA": "JA", "KO": "KO",
        "PT": "PT-BR", "PT-BR": "PT-BR", "PT-PT": "PT-PT",
        "TR": "TR",
        "CS": "CS", "SV": "SV", "DA": "DA", "FI": "FI",
        "EL": "EL", "HU": "HU", "NO": "NO", "RO": "RO",
        "SK": "SK", "SL": "SL",
    }

    return VALID_TARGET_LANGS.get(lang, "EN-US")


def whisper_lang(lang: str) -> str:
    if normalize_source_lang(lang) == "AUTO":
        return "auto"
    mapping = {
        "TR": "tr",
        "RU": "ru",
        "UK": "uk",
        "EN": "en",
        "EN-US": "en",
        "DE": "de",
        "NL": "nl",
        "AR": "ar",
        "ES": "es",
        "ZH": "zh",
        "KA": "ka",
        "FR": "fr",
        "IT": "it",
        "PT": "pt",
        "PL": "pl",
        "JA": "ja",
        "KO": "ko",
        "CS": "cs",
        "SV": "sv",
        "DA": "da",
        "FI": "fi",
        "EL": "el",
        "HU": "hu",
        "NO": "no",
        "RO": "ro",
        "SK": "sk",
        "SL": "sl",
    }
    return mapping.get(normalize_source_lang(lang), "en")


def google_lang(lang: str) -> str:
    target = normalize_target_lang(lang)
    GOOGLE_LANGS = {
        "EN-US": "en", "EN-GB": "en",
        "DE": "de", "NL": "nl", "FR": "fr", "ES": "es",
        "RU": "ru", "UK": "uk", "ZH": "zh-CN",
        "KA": "ka", "AR": "ar", "IT": "it", "PL": "pl",
        "JA": "ja", "KO": "ko",
        "PT-BR": "pt", "PT-PT": "pt",
        "TR": "tr",
        "CS": "cs", "SV": "sv", "DA": "da", "FI": "fi",
        "EL": "el", "HU": "hu", "NO": "no", "RO": "ro",
        "SK": "sk", "SL": "sl",
    }
    return GOOGLE_LANGS.get(target, "en")


def clean_transcript(text: str) -> str:
    cleaned = " ".join((text or "").replace("\n", " ").split()).strip()
    return cleaned.strip(" -–—")


def transcript_key(text: str) -> str:
    normalized = clean_transcript(text).lower()
    normalized = normalized.translate(str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocigusoc"))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def has_repeated_short_words(text: str) -> bool:
    words = transcript_key(text).split()
    if len(words) < 4:
        return False
    if len(words) % 2 == 0 and words[: len(words) // 2] == words[len(words) // 2 :]:
        return True
    unique = set(words)
    return len(unique) <= 2 and len(words) / max(len(unique), 1) >= 2.0


def has_repeated_char_garbage(text: str) -> bool:
    stt_text = clean_transcript(text)
    if len(stt_text) < 8:
        return False
    if stt_text and stt_text[0].strip() and stt_text.count(stt_text[0]) > 20:
        return True
    for char in set(stt_text):
        if char.strip() and stt_text.count(char) > 20:
            return True
    return False


def is_low_value_transcript(text: str) -> bool:
    cleaned = clean_transcript(text)
    dotted = cleaned.lower().strip(" .!?")
    normalized = transcript_key(cleaned)
    if dotted in LOW_VALUE_TRANSCRIPTS or normalized in LOW_VALUE_TRANSCRIPTS:
        return True
    if len(normalized) < 2:
        return True
    if re.fullmatch(r"(m\s*k|m\s*k\s*)+", normalized):
        return True
    if has_repeated_char_garbage(cleaned):
        return True
    return has_repeated_short_words(cleaned)


def transcription_prompt(source_lang: str, previous_text: str = "") -> str:
    lang = whisper_lang(source_lang)
    prompt = TRANSCRIPTION_PROMPTS.get(lang, TRANSCRIPTION_PROMPTS["en"])
    previous_text = clean_transcript(previous_text)
    if previous_text:
        return f"{prompt} Previous context: {previous_text[-220:]}"
    return prompt


def stable_segment_text(segments, *, relaxed: bool = False) -> tuple[str, float]:
    accepted: list[str] = []
    confidences: list[float] = []

    for segment in segments:
        text = clean_transcript(getattr(segment, "text", ""))
        if not text:
            continue

        no_speech_prob = float(getattr(segment, "no_speech_prob", 0) or 0)
        avg_logprob = float(getattr(segment, "avg_logprob", 0) or 0)
        no_speech_limit = 0.92 if relaxed else WHISPER_NO_SPEECH_THRESHOLD
        log_prob_limit = -1.15 if relaxed else WHISPER_LOG_PROB_THRESHOLD
        if no_speech_prob > no_speech_limit:
            continue
        if avg_logprob < log_prob_limit:
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


async def translate_text_fast(text: str, source_lang: str, target_lang: str) -> str:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(translate_text_value, text, source_lang, target_lang),
            timeout=REALTIME_TRANSLATE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "translate_timeout source=%s target=%s timeout_ms=%s text_len=%s",
            source_lang,
            target_lang,
            int(REALTIME_TRANSLATE_TIMEOUT_SECONDS * 1000),
            len(text),
        )
        return ""
    except Exception as exc:
        logger.warning("translate_exception source=%s target=%s error=%s", source_lang, target_lang, exc)
        return ""


async def safe_send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass


def is_valid_room_code(code: Any) -> bool:
    return isinstance(code, str) and bool(ROOM_CODE_RE.fullmatch(code.strip()))


def client_fingerprint(websocket: WebSocket) -> str:
    forwarded = websocket.headers.get("x-forwarded-for", "")
    client_host = websocket.client.host if websocket.client else "unknown"
    raw = f"{forwarded.split(',')[0].strip() or client_host}:{websocket.headers.get('user-agent', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def join_attempt_key(websocket: WebSocket, room_id: str) -> str:
    return f"join:{room_id}:{client_fingerprint(websocket)}"


async def join_is_limited(websocket: WebSocket, room_id: str) -> tuple[bool, int]:
    key = join_attempt_key(websocket, room_id)
    cooldown_key = f"{key}:cooldown"
    now = time.time()
    if redis_client is not None:
        cooldown_until = await redis_client.get(cooldown_key)
        if cooldown_until and float(cooldown_until) > now:
            return True, max(1, int(float(cooldown_until) - now))
        count = int(await redis_client.get(key) or "0")
        return count >= JOIN_ATTEMPT_LIMIT, JOIN_COOLDOWN_SECONDS

    cooldown_until = legacy_join_cooldowns.get(key, 0)
    if cooldown_until > now:
        return True, max(1, int(cooldown_until - now))
    window_start = now - JOIN_ATTEMPT_WINDOW_SECONDS
    attempts = [item for item in legacy_join_attempts.get(key, []) if item >= window_start]
    legacy_join_attempts[key] = attempts
    return len(attempts) >= JOIN_ATTEMPT_LIMIT, JOIN_COOLDOWN_SECONDS


async def record_failed_join(websocket: WebSocket, room_id: str) -> int:
    key = join_attempt_key(websocket, room_id)
    cooldown_key = f"{key}:cooldown"
    now = time.time()
    if redis_client is not None:
        count = await redis_client.incr(key)
        await redis_client.expire(key, JOIN_ATTEMPT_WINDOW_SECONDS)
        if int(count) >= JOIN_ATTEMPT_LIMIT:
            cooldown_until = now + JOIN_COOLDOWN_SECONDS
            await redis_client.setex(cooldown_key, JOIN_COOLDOWN_SECONDS, str(cooldown_until))
        return int(count)

    attempts = legacy_join_attempts.setdefault(key, [])
    attempts.append(now)
    legacy_join_attempts[key] = [
        item for item in attempts if item >= now - JOIN_ATTEMPT_WINDOW_SECONDS
    ]
    if len(legacy_join_attempts[key]) >= JOIN_ATTEMPT_LIMIT:
        legacy_join_cooldowns[key] = now + JOIN_COOLDOWN_SECONDS
    return len(legacy_join_attempts[key])


async def clear_join_attempts(websocket: WebSocket, room_id: str) -> None:
    key = join_attempt_key(websocket, room_id)
    if redis_client is not None:
        await redis_client.delete(key, f"{key}:cooldown")
        return
    legacy_join_attempts.pop(key, None)
    legacy_join_cooldowns.pop(key, None)


async def reject_join_attempt(websocket: WebSocket, room_id: str) -> None:
    attempts = await record_failed_join(websocket, room_id)
    logger.warning("legacy_join_rejected room=%s attempts=%s", room_id, attempts)
    await safe_send(
        websocket,
        {
            "type": "error",
            "message": "Oda bulunamadi veya kod hatali",
        },
    )


def cleanup_expired_legacy_rooms() -> None:
    now = time.time()
    expired = [
        room_id
        for room_id, room in legacy_signal_rooms.items()
        if float(room.get("expiresAt", 0)) <= now
    ]
    for room_id in expired:
        logger.info("legacy_room_expired room=%s", room_id)
        legacy_signal_rooms.pop(room_id, None)


def room_payload(room_id: str) -> dict[str, Any]:
    room = legacy_signal_rooms[room_id]
    return {
        "room": room_id,
        "ownerId": room["ownerId"],
        "capacity": room["capacity"],
        "memberCount": len(room["members"]),
        "privateCode": room["privateCode"],
        "expiresAt": room.get("expiresAt"),
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


def wav_rms(path: str) -> int:
    try:
        with wave.open(path, "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
        return pcm_rms(pcm)
    except Exception:
        return 0


def pcm_silence_ratio(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    threshold: int = VAD_RMS_THRESHOLD,
) -> float:
    if not pcm or sample_rate <= 0 or channels <= 0:
        return 1.0
    samples_per_frame = max(channels, int(sample_rate * channels * STT_TRIM_FRAME_MS / 1000))
    samples_per_frame -= samples_per_frame % channels
    frame_bytes = max(channels * SAMPLE_WIDTH_BYTES, samples_per_frame * SAMPLE_WIDTH_BYTES)
    usable_length = len(pcm) - (len(pcm) % frame_bytes)
    if usable_length <= 0:
        return 1.0
    total = 0
    silent = 0
    for offset in range(0, usable_length, frame_bytes):
        total += 1
        if pcm_rms(pcm[offset : offset + frame_bytes]) < threshold:
            silent += 1
    return round(silent / max(total, 1), 3)


def pcm_audio_stats(pcm: bytes, sample_rate: int, channels: int) -> AudioStats:
    if not pcm or sample_rate <= 0 or channels <= 0:
        return AudioStats(sample_rate=sample_rate, channels=channels)
    bytes_per_second = sample_rate * channels * SAMPLE_WIDTH_BYTES
    return AudioStats(
        duration_seconds=round(len(pcm) / bytes_per_second, 3),
        rms=pcm_rms(pcm),
        silence_ratio=pcm_silence_ratio(pcm, sample_rate, channels),
        sample_rate=sample_rate,
        channels=channels,
    )


def wav_audio_stats(path: str) -> AudioStats:
    try:
        pcm, sample_rate, channels = read_wav_pcm(path)
    except Exception:
        return AudioStats()
    if not pcm or sample_rate <= 0 or channels <= 0:
        return AudioStats(sample_rate=sample_rate, channels=channels)
    return pcm_audio_stats(pcm, sample_rate, channels)


def should_skip_audio_for_stt(stats: AudioStats) -> bool:
    active_ratio = 1.0 - stats.silence_ratio
    return (
        stats.duration_seconds < STT_MIN_AUDIO_SECONDS
        or stats.rms < STT_MIN_RMS
        or active_ratio < STT_MIN_ACTIVE_RATIO
    )


def is_hard_silence_for_stt(stats: AudioStats) -> bool:
    active_ratio = 1.0 - stats.silence_ratio
    return (
        stats.duration_seconds < STT_MIN_AUDIO_SECONDS
        or (stats.rms < STT_HARD_SILENCE_RMS and active_ratio < 0.02)
    )


def read_wav_pcm(path: str) -> tuple[bytes, int, int]:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        pcm = wav.readframes(wav.getnframes())
    if sample_width != SAMPLE_WIDTH_BYTES:
        return b"", sample_rate, channels
    return pcm, sample_rate, channels


def apply_gain_to_pcm(pcm: bytes, gain: float) -> bytes:
    if gain <= 1.0 or len(pcm) < SAMPLE_WIDTH_BYTES:
        return pcm
    if len(pcm) % SAMPLE_WIDTH_BYTES:
        pcm = pcm[: -(len(pcm) % SAMPLE_WIDTH_BYTES)]
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    amplified = [
        max(-32768, min(32767, int(sample * gain)))
        for sample in samples
    ]
    return struct.pack(f"<{len(amplified)}h", *amplified)


def prepare_audio_properly(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    try:
        if not audio_bytes:
            logger.warning("prepare_audio_empty_input")
            return audio_bytes
        if len(audio_bytes) % SAMPLE_WIDTH_BYTES:
            audio_bytes = audio_bytes[: -(len(audio_bytes) % SAMPLE_WIDTH_BYTES)]

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if audio.size == 0:
            return audio_bytes

        audio = audio / 32768.0
        audio = audio - float(np.mean(audio))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 0:
            audio = audio * (0.3 / rms)
        audio = np.tanh(audio)
        audio = (audio * 32767).astype(np.int16)
        return audio.tobytes()
    except Exception as e:
        logger.error("audio_preprocessing_error %s", e)
        return audio_bytes


def trim_pcm_for_stt(pcm: bytes, sample_rate: int, channels: int, original_rms: int) -> tuple[bytes, int]:
    if len(pcm) < sample_rate * channels * SAMPLE_WIDTH_BYTES // 4:
        return pcm, 0
    if sample_rate <= 0 or channels <= 0:
        return pcm, 0

    samples_per_frame = max(channels, int(sample_rate * channels * STT_TRIM_FRAME_MS / 1000))
    samples_per_frame -= samples_per_frame % channels
    frame_bytes = max(channels * SAMPLE_WIDTH_BYTES, samples_per_frame * SAMPLE_WIDTH_BYTES)
    threshold = max(STT_HARD_SILENCE_RMS, min(VAD_RMS_THRESHOLD, int(max(original_rms, 1) * 0.45)))

    first_voice = -1
    last_voice = -1
    usable_length = len(pcm) - (len(pcm) % frame_bytes)
    for offset in range(0, usable_length, frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if pcm_rms(frame) >= threshold:
            if first_voice < 0:
                first_voice = offset
            last_voice = offset + frame_bytes

    if first_voice < 0 or last_voice <= first_voice:
        return pcm, 0

    pad_bytes = int(sample_rate * channels * SAMPLE_WIDTH_BYTES * STT_TRIM_PAD_MS / 1000)
    pad_bytes -= pad_bytes % (channels * SAMPLE_WIDTH_BYTES)
    start = max(0, first_voice - pad_bytes)
    end = min(len(pcm), last_voice + pad_bytes)
    end -= (end - start) % (channels * SAMPLE_WIDTH_BYTES)
    if end <= start:
        return pcm, 0

    trimmed = pcm[start:end]
    if len(trimmed) < sample_rate * channels * SAMPLE_WIDTH_BYTES // 4:
        return pcm, 0

    trimmed_ms = int((len(pcm) - len(trimmed)) / (sample_rate * channels * SAMPLE_WIDTH_BYTES) * 1000)
    if trimmed_ms < 120:
        return pcm, 0
    return trimmed, trimmed_ms


def normalize_wav_for_stt(path: str) -> tuple[str, int, float, int]:
    try:
        pcm, sample_rate, channels = read_wav_pcm(path)
    except Exception:
        return path, 0, 1.0, 0

    original_rms = pcm_rms(pcm)
    if original_rms <= 0:
        return path, original_rms, 1.0, 0

    prepared_pcm, trimmed_ms = trim_pcm_for_stt(pcm, sample_rate, channels, original_rms)
    if trimmed_ms > 0:
        return write_wav(prepared_pcm, sample_rate, channels), original_rms, 1.0, trimmed_ms
    return path, original_rms, 1.0, 0


def write_wav(pcm: bytes, sample_rate: int, channels: int) -> str:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    file.close()
    with wave.open(file.name, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return file.name


def wav_bytes_from_pcm(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def normalize_pcm_for_stt(pcm: bytes, sample_rate: int, channels: int) -> tuple[bytes, int, float, int]:
    original_rms = pcm_rms(pcm)
    if original_rms <= 0:
        return pcm, original_rms, 1.0, 0

    prepared_pcm, trimmed_ms = trim_pcm_for_stt(pcm, sample_rate, channels, original_rms)
    return prepared_pcm, original_rms, 1.0, trimmed_ms


async def transcribe_audio_hybrid(
    audio_bytes: bytes,
    source_lang: str,
    whisper_model,
    groq_client,
) -> Optional[str]:
    logger.info("stt_whisper_attempt lang=%s", source_lang)

    stt_text = await _transcribe_with_whisper(
        audio_bytes,
        source_lang,
        whisper_model,
    )

    if _is_valid_transcription(stt_text):
        logger.info("stt_whisper_success lang=%s text_len=%s", source_lang, len(stt_text))
        return stt_text

    if stt_text:
        logger.warning("stt_whisper_garbage lang=%s text=%s", source_lang, stt_text[:50])
    else:
        logger.warning("stt_whisper_empty lang=%s", source_lang)

    logger.info("stt_groq_fallback lang=%s", source_lang)

    stt_text = await _transcribe_with_groq(
        audio_bytes,
        source_lang,
        groq_client,
    )

    if _is_valid_transcription(stt_text):
        logger.info("stt_groq_success_fallback lang=%s text_len=%s", source_lang, len(stt_text))
        return stt_text

    logger.error("stt_hybrid_complete_failure lang=%s", source_lang)
    return ""


async def _transcribe_with_whisper(
    audio_bytes: bytes,
    source_lang: str,
    whisper_model,
) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = whisper_model.transcribe(
                tmp_path,
                language=_get_whisper_lang(source_lang),
                temperature=0.0,
                beam_size=5,
                best_of=1,
                vad_filter=False,
            )

            stt_text = " ".join([seg.text for seg in segments]).strip()
            return stt_text if stt_text else None

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as e:
        logger.error("whisper_transcribe_error %s", e)
        return None


async def _transcribe_with_groq(
    audio_bytes: bytes,
    source_lang: str,
    groq_client,
) -> Optional[str]:
    if groq_client is None:
        logger.warning("groq_transcribe_unavailable missing_api_key")
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                response = groq_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=_get_groq_lang(source_lang),
                    temperature=0.0,
                )

            stt_text = response.text.strip()
            return stt_text if stt_text else None

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as e:
        logger.error("groq_transcribe_error %s", e)
        return None


def _is_valid_transcription(text: Optional[str]) -> bool:
    if not text or len(text) < 2:
        return False

    if is_low_value_transcript(text):
        return False

    garbage_patterns = [
        "altyazÄ± m.k.",
        "the system is",
        "subtitle",
    ]

    if text.lower() in garbage_patterns:
        return False

    for char in set(text):
        if text.count(char) > 20:
            return False

    return True


def _get_whisper_lang(lang: str) -> Optional[str]:
    lang_map = {
        "TR": "tr",
        "EN": "en",
        "DE": "de",
        "FR": "fr",
        "ES": "es",
        "RU": "ru",
        "UK": "uk",
        "ZH": "zh",
        "JA": "ja",
        "KO": "ko",
        "IT": "it",
        "PT": "pt",
        "PL": "pl",
        "NL": "nl",
        "AR": "ar",
    }
    normalized = normalize_source_lang(lang)
    if normalized == "AUTO":
        return None
    return lang_map.get(normalized.upper(), "en")


def _get_groq_lang(lang: str) -> Optional[str]:
    lang_map = {
        "TR": "tr",
        "EN": "en",
        "DE": "de",
        "FR": "fr",
        "ES": "es",
        "RU": "ru",
        "UK": "uk",
        "ZH": "zh",
        "JA": "ja",
        "KO": "ko",
        "IT": "it",
        "PT": "pt",
        "PL": "pl",
        "NL": "nl",
        "AR": "ar",
    }
    normalized = normalize_source_lang(lang)
    if normalized == "AUTO":
        return None
    return lang_map.get(normalized.upper(), "en")


async def transcribe_pcm(pcm: bytes, config: TranslationConfig) -> str:
    return await transcribe_pcm_bytes(pcm, config)


async def transcribe_pcm_bytes(pcm: bytes, config: TranslationConfig, previous_text: str = "") -> str:
    source_lang = normalize_source_lang(config.source_language)
    stt_lang = whisper_lang(source_lang)
    original_stats = pcm_audio_stats(pcm, config.sample_rate, config.channels)
    if config.sample_rate != SAMPLE_RATE or config.channels != CHANNELS:
        logger.info(
            "whisper_stt_bad_format lang=%s sample_rate=%s channels=%s chunk_ms=%s rms=%s silence_ratio=%.3f",
            stt_lang,
            config.sample_rate,
            config.channels,
            int(original_stats.duration_seconds * 1000),
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""
    if is_hard_silence_for_stt(original_stats):
        logger.info(
            "hybrid_stt_hard_silence_skip lang=%s chunk_ms=%s rms=%s silence_ratio=%.3f",
            stt_lang,
            int(original_stats.duration_seconds * 1000),
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""

    prepared_pcm, _original_rms, gain, trimmed_ms = normalize_pcm_for_stt(
        pcm,
        config.sample_rate,
        config.channels,
    )
    prepared_pcm = prepare_audio_properly(prepared_pcm, config.sample_rate)
    prepared_stats = pcm_audio_stats(prepared_pcm, config.sample_rate, config.channels)
    try:
        text = await transcribe_audio_hybrid(
            wav_bytes_from_pcm(prepared_pcm, config.sample_rate, config.channels),
            source_lang=config.source_language,
            whisper_model=WHISPER_MODEL,
            groq_client=GROQ_CLIENT,
        )
        if is_low_value_transcript(text):
            logger.info(
                "hybrid_stt_low_value lang=%s chunk_ms=%s rms=%s silence_ratio=%.3f stt_text=%r",
                stt_lang,
                int(original_stats.duration_seconds * 1000),
                original_stats.rms,
                original_stats.silence_ratio,
                text[:80],
            )
            return ""
        logger.info(
            "hybrid_stt_ok lang=%s chunk_ms=%s rms=%s silence_ratio=%.3f prepared_chunk_ms=%s prepared_rms=%s gain=%.2f trimmed_ms=%s stt_text=%r",
            stt_lang,
            int(original_stats.duration_seconds * 1000),
            original_stats.rms,
            original_stats.silence_ratio,
            int(prepared_stats.duration_seconds * 1000),
            prepared_stats.rms,
            gain,
            trimmed_ms,
            text[:80],
        )
        return text
    except Exception as e:
        logger.error("hybrid_stt_exception %s", e)
        return ""


async def transcribe_wav(path: str, source_lang: str, previous_text: str = "") -> str:
    stt_lang = whisper_lang(source_lang)
    original_stats = wav_audio_stats(path)
    if original_stats.sample_rate != SAMPLE_RATE or original_stats.channels != CHANNELS:
        logger.info(
            "whisper_stt_bad_format lang=%s sample_rate=%s channels=%s duration=%.3f rms=%s silence_ratio=%.3f",
            stt_lang,
            original_stats.sample_rate,
            original_stats.channels,
            original_stats.duration_seconds,
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""
    if is_hard_silence_for_stt(original_stats):
        logger.info(
            "hybrid_stt_hard_silence_skip lang=%s duration=%.3f rms=%s silence_ratio=%.3f",
            stt_lang,
            original_stats.duration_seconds,
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""

    stt_path = path
    try:
        stt_path, original_rms, gain, trimmed_ms = normalize_wav_for_stt(path)
        prepared_stats = wav_audio_stats(stt_path)
        if os.path.getsize(stt_path) < 8000:
            return ""

        prepared_pcm, prepared_sample_rate, prepared_channels = read_wav_pcm(stt_path)
        prepared_pcm = prepare_audio_properly(prepared_pcm, prepared_sample_rate)
        text = await transcribe_audio_hybrid(
            wav_bytes_from_pcm(prepared_pcm, prepared_sample_rate, prepared_channels),
            source_lang=source_lang,
            whisper_model=WHISPER_MODEL,
            groq_client=GROQ_CLIENT,
        )
        if is_low_value_transcript(text):
            logger.info(
                "hybrid_stt_low_value lang=%s duration=%.3f rms=%s silence_ratio=%.3f stt_text=%r",
                stt_lang,
                original_stats.duration_seconds,
                original_stats.rms,
                original_stats.silence_ratio,
                text[:80],
            )
            return ""

        logger.info(
            "hybrid_stt_ok lang=%s duration=%.3f rms=%s silence_ratio=%.3f prepared_duration=%.3f prepared_rms=%s gain=%.2f trimmed_ms=%s stt_text=%r",
            stt_lang,
            original_stats.duration_seconds,
            original_stats.rms,
            original_stats.silence_ratio,
            prepared_stats.duration_seconds,
            prepared_stats.rms,
            gain,
            trimmed_ms,
            text[:80],
        )
        return text

    except Exception as e:
        logger.error("hybrid_stt_exception %s", e)
        return ""
    finally:
        if stt_path != path:
            try:
                os.remove(stt_path)
            except OSError:
                pass


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
        "partial_transcribe_interval_seconds": PARTIAL_TRANSCRIBE_INTERVAL_SECONDS,
        "rolling_transcribe_window_seconds": ROLLING_TRANSCRIBE_WINDOW_SECONDS,
        "final_silence_seconds": FINAL_SILENCE_SECONDS,
        "min_transcribe_seconds": MIN_TRANSCRIBE_SECONDS,
        "stt_vad_min_silence_ms": WHISPER_RUNTIME.vad_min_silence_ms,
        "stt_vad_speech_pad_ms": WHISPER_RUNTIME.vad_speech_pad_ms,
        "vad_rms_threshold": VAD_RMS_THRESHOLD,
        "stt_hard_silence_rms": STT_HARD_SILENCE_RMS,
        "stt_target_rms": STT_TARGET_RMS,
        "stt_fast_gain_threshold": STT_FAST_GAIN_THRESHOLD,
        "stt_trim_frame_ms": STT_TRIM_FRAME_MS,
        "stt_trim_pad_ms": STT_TRIM_PAD_MS,
        "deepl_configured": bool(DEEPL_AUTH_KEY),
        "redis_configured": redis_client is not None,
        "room_code_length": ROOM_CODE_LENGTH,
        "room_ttl_seconds": ROOM_TTL_SECONDS,
        "join_attempt_limit": JOIN_ATTEMPT_LIMIT,
        "join_attempt_window_seconds": JOIN_ATTEMPT_WINDOW_SECONDS,
        "join_cooldown_seconds": JOIN_COOLDOWN_SECONDS,
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
                    if payload.get("audio_format", "pcm16") != "pcm16":
                        await websocket.send_json({"type": "error", "message": "audio_format must be pcm16"})
                        continue
                    session.update_config(config)
                elif payload.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            chunk = message.get("bytes")
            if chunk:
                await session.add_audio(chunk)
    except WebSocketDisconnect:
        logger.info("translate_client_disconnect room=%s peer=%s", room, peer_id)
        pending = [t for t in [session.partial_task, session.final_task] if t and not t.done()]
        if pending:
            try:
                done, remaining = await asyncio.wait(pending, timeout=2.0)
                for task in remaining:
                    task.cancel()
            except Exception as e:
                logger.warning("error_waiting_pending_tasks room=%s peer=%s", room, peer_id)
    except RuntimeError as e:
        logger.error("translate_runtime_error room=%s peer=%s error=%s", room, peer_id, e)
    except Exception as e:
        logger.exception("translate_unexpected_error room=%s peer=%s", room, peer_id)
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
            cleanup_expired_legacy_rooms()
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
                if not is_valid_room_code(private_code):
                    await safe_send(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Oda kodu en az {ROOM_CODE_LENGTH} alphanumeric karakter olmali",
                        },
                    )
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
                    "expiresAt": time.time() + ROOM_TTL_SECONDS,
                }
                legacy_client_info[websocket]["room"] = room_id
                await safe_send(websocket, {"type": "room_created", **room_payload(room_id)})
                await broadcast_room_state(room_id)
                continue

            if msg_type == "request_join":
                room_id = data.get("room")
                private_code = data.get("privateCode")
                if not room_id or not is_valid_room_code(private_code):
                    await reject_join_attempt(websocket, str(room_id or "missing"))
                    continue
                limited, retry_after = await join_is_limited(websocket, room_id)
                if limited:
                    await safe_send(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Cok fazla deneme. {retry_after} saniye sonra tekrar dene.",
                            "retryAfterSeconds": retry_after,
                        },
                    )
                    continue
                if room_id not in legacy_signal_rooms:
                    await reject_join_attempt(websocket, room_id)
                    continue
                room = legacy_signal_rooms[room_id]
                if room["privateCode"] != private_code:
                    await reject_join_attempt(websocket, room_id)
                    continue
                if len(room["members"]) >= room["capacity"]:
                    await safe_send(websocket, {"type": "error", "message": "Oda dolu"})
                    continue
                await clear_join_attempts(websocket, room_id)
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
    previous_sent_text = ""
    previous_sent_at = 0.0
    try:
        while True:
            request_started = time.perf_counter()
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") == "ping":
                await safe_send(websocket, {"type": "pong"})
                continue
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
                temp_write_started = time.perf_counter()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as file:
                    file.write(audio_bytes)
                    temp_path = file.name
                temp_write_ms = int((time.perf_counter() - temp_write_started) * 1000)
                audio_stats = wav_audio_stats(temp_path)
                if is_hard_silence_for_stt(audio_stats):
                    total_ms = int((time.perf_counter() - request_started) * 1000)
                    logger.info(
                        "legacy_hard_silence_skip bytes=%s source=%s target=%s duration=%.3f rms=%s silence_ratio=%.3f total_ms=%s",
                        len(audio_bytes),
                        source_lang,
                        target_lang,
                        audio_stats.duration_seconds,
                        audio_stats.rms,
                        audio_stats.silence_ratio,
                        total_ms,
                    )
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                    await safe_send(
                        websocket,
                        {
                            "noSpeech": True,
                            "audioBytes": len(audio_bytes),
                            "format": "pcm16wav",
                            "sampleRate": 16000,
                            "timingMs": {
                                "decode": decode_ms,
                                "tempWrite": temp_write_ms,
                                "stt": 0,
                                "tts": 0,
                                "total": total_ms,
                            },
                        },
                    )
                    continue
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
                        "legacy_stt_no_speech bytes=%s source=%s target=%s duration=%.3f rms=%s silence_ratio=%.3f stt_ms=%s total_ms=%s stt_text=%r",
                        len(audio_bytes),
                        source_lang,
                        target_lang,
                        audio_stats.duration_seconds,
                        audio_stats.rms,
                        audio_stats.silence_ratio,
                        stt_ms,
                        total_ms,
                        text[:80],
                    )
                    await safe_send(
                        websocket,
                        {
                            "noSpeech": True,
                            "audioBytes": len(audio_bytes),
                            "format": "pcm16wav",
                            "sampleRate": 16000,
                            "timingMs": {
                                "decode": decode_ms,
                                "tempWrite": temp_write_ms,
                                "stt": stt_ms,
                                "tts": 0,
                                "total": total_ms,
                            },
                        },
                    )
                    continue
                now = time.monotonic()
                if (
                    transcript_key(text) == previous_sent_text
                    and now - previous_sent_at < STT_REPEAT_WINDOW_SECONDS
                ):
                    total_ms = int((time.perf_counter() - request_started) * 1000)
                    logger.info(
                        "legacy_stt_repeat_filtered bytes=%s source=%s target=%s duration=%.3f rms=%s silence_ratio=%.3f stt_ms=%s total_ms=%s stt_text=%r",
                        len(audio_bytes),
                        source_lang,
                        target_lang,
                        audio_stats.duration_seconds,
                        audio_stats.rms,
                        audio_stats.silence_ratio,
                        stt_ms,
                        total_ms,
                        text[:80],
                    )
                    await safe_send(
                        websocket,
                        {
                            "noSpeech": True,
                            "audioBytes": len(audio_bytes),
                            "format": "pcm16wav",
                            "sampleRate": 16000,
                            "timingMs": {
                                "decode": decode_ms,
                                "tempWrite": temp_write_ms,
                                "stt": stt_ms,
                                "tts": 0,
                                "total": total_ms,
                            },
                        },
                    )
                    continue
                translate_started = time.perf_counter()
                translated = translate_text_value(text, source_lang, target_lang)
                translate_ms = int((time.perf_counter() - translate_started) * 1000)
                total_ms = int((time.perf_counter() - request_started) * 1000)
                previous_text = clean_transcript(f"{previous_text} {text}")[-500:]
                previous_sent_text = transcript_key(text)
                previous_sent_at = time.monotonic()
                logger.info(
                    "legacy_caption bytes=%s source=%s target=%s duration=%.3f rms=%s silence_ratio=%.3f temp_write_ms=%s stt_ms=%s translate_ms=%s tts_ms=%s total_ms=%s text_len=%s translated_len=%s stt_text=%r",
                    len(audio_bytes),
                    source_lang,
                    target_lang,
                    audio_stats.duration_seconds,
                    audio_stats.rms,
                    audio_stats.silence_ratio,
                    temp_write_ms,
                    stt_ms,
                    translate_ms,
                    0,
                    total_ms,
                    len(text),
                    len(translated),
                    text[:80],
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
                            "tempWrite": temp_write_ms,
                            "stt": stt_ms,
                            "translation": translate_ms,
                            "tts": 0,
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
