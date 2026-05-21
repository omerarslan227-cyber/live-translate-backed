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
from typing import Any
from uuid import uuid4

import deepl
import httpx
import uvicorn
from deep_translator import GoogleTranslator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE") or os.getenv("WHISPER_MODEL", "small")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
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
REALTIME_CHUNK_TARGET_SECONDS = float(os.getenv("REALTIME_CHUNK_TARGET_SECONDS", "0.45"))
REALTIME_CHUNK_MAX_SECONDS = float(os.getenv("REALTIME_CHUNK_MAX_SECONDS", "0.55"))
REALTIME_MAX_BUFFER_SECONDS = float(os.getenv("REALTIME_MAX_BUFFER_SECONDS", "3.2"))
REALTIME_MAX_QUEUE_SIZE = int(os.getenv("REALTIME_MAX_QUEUE_SIZE", "2"))
REALTIME_STT_TIMEOUT_SECONDS = float(os.getenv("REALTIME_STT_TIMEOUT_SECONDS", "1.1"))
REALTIME_TRANSLATE_TIMEOUT_SECONDS = float(os.getenv("REALTIME_TRANSLATE_TIMEOUT_SECONDS", "0.5"))
REALTIME_RESULT_MAX_AGE_SECONDS = float(os.getenv("REALTIME_RESULT_MAX_AGE_SECONDS", "1.2"))
REALTIME_PARTIAL_MIN_SECONDS = float(os.getenv("REALTIME_PARTIAL_MIN_SECONDS", "0.7"))
REALTIME_PARTIAL_INTERVAL_SECONDS = float(os.getenv("REALTIME_PARTIAL_INTERVAL_SECONDS", "0.65"))
REALTIME_UTTERANCE_MIN_SECONDS = float(os.getenv("REALTIME_UTTERANCE_MIN_SECONDS", "1.2"))
REALTIME_UTTERANCE_MAX_SECONDS = float(os.getenv("REALTIME_UTTERANCE_MAX_SECONDS", "3.0"))
REALTIME_FINAL_SILENCE_SECONDS = float(os.getenv("REALTIME_FINAL_SILENCE_SECONDS", "0.65"))
REALTIME_GAIN_TARGET_RMS = int(os.getenv("REALTIME_GAIN_TARGET_RMS", "1200"))
REALTIME_MAX_GAIN = float(os.getenv("REALTIME_MAX_GAIN", "8.0"))
REDIS_URL = os.getenv("REDIS_URL", "")
ROOM_CODE_LENGTH = int(os.getenv("ROOM_CODE_LENGTH", "8"))
ROOM_TTL_SECONDS = int(os.getenv("ROOM_TTL_SECONDS", "7200"))
JOIN_ATTEMPT_LIMIT = int(os.getenv("JOIN_ATTEMPT_LIMIT", "5"))
JOIN_ATTEMPT_WINDOW_SECONDS = int(os.getenv("JOIN_ATTEMPT_WINDOW_SECONDS", "300"))
JOIN_COOLDOWN_SECONDS = int(os.getenv("JOIN_COOLDOWN_SECONDS", "60"))
ROOM_CODE_RE = re.compile(rf"^[A-Za-z0-9]{{{ROOM_CODE_LENGTH},}}$")

translator = deepl.Translator(DEEPL_AUTH_KEY, server_url=DEEPL_API_URL or None) if DEEPL_AUTH_KEY else None


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

GROQ_LANGUAGE_MAP = {
    "tr": "tr",
    "en": "en",
    "ru": "ru",
    "de": "de",
    "nl": "nl",
    "uk": "uk",
    "ar": "ar",
    "es": "es",
    "zh": "zh",
    "ka": "ka",
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


@dataclass(frozen=True)
class RealtimeAudioPrep:
    pcm: bytes
    raw_rms: int
    normalized_rms: int
    gain_applied: float
    clipped: bool
    silence_ratio: float
    vad_decision: bool
    noise_floor: float
    vad_threshold: int


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
        self.last_voice_at = 0.0
        self.utterance_started_at = 0.0
        self.last_partial_at = 0.0
        self.last_caption_text = ""
        self.last_caption_at = 0.0
        self.last_final_text = ""
        self.last_partial_text = ""
        self.in_speech = False
        self.generation = 0
        self.utterance_id = 0
        self.partial_task: asyncio.Task[None] | None = None
        self.final_task: asyncio.Task[None] | None = None
        self.closed = False
        self.noise_floor = float(STT_MIN_RMS)

    def update_config(self, config: TranslationConfig) -> None:
        self.config = self._fixed_audio_config(config)
        self.audio_buffer.clear()
        self.last_voice_at = 0.0
        self.utterance_started_at = 0.0
        self.last_partial_at = 0.0
        self.last_caption_text = ""
        self.last_partial_text = ""
        self.in_speech = False
        self.generation += 1
        self.utterance_id += 1

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
        cleaned = clean_pcm(chunk)
        if not cleaned:
            return

        now = time.monotonic()
        prep = self._prepare_realtime_audio(cleaned)
        chunk_ms = int(len(prep.pcm) / self.config.bytes_per_second * 1000)
        logger.info(
            "modern_audio_ingest room=%s peer=%s utterance_id=%s chunk_ms=%s raw_rms=%s normalized_rms=%s gain_applied=%.2f clipped=%s vad_decision=%s noise_floor=%.1f silence_ratio=%.3f queue_size=%s source_language=%s target_language=%s",
            self.room,
            self.peer_id,
            self.utterance_id,
            chunk_ms,
            prep.raw_rms,
            prep.normalized_rms,
            prep.gain_applied,
            prep.clipped,
            prep.vad_decision,
            prep.noise_floor,
            prep.silence_ratio,
            1 if self.final_task and not self.final_task.done() else 0,
            self.config.source_language,
            self.config.target_language,
        )

        if prep.vad_decision:
            if not self.in_speech:
                self.utterance_id += 1
                self.generation += 1
                self.utterance_started_at = now
                self.last_partial_at = 0.0
                self.last_partial_text = ""
            self.in_speech = True
            self.last_voice_at = now
            self.audio_buffer.extend(prep.pcm)
            self._trim_buffer()
            self._maybe_start_partial(now)
            if self._buffer_seconds() >= REALTIME_UTTERANCE_MAX_SECONDS:
                self._start_final(now, reason="max_duration")
            return

        self._update_noise_floor(prep.raw_rms)
        if self.in_speech and now - self.last_voice_at >= REALTIME_FINAL_SILENCE_SECONDS:
            self._start_final(now, reason="silence")

    async def close(self) -> None:
        self.closed = True
        self.audio_buffer.clear()
        tasks = [task for task in (self.partial_task, self.final_task) if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _prepare_realtime_audio(self, pcm: bytes) -> RealtimeAudioPrep:
        raw_rms = pcm_rms(pcm)
        self._update_noise_floor(raw_rms)
        vad_threshold = self._adaptive_vad_threshold()
        gain = 1.0
        if 0 < raw_rms < REALTIME_GAIN_TARGET_RMS:
            gain = min(REALTIME_MAX_GAIN, REALTIME_GAIN_TARGET_RMS / max(raw_rms, 1))
        normalized = apply_gain_to_pcm(pcm, gain)
        normalized_rms = pcm_rms(normalized)
        silence_ratio = pcm_silence_ratio(
            normalized,
            self.config.sample_rate,
            self.config.channels,
            threshold=max(STT_HARD_SILENCE_RMS, min(vad_threshold, 380)),
        )
        clipped = pcm_has_clipping(normalized)
        active_ratio = 1.0 - silence_ratio
        vad_decision = (
            active_ratio >= max(0.035, STT_MIN_ACTIVE_RATIO * 0.55)
            and normalized_rms >= max(70, self.noise_floor * 1.35)
        ) or normalized_rms >= vad_threshold
        return RealtimeAudioPrep(
            pcm=normalized,
            raw_rms=raw_rms,
            normalized_rms=normalized_rms,
            gain_applied=round(gain, 2),
            clipped=clipped,
            silence_ratio=silence_ratio,
            vad_decision=vad_decision,
            noise_floor=round(self.noise_floor, 1),
            vad_threshold=vad_threshold,
        )

    def _update_noise_floor(self, rms: int) -> None:
        if rms <= 0:
            return
        if rms < max(180, self.noise_floor * 1.7):
            self.noise_floor = max(30.0, min(360.0, (self.noise_floor * 0.94) + (rms * 0.06)))

    def _adaptive_vad_threshold(self) -> int:
        return int(max(90, min(420, self.noise_floor * 2.1)))

    def _buffer_seconds(self) -> float:
        if self.config.bytes_per_second <= 0:
            return 0.0
        return len(self.audio_buffer) / self.config.bytes_per_second

    def _maybe_start_partial(self, now: float) -> None:
        utterance_seconds = self._buffer_seconds()
        if utterance_seconds < REALTIME_PARTIAL_MIN_SECONDS:
            return
        if now - self.last_partial_at < REALTIME_PARTIAL_INTERVAL_SECONDS:
            return
        if self.partial_task is not None and not self.partial_task.done():
            return
        pcm = bytes(self.audio_buffer)
        self.last_partial_at = now
        self.partial_task = asyncio.create_task(
            self._process_utterance(
                pcm=pcm,
                utterance_id=self.utterance_id,
                generation=self.generation,
                is_final=False,
                queued_at=now,
                reason="partial",
            )
        )

    def _start_final(self, now: float, reason: str) -> None:
        if not self.audio_buffer:
            self.in_speech = False
            return
        utterance_seconds = self._buffer_seconds()
        if utterance_seconds < REALTIME_UTTERANCE_MIN_SECONDS:
            logger.info(
                "modern_utterance_short_skip room=%s peer=%s utterance_id=%s partial=%s final=%s utterance_ms=%s rms=%s silence_ratio=%.3f stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s reason=%s",
                self.room,
                self.peer_id,
                self.utterance_id,
                False,
                True,
                int(utterance_seconds * 1000),
                pcm_rms(bytes(self.audio_buffer)),
                pcm_silence_ratio(bytes(self.audio_buffer), self.config.sample_rate, self.config.channels),
                0,
                0,
                0,
                True,
                self.config.source_language,
                self.config.target_language,
                reason,
            )
            self.audio_buffer.clear()
            self.in_speech = False
            self.last_partial_text = ""
            return
        if self.final_task is not None and not self.final_task.done():
            logger.info("modern_final_busy_drop room=%s peer=%s utterance_id=%s utterance_ms=%s", self.room, self.peer_id, self.utterance_id, int(utterance_seconds * 1000))
            self.audio_buffer.clear()
            self.in_speech = False
            return
        pcm = bytes(self.audio_buffer)
        utterance_id = self.utterance_id
        generation = self.generation
        self.audio_buffer.clear()
        self.in_speech = False
        self.last_partial_text = ""
        self.final_task = asyncio.create_task(
            self._process_utterance(
                pcm=pcm,
                utterance_id=utterance_id,
                generation=generation,
                is_final=True,
                queued_at=now,
                reason=reason,
            )
        )

    async def _process_utterance(
        self,
        *,
        pcm: bytes,
        utterance_id: int,
        generation: int,
        is_final: bool,
        queued_at: float,
        reason: str,
    ) -> None:
        started = time.perf_counter()
        stats = pcm_audio_stats(pcm, self.config.sample_rate, self.config.channels)
        utterance_ms = int(stats.duration_seconds * 1000)
        try:
            if is_final and stats.duration_seconds < REALTIME_UTTERANCE_MIN_SECONDS:
                return
            if not is_final and stats.duration_seconds < REALTIME_PARTIAL_MIN_SECONDS:
                return
            timeout = REALTIME_STT_TIMEOUT_SECONDS if is_final else min(0.9, REALTIME_STT_TIMEOUT_SECONDS)
            try:
                text = await asyncio.wait_for(
                    transcribe_pcm_bytes(pcm, self.config, self.last_final_text),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                stt_ms = int((time.perf_counter() - started) * 1000)
                self._log_utterance_event("modern_stt_timeout_discard", utterance_id, is_final, stats, stt_ms, 0, stt_ms, True, reason)
                return
            stt_ms = int((time.perf_counter() - started) * 1000)
            if generation != self.generation and not is_final:
                return
            if not self._is_reliable_transcript(text, is_final=is_final):
                self._log_utterance_event("modern_stt_unreliable_discard", utterance_id, is_final, stats, stt_ms, 0, stt_ms, True, reason, text)
                return
            key = transcript_key(text)
            if key == self.last_caption_text and time.monotonic() - self.last_caption_at < STT_REPEAT_WINDOW_SECONDS:
                self._log_utterance_event("modern_stt_repeat_filtered", utterance_id, is_final, stats, stt_ms, 0, stt_ms, True, reason, text)
                return
            if not is_final:
                if key == self.last_partial_text:
                    return
                self.last_partial_text = key
                self.last_caption_text = key
                self.last_caption_at = time.monotonic()
                total_ms = int((time.perf_counter() - started) * 1000)
                self._log_utterance_event("modern_partial_caption", utterance_id, False, stats, stt_ms, 0, total_ms, False, reason, text)
                await manager.send(
                    "translation",
                    self.room,
                    self.peer_id,
                    {
                        "type": "caption",
                        "text": text,
                        "translation": "",
                        "source_language": self.config.source_language,
                        "target_language": self.config.target_language,
                        "is_final": False,
                        "utterance_id": utterance_id,
                        "timing_ms": {"stt": stt_ms, "translation": 0, "tts": 0, "total": total_ms},
                    },
                )
                return

            if is_probably_incomplete_sentence(text):
                self._log_utterance_event("modern_incomplete_final_discard", utterance_id, True, stats, stt_ms, 0, stt_ms, True, reason, text)
                return
            translate_started = time.perf_counter()
            translated = await translate_text_fast(text, self.config.source_language, self.config.target_language)
            translate_ms = int((time.perf_counter() - translate_started) * 1000)
            total_ms = int((time.perf_counter() - started) * 1000)
            if not translated or total_ms > 1200:
                self._log_utterance_event("modern_final_discard", utterance_id, True, stats, stt_ms, translate_ms, total_ms, True, reason, text)
                return
            self.last_caption_text = key
            self.last_caption_at = time.monotonic()
            self.last_final_text = clean_transcript(f"{self.last_final_text} {text}")[-500:]
            self._log_utterance_event("modern_caption", utterance_id, True, stats, stt_ms, translate_ms, total_ms, False, reason, text)
            await manager.send(
                "translation",
                self.room,
                self.peer_id,
                {
                    "type": "caption",
                    "text": text,
                    "translation": translated,
                    "source_language": self.config.source_language,
                    "target_language": self.config.target_language,
                    "is_final": True,
                    "utterance_id": utterance_id,
                    "timing_ms": {"stt": stt_ms, "translation": translate_ms, "tts": 0, "total": total_ms},
                },
            )
        except Exception as exc:
            logger.exception("modern_utterance_error room=%s peer=%s utterance_id=%s", self.room, self.peer_id, utterance_id)
            if not self.closed:
                await manager.send("translation", self.room, self.peer_id, {"type": "error", "message": str(exc)})

    def _is_reliable_transcript(self, text: str, *, is_final: bool) -> bool:
        cleaned = clean_transcript(text)
        if is_low_value_transcript(cleaned):
            return False
        key = transcript_key(cleaned)
        words = key.split()
        if not words:
            return False
        if cleaned.lower().strip(" .!?") in {"pures", "neyiz", "niye kafayi", "niye kafay?", "ne kadar bicim", "ne kadar bi?im"}:
            return False
        if is_final and len(words) == 1 and len(key) < 6:
            return False
        if is_final and len(words) <= 2 and len(key) < 10:
            return False
        if has_repeated_short_words(cleaned):
            return False
        return True

    def _log_utterance_event(
        self,
        event: str,
        utterance_id: int,
        is_final: bool,
        stats: AudioStats,
        stt_ms: int,
        translate_ms: int,
        total_ms: int,
        discarded_old_chunk: bool,
        reason: str,
        stt_text: str = "",
    ) -> None:
        logger.info(
            "%s room=%s peer=%s utterance_id=%s partial=%s final=%s utterance_ms=%s chunk_ms=%s rms=%s silence_ratio=%.3f queue_size=%s stt_ms=%s translate_ms=%s total_ms=%s discarded_old_chunk=%s source_language=%s target_language=%s reason=%s stt_text=%r",
            event,
            self.room,
            self.peer_id,
            utterance_id,
            not is_final,
            is_final,
            int(stats.duration_seconds * 1000),
            int(stats.duration_seconds * 1000),
            stats.rms,
            stats.silence_ratio,
            1 if self.final_task and not self.final_task.done() else 0,
            stt_ms,
            translate_ms,
            total_ms,
            discarded_old_chunk,
            self.config.source_language,
            self.config.target_language,
            reason,
            stt_text[:80],
        )

    def _trim_buffer(self) -> None:
        max_bytes = int(self.config.bytes_per_second * REALTIME_UTTERANCE_MAX_SECONDS)
        if len(self.audio_buffer) > max_bytes:
            dropped_bytes = len(self.audio_buffer) - max_bytes
            del self.audio_buffer[: len(self.audio_buffer) - max_bytes]
            logger.info(
                "modern_buffer_trim room=%s peer=%s utterance_id=%s dropped_ms=%s buffer_ms=%s source_language=%s target_language=%s",
                self.room,
                self.peer_id,
                self.utterance_id,
                int(dropped_bytes / self.config.bytes_per_second * 1000),
                int(len(self.audio_buffer) / self.config.bytes_per_second * 1000),
                self.config.source_language,
                self.config.target_language,
            )

def normalize_source_lang(lang: str) -> str:
    if not lang:
        return "TR"
    lang = lang.upper()
    if lang == "AUTO":
        return "AUTO"
    if lang == "EN-US":
        return "EN"
    return lang


def normalize_target_lang(lang: str) -> str:
    if not lang:
        return "EN-US"
    lang = lang.upper()
    if lang == "EN":
        return "EN-US"
    return lang


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
        "NL": "nl",
        "AR": "ar",
        "ES": "es",
        "ZH": "zh-CN",
        "KA": "ka",
    }
    return mapping.get(normalize_target_lang(lang), "en")


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
    return has_repeated_short_words(cleaned)


def is_probably_incomplete_sentence(text: str) -> bool:
    cleaned = clean_transcript(text)
    if not cleaned:
        return True
    key = transcript_key(cleaned)
    words = key.split()
    if len(words) <= 1:
        return True
    if cleaned.endswith(("...", "…")):
        return True
    dangling = {
        "ve",
        "ama",
        "fakat",
        "çünkü",
        "cunku",
        "için",
        "icin",
        "ile",
        "to",
        "and",
        "but",
        "because",
        "for",
        "of",
    }
    return words[-1] in dangling


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


def pcm_has_clipping(pcm: bytes) -> bool:
    if len(pcm) < SAMPLE_WIDTH_BYTES:
        return False
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    if not samples:
        return False
    clipped = sum(1 for sample in samples if abs(sample) >= 32000)
    return clipped / len(samples) > 0.01


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
    prepared_rms = pcm_rms(prepared_pcm)
    if prepared_rms >= STT_TARGET_RMS:
        if trimmed_ms > 0:
            return write_wav(prepared_pcm, sample_rate, channels), original_rms, 1.0, trimmed_ms
        return path, original_rms, 1.0, 0

    gain = min(STT_MAX_GAIN, STT_TARGET_RMS / max(prepared_rms, 1))
    if gain <= 1.05:
        if trimmed_ms > 0:
            return write_wav(prepared_pcm, sample_rate, channels), original_rms, 1.0, trimmed_ms
        return path, original_rms, 1.0, 0

    amplified_pcm = apply_gain_to_pcm(prepared_pcm, gain)
    normalized_path = write_wav(amplified_pcm, sample_rate, channels)
    return normalized_path, original_rms, round(gain, 2), trimmed_ms


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
    prepared_rms = pcm_rms(prepared_pcm)
    if prepared_rms >= STT_TARGET_RMS:
        return prepared_pcm, original_rms, 1.0, trimmed_ms

    gain = min(STT_MAX_GAIN, STT_TARGET_RMS / max(prepared_rms, 1))
    if gain <= 1.05:
        return prepared_pcm, original_rms, 1.0, trimmed_ms

    return apply_gain_to_pcm(prepared_pcm, gain), original_rms, round(gain, 2), trimmed_ms


async def transcribe_pcm(pcm: bytes, config: TranslationConfig) -> str:
    return await transcribe_pcm_bytes(pcm, config)


async def transcribe_pcm_bytes(pcm: bytes, config: TranslationConfig, previous_text: str = "") -> str:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY eksik!")
        return ""

    source_lang = normalize_source_lang(config.source_language)
    groq_lang = whisper_lang(source_lang)
    original_stats = pcm_audio_stats(pcm, config.sample_rate, config.channels)
    if config.sample_rate != SAMPLE_RATE or config.channels != CHANNELS:
        logger.info(
            "groq_stt_bad_format lang=%s sample_rate=%s channels=%s chunk_ms=%s rms=%s silence_ratio=%.3f",
            groq_lang,
            config.sample_rate,
            config.channels,
            int(original_stats.duration_seconds * 1000),
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""
    if should_skip_audio_for_stt(original_stats):
        logger.info(
            "groq_stt_vad_skip lang=%s chunk_ms=%s rms=%s silence_ratio=%.3f",
            groq_lang,
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
    prepared_stats = pcm_audio_stats(prepared_pcm, config.sample_rate, config.channels)
    try:
        async with httpx.AsyncClient(timeout=REALTIME_STT_TIMEOUT_SECONDS) as client:
            files = {
                "file": (
                    "audio.wav",
                    wav_bytes_from_pcm(prepared_pcm, config.sample_rate, config.channels),
                    "audio/wav",
                )
            }
            data = {
                "model": GROQ_STT_MODEL,
                "response_format": "json",
                "temperature": "0",
                "prompt": transcription_prompt(source_lang, previous_text),
            }
            if groq_lang != "auto":
                data["language"] = groq_lang

            response = await client.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files=files,
                data=data,
            )
            response.raise_for_status()
            text = response.json().get("text", "").strip()
            if is_low_value_transcript(text):
                logger.info(
                    "groq_stt_low_value lang=%s chunk_ms=%s rms=%s silence_ratio=%.3f stt_text=%r",
                    groq_lang,
                    int(original_stats.duration_seconds * 1000),
                    original_stats.rms,
                    original_stats.silence_ratio,
                    text[:80],
                )
                return ""
            logger.info(
                "groq_stt_ok lang=%s chunk_ms=%s rms=%s silence_ratio=%.3f prepared_chunk_ms=%s prepared_rms=%s gain=%.2f trimmed_ms=%s stt_text=%r",
                groq_lang,
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
    except httpx.TimeoutException:
        logger.warning("groq_stt_timeout lang=%s", groq_lang)
        return ""
    except Exception as e:
        logger.error("groq_stt_exception %s", e)
        return ""


async def transcribe_wav(path: str, source_lang: str, previous_text: str = "") -> str:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY eksik!")
        return ""

    groq_lang = GROQ_LANGUAGE_MAP.get(source_lang.lower(), source_lang.lower())
    original_stats = wav_audio_stats(path)
    if original_stats.sample_rate != SAMPLE_RATE or original_stats.channels != CHANNELS:
        logger.info(
            "groq_stt_bad_format lang=%s sample_rate=%s channels=%s duration=%.3f rms=%s silence_ratio=%.3f",
            groq_lang,
            original_stats.sample_rate,
            original_stats.channels,
            original_stats.duration_seconds,
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""
    if should_skip_audio_for_stt(original_stats):
        logger.info(
            "groq_stt_vad_skip lang=%s duration=%.3f rms=%s silence_ratio=%.3f",
            groq_lang,
            original_stats.duration_seconds,
            original_stats.rms,
            original_stats.silence_ratio,
        )
        return ""

    stt_path = path
    try:
        stt_path, original_rms, gain, trimmed_ms = normalize_wav_for_stt(path)
        prepared_stats = wav_audio_stats(stt_path)
        async with httpx.AsyncClient(timeout=10.0) as client:
            with open(stt_path, "rb") as f:
                audio_data = f.read()

            if len(audio_data) < 8000:
                return ""

            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            data = {
                "model": GROQ_STT_MODEL,
                "response_format": "json",
                "temperature": "0",
            }
            if groq_lang != "auto":
                data["language"] = groq_lang
            data["prompt"] = transcription_prompt(source_lang, previous_text)

            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

            response = await client.post(
                GROQ_STT_URL,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()

            text = response.json().get("text", "").strip()
            if is_low_value_transcript(text):
                logger.info(
                    "groq_stt_low_value lang=%s duration=%.3f rms=%s silence_ratio=%.3f stt_text=%r",
                    groq_lang,
                    original_stats.duration_seconds,
                    original_stats.rms,
                    original_stats.silence_ratio,
                    text[:80],
                )
                return ""

            logger.info(
                "groq_stt_ok lang=%s duration=%.3f rms=%s silence_ratio=%.3f prepared_duration=%.3f prepared_rms=%s gain=%.2f trimmed_ms=%s stt_text=%r",
                groq_lang,
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

    except httpx.TimeoutException:
        logger.warning("groq_stt_timeout lang=%s", groq_lang)
        return ""
    except Exception as e:
        logger.error("groq_stt_exception %s", e)
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
    except (WebSocketDisconnect, RuntimeError):
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
    logger.info("legacy_translate_disabled client=%s", websocket.client.host if websocket.client else "unknown")
    await safe_send(
        websocket,
        {
            "error": "legacy_translate_disabled",
            "message": "Use /ws/translate/{room}/{peer_id} with binary pcm16 audio.",
        },
    )
    try:
        await websocket.close(code=1008)
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
