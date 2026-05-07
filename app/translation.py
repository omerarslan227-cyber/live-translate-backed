from __future__ import annotations

import asyncio
from dataclasses import dataclass

import deepl


@dataclass(frozen=True)
class TranslationResult:
    text: str
    detected_source_lang: str | None = None


class Translator:
    def __init__(self, api_key: str | None) -> None:
        self._client = deepl.Translator(api_key) if api_key else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def translate_text(
        self,
        text: str,
        *,
        source_lang: str | None,
        target_lang: str | None,
        formality: str | None = None,
    ) -> TranslationResult:
        if not self._client:
            raise RuntimeError("DEEPL_API_KEY is not configured")
        cleaned = text.strip()
        if not cleaned:
            return TranslationResult(text="")

        def run() -> deepl.TextResult:
            kwargs: dict[str, str] = {}
            if source_lang:
                kwargs["source_lang"] = normalize_source_lang(source_lang)
            if formality:
                kwargs["formality"] = formality
            return self._client.translate_text(
                cleaned,
                target_lang=normalize_target_lang(target_lang),
                split_sentences="1",
                preserve_formatting=True,
                **kwargs,
            )

        result = await asyncio.to_thread(run)
        return TranslationResult(
            text=str(result.text).strip(),
            detected_source_lang=getattr(result, "detected_source_lang", None),
        )


def normalize_source_lang(value: str | None) -> str | None:
    if not value:
        return None
    lang = value.upper().replace("_", "-")
    if lang in {"EN-US", "EN-GB"}:
        return "EN"
    return lang


def normalize_target_lang(value: str | None) -> str:
    if not value:
        return "EN-US"
    lang = value.upper().replace("_", "-")
    if lang == "EN":
        return "EN-US"
    return lang
