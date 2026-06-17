#!/usr/bin/env python3
"""Mandarin :class:`LangConfig` plus the OpenCC s2tw word post-processor, shared
by the Mandarin ytconverter and applepodcastconverter scripts (voices, language
codes, the OpenAI vocab params, Simplified → Traditional-Taiwan conversion)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable so `from common.ytcommon import ...` works when imported
# from a sibling per-language script run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.ytcommon import LangConfig, WordRec  # noqa: E402

MANDARIN = LangConfig(
    native_voice="zh-TW-YunJheNeural",
    en_voice="en-US-Ava:DragonHDLatestNeural",
    tts_rate="0.9",
    xml_lang="zh-TW",
    language_code="cmn-Hans-CN",
    chirp_location="us",
    chirp_model="chirp_3",
    mai_locale="zh",
    sentence_end_chars="。！？!?.",
    sub_sentence_break_chars="，,、；;：:",
    word_joiner="",
    translate_source="zh-TW",
    vocab_extra_field="pinyin",
    vocab_extra_explain='"pinyin" is the Hanyu Pinyin romanization (with tone marks) of "text".',
    album="LearnLangs Mandarin",
)


# ─── OpenCC s2tw (Simplified → Traditional Taiwan) ────────────────────────────

_S2T_CONVERTER = None


def s2t(text: str) -> str:
    global _S2T_CONVERTER
    if _S2T_CONVERTER is None:
        from opencc import OpenCC
        _S2T_CONVERTER = OpenCC("s2tw")
    return _S2T_CONVERTER.convert(text)


def mandarin_word_postprocess(words: list[WordRec]) -> list[WordRec]:
    """Convert each STT word from Simplified to Traditional-Taiwan before sentence
    segmentation (passed as run_pipeline's `word_postprocess`)."""
    return [
        WordRec(word=s2t(w.word), start_ms=w.start_ms, end_ms=w.end_ms, speaker=w.speaker)
        for w in words
    ]
