#!/usr/bin/env python3
"""Mandarin YouTube → Vocab Listening-Practice Generator.

End-to-end pipeline:
  1. Prompt for a YouTube URL.
  2. Download audio (yt-dlp -x mp3) into ./inputs/.
  3. Transcribe with Google Chirp 3 (default; pass --stt mai for Azure
     MAI-Transcribe-1.5), with word-level timestamps. Convert Hans → Hant-TW
     via OpenCC. Cache JSON.
  4. Extract the top-N vocab words/phrases (N = --vocab-number, default 40) with
     the OpenAI API (each item carries pinyin, a contextual SSML explanation, a
     plain-text explanation and a short English gloss). Cache vocab.json, then
     write vocab.tsv.
  5. Translate every sentence (Cloud Translate v3) for the playback pairs.
  6. Slice the source audio into ~10-min chunks snapped to sentence ends.
  7. For each sentence containing ≥1 vocab item, render an Azure TTS explanation
     clip = OpenAI's per-vocab SSML explanation(s) + original sentence slice +
     synthetic sentence TTS + English sentence translation, with 500 ms breaks.
  8. Assemble each chunk:
       original_chunk + 1s + part1 + expl1 + 1s + part2 + expl2 + 1s + … + tail
     and concatenate all chunks (2s between chunks) into outputs/<stem>.mp3.

The language-agnostic pipeline lives in src/common/ytpipeline.py (orchestration)
and src/common/ytcommon.py (library helpers); this script only carries the
Mandarin-specific pieces (OpenCC s2tw conversion, the OpenAI vocab params, and
the voices).

I/O folders (created at invocation cwd):
  inputs/                 - downloaded MP3
  intermediates/<stem>/   - transcript.json, vocab.json, vocab.tsv, tts/, chunks/
  outputs/                - final concatenated study MP3

Credentials (next to this script):
  key.json       - {"azSpeechKey": "<Azure Cognitive Services key>",
                    "azSpeechRegion": "<e.g. eastus>",
                    "azSttEndpoint": "<Foundry host for MAI-Transcribe; only for --stt mai,
                                       e.g. https://<resource>.cognitiveservices.azure.com;
                                       optional, else derived from azSpeechRegion>",
                    "gcsBucket": "<GCS bucket for STT staging (default Chirp path)>",
                    "cApi": "<OpenAI API key for vocab extraction>",
                    "cApiBaseUrl": "<Azure OpenAI base URL for vocab extraction>"}
  jumeau-gc.json - Google Cloud service account JSON (used for Chirp STT + Translate v3)

Dependencies:
  pip install -r requirements.txt
  brew install ffmpeg   # pydub MP3 decode; also used by yt-dlp
  python3 ytconverter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable so `from common.ytpipeline import ...` works when the
# script is run directly from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.ytcommon import LangConfig, WordRec  # noqa: E402
from common.ytpipeline import run_pipeline  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

MANDARIN = LangConfig(
    native_voice="zh-TW-YunJheNeural",
    en_voice="en-US-AvaNeural",
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


def _s2t_words(words: list[WordRec]) -> list[WordRec]:
    return [
        WordRec(word=s2t(w.word), start_ms=w.start_ms, end_ms=w.end_ms, speaker=w.speaker)
        for w in words
    ]


def main():
    run_pipeline(MANDARIN, SCRIPT_DIR, word_postprocess=_s2t_words, description=__doc__)


if __name__ == "__main__":
    main()
