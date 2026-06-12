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
and src/common/ytcommon.py (library helpers); the Mandarin-specific pieces
(OpenCC s2tw conversion, the OpenAI vocab params, and the voices) live in
src/mandarin/common/langconfig.py, shared with the Apple Podcast converter.

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
from common.ytpipeline import run_pipeline  # noqa: E402
from mandarin.common.langconfig import MANDARIN, mandarin_word_postprocess  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    run_pipeline(MANDARIN, SCRIPT_DIR, word_postprocess=mandarin_word_postprocess, description=__doc__)


if __name__ == "__main__":
    main()
