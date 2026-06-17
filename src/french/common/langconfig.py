#!/usr/bin/env python3
"""French :class:`LangConfig`, shared by the French ytconverter and
applepodcastconverter scripts (voices, language codes, the OpenAI vocab params)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable so `from common.ytcommon import ...` works when imported
# from a sibling per-language script run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.ytcommon import LangConfig  # noqa: E402

FRENCH = LangConfig(
    native_voice="fr-FR-Remy:DragonHDLatestNeural",
    en_voice="en-US-Ava:DragonHDLatestNeural",
    tts_rate="0.9",
    xml_lang="fr-FR",
    language_code="fr-FR",
    chirp_location="us",
    chirp_model="chirp_3",
    mai_locale="fr",
    sentence_end_chars=".!?…",
    sub_sentence_break_chars=",;:",
    word_joiner=" ",
    translate_source="fr",
    vocab_extra_field="",
    vocab_extra_explain="",
    album="LearnLangs French",
)
