#!/usr/bin/env python3
"""Mandarin YouTube → Vocab Listening-Practice Generator.

End-to-end pipeline:
  1. Prompt for a YouTube URL.
  2. Download audio (yt-dlp -x mp3) into ./inputs/.
  3. Transcribe with Google Chirp v2 (cmn-Hans-CN), with word-level timestamps.
     Convert Hans → Hant-TW via OpenCC. Cache JSON.
  4. Tokenize with jieba and count occurrences.
  5. Interactively pick X most-occurring + X least-occurring NEW words
     (single keypress: 1 = NEW, 2 = mark KNOWN/append to hsk1to4_zh-TW.txt).
  6. Slice the source audio into ~10-min chunks snapped to sentence ends.
  7. For each sentence containing ≥1 new word, render an Azure TTS explanation
     clip = word(s) + English meaning(s) + original sentence slice + synthetic
     sentence TTS + English sentence translation, with 500 ms breaks.
  8. Assemble each chunk:
       original_chunk + 1s + part1 + expl1 + 1s + part2 + expl2 + 1s + … + tail
     and concatenate all chunks (2s between chunks) into outputs/<stem>.mp3.

I/O folders (created at invocation cwd):
  inputs/                 - downloaded MP3
  intermediates/<stem>/   - transcript.json, vocab.tsv, tts/, chunks/
  outputs/                - final concatenated study MP3

Credentials (next to this script):
  key.json       - {"azSpeechKey": "<Azure Cognitive Services key>",
                    "azSpeechRegion": "<e.g. eastus>",
                    "gcsBucket": "<GCS bucket for STT staging>"}
  jumeau-gc.json - Google Cloud service account JSON (used for STT, Translate v3)

Dependencies:
  pip install -r requirements.txt
  brew install ffmpeg   # pydub MP3 decode; also used by yt-dlp
  python3 ytconverter.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import requests
from pydub import AudioSegment

LANGUAGE_CODE = "cmn-Hans-CN"
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
CHIRP_LOCATION = "us"

TW_VOICE = "zh-TW-YunJheNeural"
EN_VOICE = "en-US-AvaNeural"
TTS_RATE = "0.9"
INTRA_GROUP_BREAK_MS = 500
INTER_PART_BREAK_MS = 1000
INTER_CHUNK_BREAK_MS = 2000
CHUNK_ANNOUNCEMENT_PAD_MS = 600
NO_VOCAB_BREAK_MS = 600
CHUNK_TARGET_MS = 5 * 60 * 1000
SILENCE_LEN_MS = 500
SILENCE_THRESH_DB = -16  # dB below the audio's average dBFS
SILENCE_SEARCH_WINDOW_MS = 60 * 1000

SENTENCE_BREAK_CHARS = "。！？!?.，,、；;：:"
MIN_SENTENCE_MS = 2000
MAX_SENTENCE_MS = 7000
PROPER_NOUN_POS = {"nr", "ns", "nt", "nz", "nrfg", "nrt"}
SKIP_POS = {"u", "uj", "ul", "ud", "uv", "uz", "ug", "p", "c", "y", "e", "o", "x", "w", "m"}
PUNCT_OR_DIGIT_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)
HAN_ONLY_RE = re.compile(r"^[㐀-䶿一-鿿豈-﫿]+$")

SCRIPT_DIR = Path(__file__).resolve().parent
HSK_PATH = SCRIPT_DIR / "hsk1to4_zh-TW.txt"
HSK_HEADER = (
    "# HSK 1-4 vocabulary list — traditional Chinese characters\n"
    "# Words NOT in this list are flagged as candidates by ytconverter.py.\n"
    "# Add one traditional word per line; lines starting with '#' are ignored.\n\n"
)


# ─── Single-keypress input ────────────────────────────────────────────────────

def getch() -> str:
    """Read a single character from stdin with no Enter required (POSIX)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


# ─── Bootstrap / config ───────────────────────────────────────────────────────

def load_keys(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_hsk_file(path: Path) -> None:
    if not path.exists():
        path.write_text(HSK_HEADER, encoding="utf-8")
        print(f"  created empty HSK list: {path}")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def sanitize_stem(stem: str) -> str:
    """Make a filesystem-safe stem (also used as cache directory name)."""
    cleaned = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip()
    return cleaned or f"episode-{int(time.time())}"


# ─── yt-dlp download ──────────────────────────────────────────────────────────

def download_youtube(url: str, inputs_dir: Path) -> Path:
    """Run yt-dlp and return the path to the downloaded MP3."""
    ensure_dirs(inputs_dir)
    output_template = str(inputs_dir / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_template,
        "--print", "after_move:filepath",
        "--no-simulate",
        url,
    ]
    print(f"  → yt-dlp: {url}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not paths:
        raise RuntimeError(f"yt-dlp did not report an output filepath. stderr:\n{result.stderr}")
    return Path(paths[-1]).resolve()


# ─── Audio convert + GCS upload (reused from prior version) ───────────────────

def split_mp3_to_flac_chunks(
    mp3_path: Path, target_ms: int, out_dir: Path
) -> list[tuple[Path, int, int]]:
    """Decode MP3 once, split at the closest ≥SILENCE_LEN_MS silence after each
    target_ms mark. Returns [(path, start_ms, end_ms)]."""
    from pydub.silence import detect_silence

    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)
    total_ms = len(audio)
    thresh = audio.dBFS + SILENCE_THRESH_DB

    out: list[tuple[Path, int, int]] = []
    start = 0
    i = 0
    while start < total_ms:
        target_end = start + target_ms
        if target_end >= total_ms:
            end = total_ms
        else:
            search_end = min(target_end + SILENCE_SEARCH_WINDOW_MS, total_ms)
            silences = detect_silence(
                audio[target_end:search_end],
                min_silence_len=SILENCE_LEN_MS,
                silence_thresh=thresh,
            )
            if silences:
                sil_start, sil_end = silences[0]
                end = target_end + (sil_start + sil_end) // 2
            else:
                end = target_end
        chunk_path = out_dir / f"chunk_{i:03d}.flac"
        audio[start:end].export(str(chunk_path), format="flac")
        out.append((chunk_path, start, end))
        start = end
        i += 1
    return out


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str):
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}", blob


# ─── Chirp transcription (with word-level timestamps) ─────────────────────────

@dataclass
class WordRec:
    word: str
    start_ms: int
    end_ms: int


def _duration_to_ms(d) -> int:
    if hasattr(d, "total_seconds"):
        return int(d.total_seconds() * 1000)
    return int(d.seconds * 1000 + d.nanos // 1_000_000)


def transcribe(files: list[tuple[str, int]], project_id: str, speaker_count: int) -> list[WordRec]:
    """Chirp v2 BatchRecognize across N files. files = [(gcs_uri, offset_ms), ...].
    Returns flat list of word records (Hans), timestamps already offset and sorted."""
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    client = SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{CHIRP_LOCATION}-speech.googleapis.com")
    )
    recognizer = f"projects/{project_id}/locations/{CHIRP_LOCATION}/recognizers/_"
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[LANGUAGE_CODE],
        model="chirp_3",
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=speaker_count,
                max_speaker_count=speaker_count,
            ),
        ),
    )
    words: list[WordRec] = []
    for i, (uri, offset) in enumerate(files, 1):
        request = cloud_speech.BatchRecognizeRequest(
            recognizer=recognizer,
            config=config,
            files=[cloud_speech.BatchRecognizeFileMetadata(uri=uri)],
            recognition_output_config=cloud_speech.RecognitionOutputConfig(
                inline_response_config=cloud_speech.InlineOutputConfig(),
            ),
        )
        print(f"  → [{i}/{len(files)}] STT v2 (chirp_3) BatchRecognize submitted (+{offset}ms); waiting...")
        operation = client.batch_recognize(request=request)
        response = operation.result(timeout=3600)

        for resp_uri, file_result in response.results.items():
            err = getattr(file_result, "error", None)
            if err and getattr(err, "code", 0):
                print(f"  ! file error for {resp_uri}: code={err.code} message={err.message}", file=sys.stderr)
                continue
            results = list(file_result.transcript.results) if file_result.transcript else []
            print(f"  · {resp_uri}: {len(results)} transcript result(s)")
            for ri, result in enumerate(results):
                if not result.alternatives:
                    print(f"    [{ri}] no alternatives", file=sys.stderr)
                    continue
                alt = result.alternatives[0]
                if ri == 0:
                    preview = (alt.transcript or "")[:80].replace("\n", " ")
                    print(f"    [{ri}] words={len(alt.words)} transcript[:80]={preview!r}")
                for wi in alt.words:
                    if not wi.word:
                        continue
                    words.append(WordRec(
                        word=wi.word,
                        start_ms=_duration_to_ms(wi.start_offset) + offset,
                        end_ms=_duration_to_ms(wi.end_offset) + offset,
                    ))
    words.sort(key=lambda w: w.start_ms)
    return words


# ─── OpenCC s2tw ──────────────────────────────────────────────────────────────

_S2T_CONVERTER = None


def s2t(text: str) -> str:
    global _S2T_CONVERTER
    if _S2T_CONVERTER is None:
        from opencc import OpenCC
        _S2T_CONVERTER = OpenCC("s2tw")
    return _S2T_CONVERTER.convert(text)


# ─── Sentence segmentation from word records ──────────────────────────────────

@dataclass
class Sentence:
    text: str
    start_ms: int
    end_ms: int
    words: list[WordRec] = field(default_factory=list)


def _sentence_from_words(buf: list[WordRec]) -> Sentence:
    return Sentence(
        text="".join(x.word for x in buf).strip(),
        start_ms=buf[0].start_ms,
        end_ms=buf[-1].end_ms,
        words=list(buf),
    )


def build_sentences(words: list[WordRec]) -> list[Sentence]:
    """Group word records into sentences at break punctuation, keeping each
    sentence between MIN_SENTENCE_MS and MAX_SENTENCE_MS. A break char is only
    taken once the buffer has reached MIN_SENTENCE_MS; once it exceeds
    MAX_SENTENCE_MS, the buffer is flushed at the next word regardless."""
    sentences: list[Sentence] = []
    buf: list[WordRec] = []
    for w in words:
        buf.append(w)
        duration = buf[-1].end_ms - buf[0].start_ms
        is_break = bool(w.word) and w.word[-1] in SENTENCE_BREAK_CHARS
        if (is_break and duration >= MIN_SENTENCE_MS) or duration >= MAX_SENTENCE_MS:
            sentences.append(_sentence_from_words(buf))
            buf = []
    if buf:
        sentences.append(_sentence_from_words(buf))
    return sentences


def sentences_to_jsonable(sentences: list[Sentence]) -> list[dict]:
    return [
        {
            "text": s.text,
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "words": [{"w": w.word, "s": w.start_ms, "e": w.end_ms} for w in s.words],
        }
        for s in sentences
    ]


def sentences_from_jsonable(rows: list[dict]) -> list[Sentence]:
    out = []
    for r in rows:
        out.append(Sentence(
            text=r["text"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            words=[WordRec(word=w["w"], start_ms=w["s"], end_ms=w["e"]) for w in r["words"]],
        ))
    return out


# ─── Tokenization ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[tuple[str, str]]:
    import jieba.posseg as pseg

    out: list[tuple[str, str]] = []
    for w, pos in pseg.cut(text, HMM=True):
        w = w.strip()
        if not w or PUNCT_OR_DIGIT_RE.match(w):
            continue
        if pos in SKIP_POS:
            continue
        if len(w) < 2 and pos not in PROPER_NOUN_POS:
            continue
        out.append((w, pos))
    return out


def load_hsk(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def append_to_hsk(path: Path, word: str) -> None:
    data = path.read_bytes() if path.exists() else b""
    needs_newline = data and not data.endswith(b"\n")
    with open(path, "ab") as f:
        if needs_newline:
            f.write(b"\n")
        f.write(word.encode("utf-8") + b"\n")


# ─── Interactive vocab picker ─────────────────────────────────────────────────

def pick_round(
    label: str,
    counts: Counter,
    hsk: set[str],
    target: int,
    least_first: bool,
) -> dict[str, int]:
    """Iterate candidates and prompt single-keypress 1=NEW, 2=KNOWN, q=stop.
    Returns dict {word: count} for accepted-as-new words."""
    sign = 1 if least_first else -1
    ordered = sorted(
        counts.items(),
        key=lambda x: (sign * x[1], -len(x[0]), x[0]),
    )
    picked: dict[str, int] = {}
    print(f"\n── {label} — pick up to {target}; 1=NEW, 2=KNOWN, q=stop round ──")
    for word, count in ordered:
        if word in hsk:
            continue
        if not HAN_ONLY_RE.match(word):
            continue
        print(f"  [{count:>3}× len={len(word)}] {word}  ", end="", flush=True)
        ch = getch()
        print(ch)
        if ch == "1":
            picked[word] = count
            print(f"    → NEW ({len(picked)}/{target})")
            if len(picked) >= target:
                break
        elif ch == "2":
            hsk.add(word)
            append_to_hsk(HSK_PATH, word)
            print("    → KNOWN (added to HSK)")
        elif ch.lower() == "q":
            print("    → stop")
            break
        else:
            print("    → skip")
    return picked


# ─── Pinyin + translation ─────────────────────────────────────────────────────

def transliterate(text: str, az_key: str, az_region: str | None = None) -> str:
    url = (
        "https://api.cognitive.microsofttranslator.com/transliterate"
        "?api-version=3.0&language=zh-Hant&fromScript=Hant&toScript=Latn"
    )
    headers = {"Ocp-Apim-Subscription-Key": az_key, "Content-Type": "application/json"}
    if az_region:
        headers["Ocp-Apim-Subscription-Region"] = az_region
    resp = requests.post(url, headers=headers, json=[{"Text": text}], timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result and "text" in result[0]:
        return result[0]["text"]
    return ""


def translate_batch(texts: list[str], project_id: str, batch_size: int = 100) -> dict[str, str]:
    """zh-TW → en via Cloud Translate v3 (service-account auth from GOOGLE_APPLICATION_CREDENTIALS)."""
    from google.cloud import translate_v3

    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{project_id}/locations/global"
    out: dict[str, str] = {}
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.translate_text(
            parent=parent,
            contents=chunk,
            mime_type="text/plain",
            source_language_code="zh-TW",
            target_language_code="en",
        )
        for src, t in zip(chunk, resp.translations):
            out[src] = t.translated_text
    return out


# ─── Audio chunking by sentence ───────────────────────────────────────────────

@dataclass
class Chunk:
    idx: int
    start_ms: int
    end_ms: int
    sentences: list[Sentence]


def chunk_sentences_by_boundaries(
    sentences: list[Sentence], boundaries: list[tuple[int, int]]
) -> list[Chunk]:
    """Group sentences into chunks using pre-computed audio boundaries
    (start_ms, end_ms) from silence-aware splitting."""
    chunks: list[Chunk] = []
    for idx, (start_ms, end_ms) in enumerate(boundaries, 1):
        chunk_sents = [s for s in sentences if start_ms <= s.start_ms < end_ms]
        if not chunk_sents:
            continue
        chunks.append(Chunk(
            idx=idx,
            start_ms=start_ms,
            end_ms=end_ms,
            sentences=chunk_sents,
        ))
    return chunks


def chunk_sentences(sentences: list[Sentence], target_ms: int = CHUNK_TARGET_MS) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Sentence] = []
    start_ms = sentences[0].start_ms if sentences else 0
    for s in sentences:
        current.append(s)
        if current[-1].end_ms - start_ms >= target_ms:
            chunks.append(Chunk(
                idx=len(chunks) + 1,
                start_ms=start_ms,
                end_ms=current[-1].end_ms,
                sentences=current,
            ))
            current = []
            start_ms = s.end_ms
    if current:
        chunks.append(Chunk(
            idx=len(chunks) + 1,
            start_ms=start_ms,
            end_ms=current[-1].end_ms,
            sentences=current,
        ))
    return chunks


# ─── Azure TTS (cached) ───────────────────────────────────────────────────────

def render_tts(ssml: str, cache_dir: Path, az_key: str, az_region: str) -> AudioSegment:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha1(ssml.encode("utf-8")).hexdigest()
    out_path = cache_dir / f"{sha}.mp3"
    if not out_path.exists():
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(subscription=az_key, region=az_region)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(out_path))
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "cancellation_details", None)
            err = details.error_details if details else "unknown"
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(f"Azure TTS failed: {result.reason} / {err}")
    return AudioSegment.from_mp3(str(out_path))


def _voice(name: str, text: str, lead_break_ms: int = 0, trail_break_ms: int = 0) -> str:
    lead = f'<break time="{lead_break_ms}ms"/>' if lead_break_ms else ""
    trail = f'<break time="{trail_break_ms}ms"/>' if trail_break_ms else ""
    return (
        f'<voice name="{name}">'
        f'<prosody rate="{TTS_RATE}">{lead}{html.escape(text)}{trail}</prosody>'
        f'</voice>'
    )


def _wrap_ssml(body: str) -> str:
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-TW">'
        + body
        + '</speak>'
    )


def ssml_words_and_meanings(pairs: list[tuple[str, str]]) -> str:
    """pairs = [(traditional_word, english_meaning), ...]"""
    parts: list[str] = []
    for i, (w, en) in enumerate(pairs):
        # break before each pair after the first — placed inside the leading voice
        lead = INTRA_GROUP_BREAK_MS if i else 0
        parts.append(_voice(TW_VOICE, w, lead_break_ms=lead, trail_break_ms=INTRA_GROUP_BREAK_MS))
        parts.append(_voice(EN_VOICE, en or "(no translation)"))
    return _wrap_ssml("".join(parts))


def ssml_sentence_pair(zh_text: str, en_text: str) -> str:
    body = (
        _voice(TW_VOICE, zh_text, trail_break_ms=INTRA_GROUP_BREAK_MS)
        + _voice(EN_VOICE, en_text or "(no translation)")
    )
    return _wrap_ssml(body)


def ssml_chunk_announcement(idx: int, total: int) -> str:
    return _wrap_ssml(_voice(EN_VOICE, f"Playback part {idx} of {total}."))


def ssml_part_announcement(part_num: int, total_parts: int) -> str:
    body = _voice(
        EN_VOICE,
        f"Part {part_num} of {total_parts}",
        lead_break_ms=600,
        trail_break_ms=600,
    )
    return _wrap_ssml(body)


# ─── Per-sentence explanation clip ────────────────────────────────────────────

def build_explanation_clip(
    sentence: Sentence,
    new_words_in_order: list[str],
    word_meanings: dict[str, str],
    sentence_translation: str,
    original_audio: AudioSegment,
    tts_cache: Path,
    az_key: str,
    az_region: str,
) -> AudioSegment:
    pairs = [(w, word_meanings.get(w, "")) for w in new_words_in_order]
    clip_a = render_tts(ssml_words_and_meanings(pairs), tts_cache, az_key, az_region)
    original_slice = original_audio[sentence.start_ms : sentence.end_ms]
    clip_b = render_tts(
        ssml_sentence_pair(sentence.text, sentence_translation),
        tts_cache, az_key, az_region,
    )
    gap = AudioSegment.silent(duration=INTRA_GROUP_BREAK_MS)
    return clip_a + gap + original_slice + gap + clip_b


def build_no_vocab_clip(
    sentence: Sentence,
    sentence_translation: str,
    original_audio: AudioSegment,
    tts_cache: Path,
    az_key: str,
    az_region: str,
) -> AudioSegment:
    """Per-sentence clip for sentences without any new vocab:
    original_slice + 600 + EN_TTS + 600 + ZH_TTS + 600 (trailing pause)."""
    original_slice = original_audio[sentence.start_ms : sentence.end_ms]
    en_clip = render_tts(
        _wrap_ssml(_voice(EN_VOICE, sentence_translation or "(no translation)")),
        tts_cache, az_key, az_region,
    )
    zh_clip = render_tts(
        _wrap_ssml(_voice(TW_VOICE, sentence.text)),
        tts_cache, az_key, az_region,
    )
    gap = AudioSegment.silent(duration=NO_VOCAB_BREAK_MS)
    return original_slice + gap + en_clip + gap + zh_clip + gap


# ─── Chunk assembly ───────────────────────────────────────────────────────────

def assemble_chunk(
    chunk: Chunk,
    total_chunks: int,
    original_audio: AudioSegment,
    explanations_by_sentence: dict[int, AudioSegment],
    sentence_translations: dict[str, str],
    sentence_index: dict[int, int],
    tts_cache: Path,
    az_key: str,
    az_region: str,
) -> AudioSegment:
    """Build: original_chunk + 1s + 600ms + announcement + 600ms +
    per-sentence playbacks (explanation clip for new-vocab sentences,
    no-vocab clip otherwise) — pauses after every sentence."""
    out = original_audio[chunk.start_ms : chunk.end_ms]
    out += AudioSegment.silent(duration=INTER_PART_BREAK_MS)
    announcement = render_tts(
        ssml_chunk_announcement(chunk.idx, total_chunks),
        tts_cache, az_key, az_region,
    )
    pad = AudioSegment.silent(duration=CHUNK_ANNOUNCEMENT_PAD_MS)
    out += pad + announcement + pad

    for s in chunk.sentences:
        idx = sentence_index[id(s)]
        if idx in explanations_by_sentence:
            out += explanations_by_sentence[idx]
            out += AudioSegment.silent(duration=INTER_PART_BREAK_MS)
        else:
            out += build_no_vocab_clip(
                sentence=s,
                sentence_translation=sentence_translations.get(s.text, ""),
                original_audio=original_audio,
                tts_cache=tts_cache,
                az_key=az_key,
                az_region=az_region,
            )
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help="YouTube URL (prompted if omitted)")
    parser.add_argument("--keys", default=str(SCRIPT_DIR / "key.json"))
    parser.add_argument("--gc", default=str(SCRIPT_DIR / "jumeau-gc.json"))
    parser.add_argument("--gcs-bucket", help="GCS bucket for STT staging (else read from key.json:gcsBucket)")
    parser.add_argument("--azure-region", help="Azure Speech region (else read from key.json:azSpeechRegion)")
    parser.add_argument("--chunk-minutes", type=float, default=5.0)
    args = parser.parse_args()

    ensure_hsk_file(HSK_PATH)

    cwd = Path.cwd()
    inputs_dir = cwd / "inputs"
    intermediates_root = cwd / "intermediates"
    outputs_dir = cwd / "outputs"
    ensure_dirs(inputs_dir, intermediates_root, outputs_dir)

    keys = load_keys(Path(args.keys))
    az_key = keys.get("azSpeechKey") or keys.get("azDictKey")
    if not az_key:
        print("key.json must contain 'azSpeechKey' (or legacy 'azDictKey').", file=sys.stderr)
        sys.exit(1)
    az_region = args.azure_region or keys.get("azSpeechRegion")
    if not az_region:
        az_region = input("Azure Speech region (e.g. eastus): ").strip()
    gcs_bucket = args.gcs_bucket or keys.get("gcsBucket")
    if not gcs_bucket:
        gcs_bucket = input("GCS bucket for STT staging: ").strip()

    gc_path = Path(args.gc).resolve()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gc_path)
    with open(gc_path, encoding="utf-8") as f:
        project_id = json.load(f)["project_id"]

    url = args.url or input("YouTube URL: ").strip()
    if not url:
        print("No URL provided.", file=sys.stderr)
        sys.exit(1)

    # ── 1. Download ─────────────────────────────────────────────────────────
    print("\n[1/7] download")
    mp3_path = download_youtube(url, inputs_dir)
    stem = sanitize_stem(mp3_path.stem)
    print(f"  → {mp3_path}")

    inter_dir = intermediates_root / stem
    tts_cache = inter_dir / "tts"
    chunks_cache = inter_dir / "chunks"
    transcript_json_path = inter_dir / "transcript.json"
    boundaries_json_path = inter_dir / "chunk_boundaries.json"
    vocab_tsv_path = inter_dir / "vocab.tsv"
    for stale in (tts_cache, chunks_cache):
        if stale.exists():
            shutil.rmtree(stale)
    ensure_dirs(inter_dir, tts_cache, chunks_cache)

    chunk_target_ms = int(args.chunk_minutes * 60 * 1000)

    # ── 2. Transcribe (cached) ──────────────────────────────────────────────
    print("\n[2/7] transcribe")
    cached_rows: list[dict] = []
    boundaries: list[tuple[int, int]] | None = None
    if transcript_json_path.exists() and transcript_json_path.stat().st_size > 0:
        try:
            cached_rows = json.loads(transcript_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached_rows = []
    if cached_rows:
        print(f"  cached: {transcript_json_path.name}")
        sentences = sentences_from_jsonable(cached_rows)
        if boundaries_json_path.exists():
            try:
                raw = json.loads(boundaries_json_path.read_text(encoding="utf-8"))
                boundaries = [(int(s), int(e)) for s, e in raw]
                print(f"  cached: {boundaries_json_path.name} ({len(boundaries)} boundaries)")
            except (json.JSONDecodeError, ValueError, TypeError):
                boundaries = None
    else:
        try:
            speaker_count = int(input("How many speakers in the audio? ").strip() or "2")
        except ValueError:
            speaker_count = 2
        if speaker_count < 1:
            speaker_count = 1
        print(f"  → diarization: {speaker_count} speaker(s)")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            print(f"  → MP3 → FLAC chunks ({TARGET_SAMPLE_RATE}Hz mono, ~{chunk_target_ms // 60000}min each, split at silence)")
            flac_chunks = split_mp3_to_flac_chunks(mp3_path, chunk_target_ms, td_path)
            print(f"  → {len(flac_chunks)} chunk(s)")
            boundaries = [(start, end) for _, start, end in flac_chunks]
            timestamp = int(time.time())
            uploaded: list[tuple[str, int, object]] = []  # (uri, offset_ms, blob)
            for chunk_path, offset_ms, _end_ms in flac_chunks:
                blob_name = f"stt-staging/{stem}-{timestamp}-{chunk_path.stem}.flac"
                print(f"  → upload to gs://{gcs_bucket}/{blob_name} (+{offset_ms}ms)")
                gcs_uri, blob = upload_to_gcs(chunk_path, gcs_bucket, blob_name)
                uploaded.append((gcs_uri, offset_ms, blob))
            try:
                hans_words = transcribe([(u, o) for u, o, _ in uploaded], project_id, speaker_count)
            finally:
                for _, _, blob in uploaded:
                    try:
                        blob.delete()
                    except Exception as e:
                        print(f"  (warning: failed to delete staged blob: {e})", file=sys.stderr)
        print(f"  → {len(hans_words)} word records; OpenCC s2tw")
        tw_words = [WordRec(word=s2t(w.word), start_ms=w.start_ms, end_ms=w.end_ms) for w in hans_words]
        sentences = build_sentences(tw_words)
        if sentences:
            transcript_json_path.write_text(
                json.dumps(sentences_to_jsonable(sentences), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  → {len(sentences)} sentences → {transcript_json_path.name}")
            boundaries_json_path.write_text(
                json.dumps(boundaries), encoding="utf-8"
            )
            print(f"  → {len(boundaries)} boundaries → {boundaries_json_path.name}")
        else:
            print("  → 0 sentences (not caching empty transcript)")

    if not sentences:
        print("Empty transcript; aborting.", file=sys.stderr)
        sys.exit(1)

    transcript_text = "".join(s.text for s in sentences)

    # ── 3. Tokenize + count ─────────────────────────────────────────────────
    print("\n[3/7] tokenize")
    tokens = tokenize(transcript_text)
    counts = Counter(w for w, _ in tokens)
    print(f"  → {sum(counts.values())} kept tokens, {len(counts)} unique")

    # ── 4. Interactive vocab picker ─────────────────────────────────────────
    print("\n[4/7] vocab")
    try:
        x = int(input("How many new words per round (X): ").strip() or "10")
    except ValueError:
        x = 10
    hsk = load_hsk(HSK_PATH)
    picked_most = pick_round("MOST occurring", counts, hsk, target=x, least_first=False)
    picked_least = pick_round("LEAST occurring", counts, hsk, target=x, least_first=True)
    new_words: dict[str, int] = {**picked_most, **picked_least}
    print(f"\n  → {len(new_words)} total new words")
    if not new_words:
        print("No new words picked; nothing to synthesize. Exiting.")
        sys.exit(0)

    # ── 5. Enrich (pinyin + per-word + per-sentence translation) ────────────
    print("\n[5/7] enrich (translate words + sentences, pinyin)")
    words_list = list(new_words.keys())
    try:
        word_meanings = translate_batch(words_list, project_id)
    except Exception as e:
        print(f"  (word translate failed: {e})", file=sys.stderr)
        word_meanings = {w: "" for w in words_list}

    pinyin_map: dict[str, str] = {}
    for i, w in enumerate(words_list, 1):
        try:
            pinyin_map[w] = transliterate(w, az_key, az_region)
        except Exception as e:
            print(f"    pinyin failed for {w!r}: {e}", file=sys.stderr)
            pinyin_map[w] = ""
        if i % 25 == 0:
            print(f"    pinyin [{i}/{len(words_list)}]")

    # Identify which sentences contain ≥1 new word (preserve order of new words in sentence)
    new_word_set = set(words_list)
    sentence_new_words: dict[int, list[str]] = {}
    for idx, s in enumerate(sentences):
        seen: list[str] = []
        for w, _ in tokenize(s.text):
            if w in new_word_set and w not in seen:
                seen.append(w)
        if seen:
            sentence_new_words[idx] = seen
    print(f"  → {len(sentence_new_words)} sentences contain ≥1 new word")

    if not sentence_new_words:
        print("None of the picked words appear in sentence-level tokens; aborting.")
        sys.exit(0)

    all_sentence_texts = list({s.text for s in sentences})
    print(f"  → translating {len(all_sentence_texts)} sentence(s)")
    try:
        sentence_translations_raw = translate_batch(all_sentence_texts, project_id, batch_size=50)
    except Exception as e:
        print(f"  (sentence translate failed: {e})", file=sys.stderr)
        sentence_translations_raw = {}

    with open(vocab_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "pinyin", "english", "count"])
        for w in words_list:
            writer.writerow([w, pinyin_map.get(w, ""), word_meanings.get(w, ""), new_words[w]])
    print(f"  → wrote {vocab_tsv_path.name}")

    # ── 6. Load source audio + chunk ────────────────────────────────────────
    print("\n[6/7] chunk audio")
    print(f"  → loading {mp3_path.name}")
    original_audio = AudioSegment.from_mp3(str(mp3_path))
    if boundaries:
        chunks = chunk_sentences_by_boundaries(sentences, boundaries)
        print(f"  → {len(chunks)} chunks (reused from silence-aware split)")
    else:
        chunks = chunk_sentences(sentences, target_ms=chunk_target_ms)
        print(f"  → {len(chunks)} chunks (target {args.chunk_minutes:.1f} min each)")

    # ── 7. Build explanation clips, assemble chunks, concat ─────────────────
    print("\n[7/7] synth + assemble")
    sentence_index: dict[int, int] = {id(s): idx for idx, s in enumerate(sentences)}
    explanations: dict[int, AudioSegment] = {}
    for n, (idx, nws) in enumerate(sentence_new_words.items(), 1):
        s = sentences[idx]
        translation = sentence_translations_raw.get(s.text, "")
        explanations[idx] = build_explanation_clip(
            sentence=s,
            new_words_in_order=nws,
            word_meanings=word_meanings,
            sentence_translation=translation,
            original_audio=original_audio,
            tts_cache=tts_cache,
            az_key=az_key,
            az_region=az_region,
        )
        if n % 5 == 0 or n == len(sentence_new_words):
            print(f"  expl [{n}/{len(sentence_new_words)}]")

    final = AudioSegment.silent(duration=0)
    total_chunks = len(chunks)
    for c in chunks:
        chunk_path = chunks_cache / f"chunk_{c.idx:02d}.mp3"
        announcement = render_tts(
            ssml_part_announcement(c.idx, total_chunks),
            tts_cache, az_key, az_region,
        )
        chunk_body = assemble_chunk(
            chunk=c,
            total_chunks=total_chunks,
            original_audio=original_audio,
            explanations_by_sentence=explanations,
            sentence_translations=sentence_translations_raw,
            sentence_index=sentence_index,
            tts_cache=tts_cache,
            az_key=az_key,
            az_region=az_region,
        )
        chunk_audio = announcement + chunk_body
        chunk_audio.export(str(chunk_path), format="mp3", bitrate="192k")
        print(f"  → {chunk_path.name} ({len(chunk_audio) / 1000:.1f}s)")
        final += chunk_audio
        if c.idx != len(chunks):
            final += AudioSegment.silent(duration=INTER_CHUNK_BREAK_MS)

    final_path = outputs_dir / f"{stem}.mp3"
    final.export(
        str(final_path),
        format="mp3",
        bitrate="192k",
        tags={
            "title": stem,
            "artist": "LearnLangs Youtube Converter",
            "album": "LearnLangs Mandarin",
        },
    )
    print(f"\nDone! {len(final) / 1000:.1f}s → {final_path}")


if __name__ == "__main__":
    main()
