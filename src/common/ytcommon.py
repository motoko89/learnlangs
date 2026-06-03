#!/usr/bin/env python3
"""Shared pipeline for the per-language YouTube → Vocab Listening-Practice
generators (see src/<lang>/ytconverter/ytconverter.py).

Everything here is language-agnostic. Language-specific behaviour (tokenizer,
script post-processing, pinyin/transliteration, word filter, voices, language
codes) lives in the per-language scripts and is injected via :class:`LangConfig`
or explicit arguments. Heavy third-party libraries are imported lazily inside
the functions that need them so importing this module is cheap and never forces
a language-specific dependency (jieba, opencc, spacy, azure, google, ...).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydub import AudioSegment

# ─── Language-agnostic constants ──────────────────────────────────────────────

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

INTRA_GROUP_BREAK_MS = 500
INTER_PART_BREAK_MS = 1000
INTER_CHUNK_BREAK_MS = 2000
CHUNK_ANNOUNCEMENT_PAD_MS = 600
NO_VOCAB_BREAK_MS = 600

SILENCE_LEN_MS = 500
SILENCE_THRESH_DB = -16  # dB below the audio's average dBFS
SILENCE_SEARCH_WINDOW_MS = 60 * 1000

MIN_SENTENCE_MS = 3000
MAX_SENTENCE_MS = 8000

PUNCT_OR_DIGIT_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)
# Collapse whitespace that sits before punctuation when joining word tokens with
# spaces (only relevant for space-delimited languages such as French).
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?…»)\]])")


# ─── Per-language configuration ───────────────────────────────────────────────

@dataclass
class LangConfig:
    """Everything the shared functions need to know about a target language."""

    native_voice: str           # e.g. "zh-TW-YunJheNeural" / "fr-FR-DeniseNeural"
    en_voice: str               # e.g. "en-US-AvaNeural"
    tts_rate: str               # e.g. "0.9"
    xml_lang: str               # SSML xml:lang, e.g. "zh-TW" / "fr-FR"
    language_code: str          # STT language code, e.g. "cmn-Hans-CN" / "fr-FR"
    chirp_location: str         # STT region, e.g. "us"
    chirp_model: str            # e.g. "chirp_3"
    sentence_end_chars: str     # chars that terminate a sentence
    sub_sentence_break_chars: str  # chars used to subdivide long sentences
    word_joiner: str            # "" for Chinese, " " for French
    translate_source: str       # Cloud Translate source code, e.g. "zh-TW" / "fr"


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


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def sanitize_stem(stem: str) -> str:
    """Make a filesystem-safe stem (also used as cache directory name)."""
    cleaned = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip()
    return cleaned or f"episode-{int(time.time())}"


# ─── Known-words list (HSK list for Mandarin, learntwords for French, ...) ─────

def ensure_known_file(path: Path, header: str) -> None:
    if not path.exists():
        path.write_text(header, encoding="utf-8")
        print(f"  created empty known-words list: {path}")


def load_known_words(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def append_known_word(path: Path, word: str) -> None:
    data = path.read_bytes() if path.exists() else b""
    needs_newline = data and not data.endswith(b"\n")
    with open(path, "ab") as f:
        if needs_newline:
            f.write(b"\n")
        f.write(word.encode("utf-8") + b"\n")


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


# ─── Audio convert + GCS upload ───────────────────────────────────────────────

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


def transcribe(
    files: list[tuple[str, int]],
    project_id: str,
    location: str,
    model: str,
    language_code: str,
    min_speaker_count: int,
    max_speaker_count: int,
) -> list[WordRec]:
    """Chirp BatchRecognize across N files. files = [(gcs_uri, offset_ms), ...].
    Returns a flat list of word records, timestamps already offset and sorted."""
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    client = SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
    )
    recognizer = f"projects/{project_id}/locations/{location}/recognizers/_"
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language_code],
        model=model,
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=min_speaker_count,
                max_speaker_count=max_speaker_count,
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
        print(f"  → [{i}/{len(files)}] STT v2 ({model}) BatchRecognize submitted (+{offset}ms); waiting...")
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


# ─── Sentence segmentation from word records ──────────────────────────────────

@dataclass
class Sentence:
    text: str
    start_ms: int
    end_ms: int
    words: list[WordRec] = field(default_factory=list)


def _join_words(words: list[WordRec], joiner: str) -> str:
    text = joiner.join(x.word for x in words).strip()
    if joiner:
        # Space-delimited language: drop whitespace inserted before punctuation.
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def _sentence_from_words(buf: list[WordRec], joiner: str) -> Sentence:
    return Sentence(
        text=_join_words(buf, joiner),
        start_ms=buf[0].start_ms,
        end_ms=buf[-1].end_ms,
        words=list(buf),
    )


def _split_long_sentence(s: Sentence, joiner: str, sub_break_chars: str) -> list[Sentence]:
    """Break a sentence longer than MAX_SENTENCE_MS into roughly equal pieces.
    Splits at sub_break_chars when there are enough of them; falls back to
    between-word positions to fill any remaining splits."""
    length = s.end_ms - s.start_ms
    if length <= MAX_SENTENCE_MS:
        return [s]
    n_words = len(s.words)
    if n_words < 2:
        return [s]
    n_subs = (length + MAX_SENTENCE_MS - 1) // MAX_SENTENCE_MS
    n_splits = min(n_subs - 1, n_words - 1)

    sub_breaks = [
        i for i in range(n_words - 1)
        if s.words[i].word and s.words[i].word[-1] in sub_break_chars
    ]
    candidates = sub_breaks if len(sub_breaks) >= n_splits else list(range(n_words - 1))

    chosen: list[int] = []
    remaining = list(candidates)
    for k in range(n_splits):
        if not remaining:
            break
        ideal = s.start_ms + (k + 1) * length / n_subs
        best = min(remaining, key=lambda i: abs(s.words[i].end_ms - ideal))
        chosen.append(best)
        remaining = [r for r in remaining if r > best]

    chosen.sort()
    pieces: list[Sentence] = []
    last = 0
    for ci in chosen:
        pieces.append(_sentence_from_words(s.words[last : ci + 1], joiner))
        last = ci + 1
    pieces.append(_sentence_from_words(s.words[last:], joiner))
    return pieces


def build_sentences(words: list[WordRec], cfg: LangConfig) -> list[Sentence]:
    """Three-pass segmentation:
    1) Split at cfg.sentence_end_chars.
    2) Sentences shorter than MIN_SENTENCE_MS are merged into the next one
       (or, if the tail is short, into the previous one).
    3) Sentences longer than MAX_SENTENCE_MS are subdivided into
       ceil(length / MAX_SENTENCE_MS) roughly equal pieces."""
    raw: list[list[WordRec]] = []
    buf: list[WordRec] = []
    for w in words:
        buf.append(w)
        if w.word and w.word[-1] in cfg.sentence_end_chars:
            raw.append(buf)
            buf = []
    if buf:
        raw.append(buf)

    merged: list[list[WordRec]] = []
    carry: list[WordRec] = []
    for piece in raw:
        combined = carry + piece
        duration = combined[-1].end_ms - combined[0].start_ms
        if duration < MIN_SENTENCE_MS:
            carry = combined
        else:
            merged.append(combined)
            carry = []
    if carry:
        if merged:
            merged[-1] = merged[-1] + carry
        else:
            merged.append(carry)

    out: list[Sentence] = []
    for piece in merged:
        out.extend(_split_long_sentence(
            _sentence_from_words(piece, cfg.word_joiner),
            cfg.word_joiner,
            cfg.sub_sentence_break_chars,
        ))
    return out


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


# ─── Interactive vocab picker ─────────────────────────────────────────────────

def pick_round(
    label: str,
    counts: Counter,
    known: set[str],
    target: int,
    least_first: bool,
    word_re: re.Pattern,
    known_path: Path,
) -> dict[str, int]:
    """Iterate candidates and prompt single-keypress 1=NEW, 2=KNOWN, q=stop.
    Returns dict {word: count} for accepted-as-new words. Words answered KNOWN
    are appended to known_path."""
    sign = 1 if least_first else -1
    ordered = sorted(
        counts.items(),
        key=lambda x: (sign * x[1], -len(x[0]), x[0]),
    )
    picked: dict[str, int] = {}
    print(f"\n── {label} — pick up to {target}; 1=NEW, 2=KNOWN, q=stop round ──")
    for word, count in ordered:
        if word in known:
            continue
        if not word_re.match(word):
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
            known.add(word)
            append_known_word(known_path, word)
            print("    → KNOWN (added to known-words list)")
        elif ch.lower() == "q":
            print("    → stop")
            break
        else:
            print("    → skip")
    return picked


# ─── Cloud Translate v3 ───────────────────────────────────────────────────────

def translate_batch(
    texts: list[str],
    project_id: str,
    source_language_code: str,
    batch_size: int = 100,
) -> dict[str, str]:
    """source_language_code → en via Cloud Translate v3 (service-account auth
    from GOOGLE_APPLICATION_CREDENTIALS)."""
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
            source_language_code=source_language_code,
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


def chunk_sentences(sentences: list[Sentence], target_ms: int) -> list[Chunk]:
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

        # Atomic write so concurrent renders of the same SSML never read a partial file.
        fd, tmp_str = tempfile.mkstemp(suffix=".mp3", prefix=f".{sha}.", dir=str(cache_dir))
        os.close(fd)
        tmp_path = Path(tmp_str)
        try:
            speech_config = speechsdk.SpeechConfig(subscription=az_key, region=az_region)
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
            )
            audio_config = speechsdk.audio.AudioOutputConfig(filename=str(tmp_path))
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            result = synthesizer.speak_ssml_async(ssml).get()
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                details = getattr(result, "cancellation_details", None)
                err = details.error_details if details else "unknown"
                raise RuntimeError(f"Azure TTS failed: {result.reason} / {err}")
            os.replace(tmp_path, out_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return AudioSegment.from_mp3(str(out_path))


def _voice(name: str, text: str, rate: str, lead_break_ms: int = 0, trail_break_ms: int = 0) -> str:
    lead = f'<break time="{lead_break_ms}ms"/>' if lead_break_ms else ""
    trail = f'<break time="{trail_break_ms}ms"/>' if trail_break_ms else ""
    return (
        f'<voice name="{name}">'
        f'<prosody rate="{rate}">{lead}{html.escape(text)}{trail}</prosody>'
        f'</voice>'
    )


def _wrap_ssml(body: str, xml_lang: str) -> str:
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{xml_lang}">'
        + body
        + '</speak>'
    )


def ssml_words_and_meanings(pairs: list[tuple[str, str]], cfg: LangConfig) -> str:
    """pairs = [(native_word, english_meaning), ...]"""
    parts: list[str] = []
    for i, (w, en) in enumerate(pairs):
        # break before each pair after the first — placed inside the leading voice
        lead = INTRA_GROUP_BREAK_MS if i else 0
        parts.append(_voice(cfg.native_voice, w, cfg.tts_rate, lead_break_ms=lead, trail_break_ms=INTRA_GROUP_BREAK_MS))
        parts.append(_voice(cfg.en_voice, en or "(no translation)", cfg.tts_rate))
    return _wrap_ssml("".join(parts), cfg.xml_lang)


def ssml_sentence_pair(en_text: str, native_text: str, cfg: LangConfig) -> str:
    body = (
        _voice(cfg.en_voice, en_text or "(no translation)", cfg.tts_rate, trail_break_ms=INTRA_GROUP_BREAK_MS)
        + _voice(cfg.native_voice, native_text, cfg.tts_rate)
    )
    return _wrap_ssml(body, cfg.xml_lang)


def ssml_chunk_announcement(idx: int, total: int, cfg: LangConfig) -> str:
    return _wrap_ssml(_voice(cfg.en_voice, f"Explaining part {idx} of {total}.", cfg.tts_rate), cfg.xml_lang)


def ssml_replay_announcement(idx: int, total: int, cfg: LangConfig) -> str:
    return _wrap_ssml(_voice(cfg.en_voice, f"Playback part {idx} of {total}.", cfg.tts_rate), cfg.xml_lang)


def ssml_part_announcement(part_num: int, total_parts: int, cfg: LangConfig) -> str:
    body = _voice(
        cfg.en_voice,
        f"Part {part_num} of {total_parts}",
        cfg.tts_rate,
        lead_break_ms=600,
        trail_break_ms=600,
    )
    return _wrap_ssml(body, cfg.xml_lang)


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
    cfg: LangConfig,
) -> AudioSegment:
    pairs = [(w, word_meanings.get(w, "")) for w in new_words_in_order]
    clip_a = render_tts(ssml_words_and_meanings(pairs, cfg), tts_cache, az_key, az_region)
    original_slice = original_audio[sentence.start_ms : sentence.end_ms]
    clip_b = render_tts(
        ssml_sentence_pair(sentence_translation, sentence.text, cfg),
        tts_cache, az_key, az_region,
    )
    gap = AudioSegment.silent(duration=INTRA_GROUP_BREAK_MS)
    return clip_a + gap + original_slice + gap + clip_b + gap + original_slice


def build_no_vocab_clip(
    sentence: Sentence,
    sentence_translation: str,
    original_audio: AudioSegment,
    tts_cache: Path,
    az_key: str,
    az_region: str,
    cfg: LangConfig,
) -> AudioSegment:
    """Per-sentence clip for sentences without any new vocab:
    original_slice + 600 + EN_TTS + 600 + NATIVE_TTS + 600 (trailing pause)."""
    original_slice = original_audio[sentence.start_ms : sentence.end_ms]
    en_clip = render_tts(
        _wrap_ssml(_voice(cfg.en_voice, sentence_translation or "(no translation)", cfg.tts_rate), cfg.xml_lang),
        tts_cache, az_key, az_region,
    )
    native_clip = render_tts(
        _wrap_ssml(_voice(cfg.native_voice, sentence.text, cfg.tts_rate), cfg.xml_lang),
        tts_cache, az_key, az_region,
    )
    gap = AudioSegment.silent(duration=NO_VOCAB_BREAK_MS)
    return original_slice + gap + en_clip + gap + native_clip + gap + original_slice + gap


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
    cfg: LangConfig,
) -> AudioSegment:
    """Build: original_chunk + 1s + 600ms + announcement + 600ms +
    per-sentence playbacks (explanation clip for new-vocab sentences,
    no-vocab clip otherwise) — pauses after every sentence — then a
    "Playback part X of Y" announcement and a replay of the original chunk."""
    original_slice = original_audio[chunk.start_ms : chunk.end_ms]
    out = original_slice
    out += AudioSegment.silent(duration=INTER_PART_BREAK_MS)
    announcement = render_tts(
        ssml_chunk_announcement(chunk.idx, total_chunks, cfg),
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
                cfg=cfg,
            )

    # After the explanations, announce and replay the whole original chunk.
    replay_announcement = render_tts(
        ssml_replay_announcement(chunk.idx, total_chunks, cfg),
        tts_cache, az_key, az_region,
    )
    out += pad + replay_announcement + pad
    out += original_slice
    return out
