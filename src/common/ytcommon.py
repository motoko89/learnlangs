#!/usr/bin/env python3
"""Shared pipeline for the per-language YouTube → Vocab Listening-Practice
generators (see src/<lang>/ytconverter/ytconverter.py).

Everything here is language-agnostic. Language-specific behaviour (script
post-processing, voices, language codes, the OpenAI vocab params) lives in the
per-language scripts and is injected via :class:`LangConfig` or explicit
arguments. Heavy third-party libraries are imported lazily inside the functions
that need them so importing this module is cheap and never forces a
language-specific dependency (opencc, azure, google, openai, ...).
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    chirp_location: str         # Chirp STT region, e.g. "us"
    chirp_model: str            # Chirp model, e.g. "chirp_3"
    mai_locale: str             # MAI-Transcribe locale, e.g. "zh" / "fr"
    sentence_end_chars: str     # chars that terminate a sentence
    sub_sentence_break_chars: str  # chars used to subdivide long sentences
    word_joiner: str            # "" for Chinese, " " for French
    translate_source: str       # Cloud Translate source code, e.g. "zh-TW" / "fr"
    vocab_extra_field: str = ""     # extra OpenAI JSON property name, e.g. "pinyin" ("" if none)
    vocab_extra_explain: str = ""   # sentence describing the extra property ("" if none)
    album: str = ""                 # ID3 album tag for the final MP3, e.g. "LearnLangs French"


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


# ─── yt-dlp download ──────────────────────────────────────────────────────────

def ytdlp_download(url: str, inputs_dir: Path, title: str | None = None) -> Path:
    """Run yt-dlp (extract audio → MP3) and return the downloaded file's path.

    Works for any yt-dlp-supported URL (YouTube, a direct media URL, an Apple
    Podcasts page, ...). When `title` is given it is used verbatim as the output
    filename stem (the caller is responsible for sanitizing it); otherwise the
    extractor's own ``%(title)s`` is used."""
    ensure_dirs(inputs_dir)
    name_template = f"{title}.%(ext)s" if title else "%(title)s.%(ext)s"
    output_template = str(inputs_dir / name_template)
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


def download_youtube(url: str, inputs_dir: Path) -> Path:
    """Run yt-dlp and return the path to the downloaded MP3."""
    return ytdlp_download(url, inputs_dir)


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
    speaker: str | None = None


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
    from google.api_core import exceptions as gexc
    from google.api_core import retry as garetry
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
    # api_core's operation.result() refreshes the LRO via the generic
    # google.longrunning.Operations.GetOperation, which the backend bills to the
    # *v1* operations quota and which the api_core poller hammers (sub-second
    # initial backoff) — that is what 429s and aborts a run. Poll manually
    # through the v2 client's own get_operation (it carries the routing header
    # x-goog-request-params: name=projects/.../locations/<region>/operations/...,
    # so it is billed to the v2 quota) at a fixed, gentle interval, and retry the
    # transient 429s instead of crashing.
    POLL_INTERVAL_S = 15
    POLL_TIMEOUT_S = 3600

    def _on_retry(exc: Exception) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  · [{now}] transient {type(exc).__name__} ({exc}); backing off...", file=sys.stderr)

    retryable = garetry.if_exception_type(
        gexc.ResourceExhausted, gexc.ServiceUnavailable,
        gexc.DeadlineExceeded, gexc.Aborted,
    )
    submit_retry = garetry.Retry(
        predicate=retryable, initial=5.0, maximum=120.0, multiplier=2.0,
        timeout=900.0, on_error=_on_retry,
    )
    poll_retry = garetry.Retry(
        predicate=retryable, initial=5.0, maximum=60.0, multiplier=2.0,
        timeout=300.0, on_error=_on_retry,
    )

    def _await_batch(op_name: str):
        """Poll a BatchRecognize LRO through the v2 service until done and return
        its BatchRecognizeResponse. GetOperation calls here bill the v2 quota."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while True:
            lro = client.get_operation(request={"name": op_name}, retry=poll_retry)
            if lro.done:
                if lro.error.code:
                    raise RuntimeError(
                        f"BatchRecognize failed: code={lro.error.code} {lro.error.message}"
                    )
                return cloud_speech.BatchRecognizeResponse.deserialize(lro.response.value)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"BatchRecognize operation did not finish within {POLL_TIMEOUT_S}s: {op_name}"
                )
            time.sleep(POLL_INTERVAL_S)

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
        operation = client.batch_recognize(request=request, retry=submit_retry)
        response = _await_batch(operation.operation.name)

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
                    speakers = {w.speaker_label for w in alt.words if w.speaker_label}
                    print(f"    [{ri}] words={len(alt.words)} speakers={sorted(speakers)} transcript[:80]={preview!r}")
                for wi in alt.words:
                    if not wi.word:
                        continue
                    words.append(WordRec(
                        word=wi.word,
                        start_ms=_duration_to_ms(wi.start_offset) + offset,
                        end_ms=_duration_to_ms(wi.end_offset) + offset,
                        speaker=wi.speaker_label or None,
                    ))
    words.sort(key=lambda w: w.start_ms)
    return words


# ─── MAI-Transcribe (Azure LLM Speech, synchronous) ───────────────────────────

MAI_API_VERSION = "2025-10-15"


def transcribe_mai(
    flac_chunks: list[tuple[Path, int, int]],
    host: str,
    az_key: str,
    model: str,
    locale: str,
    workers: int = 4,
) -> list[WordRec]:
    """Azure MAI-Transcribe over N local FLAC chunks, returning the same flat,
    offset-applied, sorted WordRec list as :func:`transcribe`.

    flac_chunks = [(path, start_ms, end_ms), ...]. Each chunk's local FLAC is POSTed
    directly to the synchronous LLM Speech ``transcriptions:transcribe`` endpoint (no
    GCS staging). Word timestamps are read from ``phrases[].words[]`` and shifted by the
    chunk's start_ms. Requires word-level timestamps in the response; if a chunk yields
    phrases without a ``words`` field, raises so the missing capability surfaces loudly
    rather than producing an empty transcript."""
    import requests

    url = f"{host.rstrip('/')}/speechtotext/transcriptions:transcribe?api-version={MAI_API_VERSION}"
    definition_json = json.dumps({
        "enhancedMode": {"enabled": True, "model": model, "task": "transcribe"},
        "locales": [locale],
    })

    def _one(chunk: tuple[Path, int, int]) -> list[WordRec]:
        path, offset_ms, _end_ms = chunk
        with open(path, "rb") as fh:
            resp = requests.post(
                url,
                headers={"Ocp-Apim-Subscription-Key": az_key},
                files={
                    "audio": (path.name, fh, "audio/flac"),
                    "definition": (None, definition_json),
                },
                timeout=600,
            )
        if not resp.ok:
            raise RuntimeError(
                f"MAI-Transcribe HTTP {resp.status_code} for {path.name}: {resp.text[:500]}"
            )
        data = resp.json()
        phrases = data.get("phrases") or []
        out: list[WordRec] = []
        saw_words_field = False
        for ph in phrases:
            if "words" in ph:
                saw_words_field = True
            for w in ph.get("words") or []:
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                start = int(w["offsetMilliseconds"]) + offset_ms
                end = start + int(w["durationMilliseconds"])
                out.append(WordRec(word=text, start_ms=start, end_ms=end))
        if phrases and not saw_words_field:
            raise RuntimeError(
                f"MAI-Transcribe response for {path.name} has phrases but no word-level "
                f"timestamps ('words' field absent); cannot build word records. "
                f"phrase keys={list(phrases[0].keys())}."
            )
        return out

    n = max(1, min(workers, len(flac_chunks)))
    print(f"  → MAI-Transcribe ({model}, locale={locale}) on {len(flac_chunks)} chunk(s), {n} worker(s)")
    words: list[WordRec] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(_one, c): c for c in flac_chunks}
        for fut in as_completed(futures):
            path, offset_ms, _end_ms = futures[fut]
            chunk_words = fut.result()
            print(f"  · {path.name}: {len(chunk_words)} word(s) (+{offset_ms}ms)")
            words.extend(chunk_words)
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
    if joiner:
        # Space-delimited language: join with the configured joiner, then drop
        # whitespace inserted before punctuation.
        text = joiner.join(x.word for x in words).strip()
        return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    # No-joiner language (e.g. Chinese): concatenate directly, but keep a space
    # between two adjacent ASCII tokens so embedded Latin-script words/numbers
    # (e.g. "machine" + "learning", "iPhone" + "15") aren't mashed together.
    parts: list[str] = []
    prev_ascii = False
    for x in words:
        cur_ascii = bool(x.word) and x.word.isascii()
        if parts and prev_ascii and cur_ascii:
            parts.append(" ")
        parts.append(x.word)
        prev_ascii = cur_ascii
    text = "".join(parts).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


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


def build_sentences(
    words: list[WordRec], cfg: LangConfig, split_on_speaker_change: bool = False
) -> list[Sentence]:
    """Three-pass segmentation:
    1) Split at cfg.sentence_end_chars. When split_on_speaker_change is set
       (diarization with >1 speaker), also force a boundary wherever the
       speaker label changes between adjacent words.
    2) Sentences shorter than MIN_SENTENCE_MS are merged into the next one
       (or, if the tail is short, into the previous one).
    3) Sentences longer than MAX_SENTENCE_MS are subdivided into
       ceil(length / MAX_SENTENCE_MS) roughly equal pieces."""
    raw: list[list[WordRec]] = []
    buf: list[WordRec] = []
    prev_speaker: str | None = None
    for w in words:
        if (
            split_on_speaker_change
            and buf
            and prev_speaker
            and w.speaker
            and w.speaker != prev_speaker
        ):
            raw.append(buf)
            buf = []
        buf.append(w)
        if w.word and w.word[-1] in cfg.sentence_end_chars:
            raw.append(buf)
            buf = []
        if w.speaker:
            prev_speaker = w.speaker
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


def _srt_timestamp(ms: int) -> str:
    """Format milliseconds as an SRT timestamp: HH:MM:SS,mmm."""
    ms = max(0, ms)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def sentences_to_srt(sentences: list[Sentence]) -> str:
    """Render sentences as an SRT subtitle file, one cue per sentence."""
    blocks: list[str] = []
    for i, s in enumerate(sentences, 1):
        blocks.append(
            f"{i}\n"
            f"{_srt_timestamp(s.start_ms)} --> {_srt_timestamp(s.end_ms)}\n"
            f"{s.text}\n"
        )
    return "\n".join(blocks)


def write_transcript_files(
    sentences: list[Sentence], json_path: Path, srt_path: Path, txt_path: Path
) -> None:
    """Write the transcript in all three sibling formats: JSON (full records),
    SRT (one cue per sentence), and TXT (one sentence text per line)."""
    json_path.write_text(
        json.dumps(sentences_to_jsonable(sentences), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    srt_path.write_text(sentences_to_srt(sentences), encoding="utf-8")
    txt_path.write_text("\n".join(s.text for s in sentences) + "\n", encoding="utf-8")
    print(f"  → {len(sentences)} sentences → {json_path.name}, {srt_path.name}, {txt_path.name}")


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


# ─── Azure OpenAI vocab extraction ────────────────────────────────────────────

VOCAB_MODEL = "gpt-5.4"
VOCAB_SYSTEM = "Act as language learning API"


def _vocab_prompt(
    count: int,
    extra_field: str,
    extra_explain: str,
) -> str:
    """Build the message from the fixed template.

    ``count`` is how many words/phrases to extract; ``extra_field`` is the extra
    JSON property (e.g. "pinyin") and ``extra_explain`` the sentence describing
    it. Empty extra_field/extra_explain drop their clauses entirely (the "" if
    not needed case). All explanations are plain text — no SSML is produced
    here; the playback SSML is synthesized downstream."""
    param1 = f', "{extra_field}"' if extra_field else ""
    param4 = f" {extra_explain}" if extra_explain else ""
    return (
        f"Identify top lexical units up to {count} in the attached transcript that "
        "will most improve comprehension of the transcript. A lexical unit may be:"
        "- a fixed expression; - a collocation; - a discourse marker; - a verb-object phrase; - a multi-word chunk; - or, when appropriate, a single word.; "
        "Prefer the longest meaningful unit that native speakers naturally process as one idea. Avoid extracting individual words that are only useful as part of a larger expression. "
        "Rank items by how much knowing them would improve understanding of this transcript and future native conversations, not by rarity."
        "Prioritize: - High-frequency conversational expressions.; - Fixed phrases and collocations.; - Discourse markers and connective expressions.; "
        "- Common spoken idioms.; - Frequently used verb-object combinations.; - Topic-specific terms only if they are essential to understanding this transcript.; "
        "Do NOT prioritize obscure lexical units simply because it is difficult. "
        "Avoid selecting proper nouns, place names, organization names, people's names, or technical terms unless they are essential to understanding the conversation or are likely to appear in many future conversations. "
        "Prefer lexical units characteristic of natural spoken over formal written lexical units. "
        "When two candidates are equally useful, prefer the one that: - appears multiple times in the transcript,; - or represents a recurring concept,; - or is useful in many everyday conversations.; "
        "Output these words/phrases into a JSON array. Each JSON object has these properties: "
        f'"text", "longExplain", "shortExplain"{param1}. "text" '
        'is the original text. "longExplain" is the explanation of the text in '
        "the context of the transcript, its format is plain text. "
        'Long explains must be 200-character long maximum, keep them succinct. Do not explain in target language then translate to English, just go straight to English explanation. "shortExplain" is just short English translation. Do not repeat "text" at the beginning of "longExplain". Do not start with "this is" at the beginning of "longExplain", go straight to explaining'
        f"{param4} No other response needed"
    )


def extract_vocab(
    transcript_text: str,
    api_key: str,
    base_url: str,
    vocab_number: int = 40,
    extra_field: str = "",
    extra_explain: str = "",
    model: str = VOCAB_MODEL,
) -> list[dict]:
    """Ask Azure OpenAI (high reasoning effort) for the top-`vocab_number` vocab
    items and return the parsed JSON array. Each item carries 'text',
    'longExplain', 'shortExplain' (+ extra_field, e.g. 'pinyin'). 'longExplain'
    and 'shortExplain' are plain English text; the playback SSML is synthesized
    from them downstream (see :func:`build_vocab_ssml_by_sentence`)."""
    from openai import OpenAI

    prompt = _vocab_prompt(vocab_number, extra_field, extra_explain)
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": VOCAB_SYSTEM},
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript_text}"},
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"Azure OpenAI vocab response had no JSON array:\n{text[:1000]}")
    return json.loads(text[start : end + 1])


def build_vocab_ssml_by_sentence(
    vocab: list[dict], sentences: list["Sentence"], cfg: "LangConfig"
) -> dict[int, list[str]]:
    """Resolve the per-sentence vocab explanation SSML, in playback order.

    A vocab item is cued in every sentence whose text contains its 'text'. Each
    cue plays the word itself (foreign voice) and then its explanation, so two
    SSML docs are emitted per cue: ``ssml_native_word(text)`` followed by the
    explanation. The first time a 'text' appears (scanning sentences in time
    order) the explanation is its full ``longExplain`` (English-only SSML built
    by :func:`ssml_long_explain`); every later sentence it reappears in gets a
    short English-only clip built from ``shortExplain``.
    Returns {sentence_idx: [ssml, ...]} for sentences with ≥1 cue (vocab order
    within a sentence; rendered in order with breaks by
    :func:`build_explanation_clip`). Items whose text never appears verbatim are
    skipped (still kept in vocab.json/tsv)."""
    seen: set[str] = set()
    by_sentence: dict[int, list[str]] = {}
    for idx, s in enumerate(sentences):
        for v in vocab:
            text = (v.get("text") or "").strip()
            if not text or text not in s.text:
                continue
            if text not in seen:
                seen.add(text)
                explain = ssml_long_explain(v.get("longExplain", ""), cfg)
            else:
                explain = ssml_short_explain(v.get("shortExplain", ""), cfg)
            if explain:
                by_sentence.setdefault(idx, []).extend(
                    [ssml_native_word(text, cfg), explain]
                )
    return by_sentence


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

# A <break> that sits directly under <speak> (a sibling of <voice>, not nested
# inside it) is rejected by some voices/endpoints — notably DragonHD — with
# "Node [speak] ... should not contain node [break]" (error 1007). The OpenAI
# vocab SSML emits exactly this between its two <voice> blocks. Move each such
# break to just inside the preceding </voice> so it remains a valid pause.
_ROOT_BREAK_AFTER_VOICE_RE = re.compile(r"</voice>\s*(<break\b[^>]*/>)")

# LLM-generated SSML occasionally emits a stray duplicate </voice> before
# </speak> (e.g. "...</prosody></voice></voice></speak>"). Azure rejects this
# with error 1007 ("The 'speak' start tag ... does not match the end tag of
# 'voice'"). <voice> can never be nested, so any run of consecutive closing
# </voice> tags is always wrong — collapse it to a single one.
_DUP_CLOSE_VOICE_RE = re.compile(r"</voice>(\s*</voice>)+")


def sanitize_ssml(ssml: str) -> str:
    ssml = _ROOT_BREAK_AFTER_VOICE_RE.sub(r"\1</voice>", ssml)
    ssml = _DUP_CLOSE_VOICE_RE.sub("</voice>", ssml)
    return ssml


def render_tts(ssml: str, cache_dir: Path, az_key: str, az_region: str) -> AudioSegment:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ssml = sanitize_ssml(ssml)
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
            # The service decides a request that is too slow (RTF/frame-interval
            # threshold) and cancels it — transient and retryable. Bad SSML, by
            # contrast, is cancelled with a definite error code and never recovers.
            # Retry the former with backoff; surface the latter immediately.
            transient_codes = {
                speechsdk.CancellationErrorCode.ServiceTimeout,
                speechsdk.CancellationErrorCode.ServiceUnavailable,
                speechsdk.CancellationErrorCode.ConnectionFailure,
                speechsdk.CancellationErrorCode.ServiceError,
                speechsdk.CancellationErrorCode.RuntimeError,
            }
            max_attempts = 4
            for attempt in range(1, max_attempts + 1):
                audio_config = speechsdk.audio.AudioOutputConfig(filename=str(tmp_path))
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                result = synthesizer.speak_ssml_async(ssml).get()
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    break
                details = getattr(result, "cancellation_details", None)
                err = details.error_details if details else "unknown"
                code = details.error_code if details else None
                if code in transient_codes and attempt < max_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise RuntimeError(
                    f"Azure TTS failed: {result.reason} / {err}\nSSML was:\n{ssml}"
                )
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


def ssml_en_sentence(en_text: str, cfg: LangConfig) -> str:
    """English-only SSML for a sentence's translation, spoken in the EN voice."""
    return _wrap_ssml(
        _voice(cfg.en_voice, en_text or "(no translation)", cfg.tts_rate),
        cfg.xml_lang,
    )


def ssml_long_explain(long_text: str, cfg: LangConfig) -> str:
    """English-only SSML used the first time a vocab item appears: the full
    English explanation in the EN voice (no foreign voice). Built from the
    plain-text ``longExplain`` returned by :func:`extract_vocab`."""
    return _wrap_ssml(
        _voice(cfg.en_voice, long_text or "(no explanation)", cfg.tts_rate),
        cfg.xml_lang,
    )


def ssml_short_explain(short_text: str, cfg: LangConfig) -> str:
    """English-only SSML used when a vocab item reappears in a later sentence:
    just the short English translation in the EN voice (no foreign voice)."""
    return _wrap_ssml(
        _voice(cfg.en_voice, short_text or "(no translation)", cfg.tts_rate),
        cfg.xml_lang,
    )


def ssml_native_word(text: str, cfg: LangConfig) -> str:
    """The vocab word/phrase itself, spoken in the foreign voice. Played before
    each explanation so the learner hears the term first, then the breakdown."""
    return _wrap_ssml(_voice(cfg.native_voice, text, cfg.tts_rate), cfg.xml_lang)


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
    vocab_ssml_in_order: list[str],
    sentence_translation: str,
    original_audio: AudioSegment,
    tts_cache: Path,
    az_key: str,
    az_region: str,
    cfg: LangConfig,
) -> AudioSegment:
    """Per-sentence clip for sentences containing ≥1 vocab cue:
    per-vocab SSML (the word in the foreign voice, then its explanation) +
    original_slice + EN sentence translation + original_slice, with 500 ms
    breaks. Each entry of vocab_ssml_in_order is a full Azure SSML document
    rendered directly; cues arrive as word/explanation pairs from
    :func:`build_vocab_ssml_by_sentence`."""
    gap = AudioSegment.silent(duration=INTRA_GROUP_BREAK_MS)
    clip_a = AudioSegment.silent(duration=0)
    for i, ssml in enumerate(s for s in vocab_ssml_in_order if s):
        if i:
            clip_a += gap
        clip_a += render_tts(ssml, tts_cache, az_key, az_region)
    original_slice = original_audio[sentence.start_ms : sentence.end_ms]
    clip_b = render_tts(
        ssml_en_sentence(sentence_translation, cfg),
        tts_cache, az_key, az_region,
    )
    return clip_a + gap + original_slice + gap + clip_b + gap + original_slice


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
    per-sentence playbacks (explanation clip for new-vocab sentences only;
    sentences without any vocab cue are skipped) — pauses after every
    played sentence — then a "Playback part X of Y" announcement and a
    replay of the original chunk."""
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

    # After the explanations, announce and replay the whole original chunk.
    replay_announcement = render_tts(
        ssml_replay_announcement(chunk.idx, total_chunks, cfg),
        tts_cache, az_key, az_region,
    )
    out += pad + replay_announcement + pad
    out += original_slice
    return out
