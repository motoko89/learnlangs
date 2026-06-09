#!/usr/bin/env python3
"""French YouTube → Vocab Listening-Practice Generator.

End-to-end pipeline:
  1. Prompt for a YouTube URL.
  2. Download audio (yt-dlp -x mp3) into ./inputs/.
  3. Transcribe with Google Chirp 3 (default; pass --stt mai for Azure
     MAI-Transcribe-1.5), with word-level timestamps. Cache JSON.
  4. Tokenize/lemmatize with spaCy (fr_core_news_sm) and count occurrences.
  5. Interactively pick X most-occurring + X least-occurring NEW words
     (single keypress: 1 = NEW, 2 = mark KNOWN/append to learntwords.txt).
  6. Slice the source audio into ~10-min chunks snapped to sentence ends.
  7. For each sentence containing ≥1 new word, render an Azure TTS explanation
     clip = word(s) + English meaning(s) + original sentence slice + synthetic
     sentence TTS + English sentence translation, with 500 ms breaks.
  8. Assemble each chunk:
       original_chunk + 1s + part1 + expl1 + 1s + part2 + expl2 + 1s + … + tail
     and concatenate all chunks (2s between chunks) into outputs/<stem>.mp3.

Language-agnostic pipeline code lives in src/common/ytcommon.py; this script
only carries the French-specific pieces (spaCy tokenizer/lemmatizer, the
learnt-words list, and the voices).

I/O folders (created at invocation cwd):
  inputs/                 - downloaded MP3
  intermediates/<stem>/   - transcript.json, vocab.tsv, tts/, chunks/
  outputs/                - final concatenated study MP3

Credentials (next to this script):
  key.json       - {"azSpeechKey": "<Azure Cognitive Services key>",
                    "azSpeechRegion": "<e.g. eastus>",
                    "azSttEndpoint": "<Foundry host for MAI-Transcribe; only for --stt mai,
                                       e.g. https://<resource>.cognitiveservices.azure.com;
                                       optional, else derived from azSpeechRegion>",
                    "gcsBucket": "<GCS bucket for STT staging (default Chirp path)>"}
  jumeau-gc.json - Google Cloud service account JSON (used for Chirp STT + Translate v3)

Dependencies:
  pip install -r requirements.txt
  python -m spacy download fr_core_news_sm
  brew install ffmpeg   # pydub MP3 decode; also used by yt-dlp
  python3 ytconverter.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment

# Make src/ importable so `from common.ytcommon import ...` works when the
# script is run directly from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.ytcommon import (  # noqa: E402
    INTER_CHUNK_BREAK_MS,
    Chunk,
    LangConfig,
    WordRec,
    assemble_chunk,
    build_explanation_clip,
    build_sentences,
    chunk_sentences,
    chunk_sentences_by_boundaries,
    download_youtube,
    ensure_dirs,
    ensure_known_file,
    load_keys,
    load_known_words,
    pick_round,
    render_tts,
    sanitize_stem,
    sentences_from_jsonable,
    split_mp3_to_flac_chunks,
    ssml_part_announcement,
    transcribe,
    transcribe_mai,
    translate_batch,
    upload_to_gcs,
    write_transcript_files,
)

# spaCy Universal POS tags to drop: function words, punctuation, numbers, etc.
SKIP_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART", "PUNCT", "NUM", "SYM", "X", "SPACE", "INTJ"}
# Vocab-picker filter: a French word (lowercase, accents, internal apostrophe/hyphen).
FRENCH_WORD_RE = re.compile(r"^[a-zà-öø-ÿœæ][a-zà-öø-ÿœæ'’\-]*$")

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWN_PATH = SCRIPT_DIR / "learntwords.txt"
KNOWN_HEADER = (
    "# Learnt French words — one base form (lemma) per line.\n"
    "# Words NOT in this list are flagged as candidates by ytconverter.py.\n"
    "# Add one word per line; lines starting with '#' are ignored.\n\n"
)

FRENCH = LangConfig(
    native_voice="fr-FR-Remy:DragonHDLatestNeural",
    en_voice="en-US-AvaNeural",
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
)


# ─── French tokenization (spaCy lemmatizer) ───────────────────────────────────

_NLP = None


def tokenize(text: str) -> list[tuple[str, str]]:
    """Return [(lemma, pos), ...] keeping content words only. Lemmatizing means
    conjugations/inflections collapse to their base form (mange/mangé/mangeons
    → manger; belle/beaux → beau)."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

    out: list[tuple[str, str]] = []
    for t in _NLP(text):
        lemma = t.lemma_.lower().strip()
        if not lemma or t.pos_ in SKIP_POS:
            continue
        if len(lemma) < 2 or not FRENCH_WORD_RE.match(lemma):
            continue
        out.append((lemma, t.pos_))
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
    parser.add_argument("--stt", choices=["mai", "chirp"], default="chirp",
                        help="STT backend: chirp = Google Chirp 3 (default), mai = Azure MAI-Transcribe")
    parser.add_argument("--stt-model", default="mai-transcribe-1.5",
                        help="MAI-Transcribe model id (used when --stt mai; default: mai-transcribe-1.5)")
    parser.add_argument("--min-speakers", type=int, default=2, help="Minimum speakers for diarization (chirp only; default: 2, min 1)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers for diarization (chirp only; default: same as --min-speakers; clamped to >= min)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel chunk-build workers (default: 4)")
    args = parser.parse_args()

    ensure_known_file(KNOWN_PATH, KNOWN_HEADER)

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
    if args.stt == "chirp" and not gcs_bucket:
        gcs_bucket = input("GCS bucket for STT staging: ").strip()
    stt_endpoint = keys.get("azSttEndpoint") or f"https://{az_region}.api.cognitive.microsoft.com"

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
    transcript_srt_path = inter_dir / "transcript.srt"
    transcript_txt_path = inter_dir / "transcript.txt"
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
        speaker_split = False
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            print(f"  → MP3 → FLAC chunks (16kHz mono, ~{chunk_target_ms // 60000}min each, split at silence)")
            flac_chunks = split_mp3_to_flac_chunks(mp3_path, chunk_target_ms, td_path)
            print(f"  → {len(flac_chunks)} chunk(s)")
            boundaries = [(start, end) for _, start, end in flac_chunks]
            if args.stt == "mai":
                print(f"  → MAI-Transcribe via {stt_endpoint}")
                fr_words = transcribe_mai(
                    flac_chunks, stt_endpoint, az_key, args.stt_model, FRENCH.mai_locale, args.workers,
                )
            else:
                min_speakers = max(1, args.min_speakers)
                max_speakers = args.max_speakers if args.max_speakers is not None else min_speakers
                max_speakers = max(min_speakers, max_speakers)
                speaker_split = max_speakers > 1
                print(f"  → diarization: {min_speakers}-{max_speakers} speaker(s)")
                timestamp = int(time.time())
                uploaded: list[tuple[str, int, object]] = []  # (uri, offset_ms, blob)
                for chunk_path, offset_ms, _end_ms in flac_chunks:
                    blob_name = f"stt-staging/{stem}-{timestamp}-{chunk_path.stem}.flac"
                    print(f"  → upload to gs://{gcs_bucket}/{blob_name} (+{offset_ms}ms)")
                    gcs_uri, blob = upload_to_gcs(chunk_path, gcs_bucket, blob_name)
                    uploaded.append((gcs_uri, offset_ms, blob))
                try:
                    fr_words = transcribe(
                        [(u, o) for u, o, _ in uploaded],
                        project_id,
                        FRENCH.chirp_location,
                        FRENCH.chirp_model,
                        FRENCH.language_code,
                        min_speakers,
                        max_speakers,
                    )
                finally:
                    for _, _, blob in uploaded:
                        try:
                            blob.delete()
                        except Exception as e:
                            print(f"  (warning: failed to delete staged blob: {e})", file=sys.stderr)
        print(f"  → {len(fr_words)} word records")
        sentences = build_sentences(fr_words, FRENCH, split_on_speaker_change=speaker_split)
        if sentences:
            write_transcript_files(
                sentences, transcript_json_path, transcript_srt_path, transcript_txt_path
            )
            boundaries_json_path.write_text(
                json.dumps(boundaries), encoding="utf-8"
            )
            print(f"  → {len(boundaries)} boundaries → {boundaries_json_path.name}")
        else:
            print("  → 0 sentences (not caching empty transcript)")

    if not sentences:
        print("Empty transcript; aborting.", file=sys.stderr)
        sys.exit(1)

    transcript_text = " ".join(s.text for s in sentences)

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
    known = load_known_words(KNOWN_PATH)
    prompted: set[str] = set()
    picked_most = pick_round("MOST occurring", counts, known, x, False, FRENCH_WORD_RE, KNOWN_PATH, prompted)
    picked_least = pick_round("LEAST occurring", counts, known, x, True, FRENCH_WORD_RE, KNOWN_PATH, prompted)
    new_words: dict[str, int] = {**picked_most, **picked_least}
    print(f"\n  → {len(new_words)} total new words")
    if not new_words:
        print("No new words picked; nothing to synthesize. Exiting.")
        sys.exit(0)

    # ── 5. Enrich (per-word + per-sentence translation) ─────────────────────
    print("\n[5/7] enrich (translate words + sentences)")
    words_list = list(new_words.keys())
    try:
        word_meanings = translate_batch(words_list, project_id, FRENCH.translate_source)
    except Exception as e:
        print(f"  (word translate failed: {e})", file=sys.stderr)
        word_meanings = {w: "" for w in words_list}

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
        sentence_translations_raw = translate_batch(
            all_sentence_texts, project_id, FRENCH.translate_source, batch_size=50
        )
    except Exception as e:
        print(f"  (sentence translate failed: {e})", file=sys.stderr)
        sentence_translations_raw = {}

    with open(vocab_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for w in words_list:
            writer.writerow([w, word_meanings.get(w, ""), new_words[w]])
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
            cfg=FRENCH,
        )
        if n % 5 == 0 or n == len(sentence_new_words):
            print(f"  expl [{n}/{len(sentence_new_words)}]")

    total_chunks = len(chunks)

    def _build_one_chunk(c: Chunk) -> tuple[int, AudioSegment]:
        announcement = render_tts(
            ssml_part_announcement(c.idx, total_chunks, FRENCH),
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
            cfg=FRENCH,
        )
        chunk_audio = announcement + chunk_body
        chunk_path = chunks_cache / f"chunk_{c.idx:02d}.mp3"
        chunk_audio.export(str(chunk_path), format="mp3", bitrate="192k")
        return c.idx, chunk_audio

    workers = max(1, min(args.workers, total_chunks))
    print(f"  → building {total_chunks} chunk(s) with {workers} worker(s)")
    built: dict[int, AudioSegment] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_build_one_chunk, c) for c in chunks]
        for n, future in enumerate(as_completed(futures), 1):
            idx, chunk_audio = future.result()
            built[idx] = chunk_audio
            print(f"  → chunk_{idx:02d}.mp3 ({len(chunk_audio) / 1000:.1f}s) [{n}/{total_chunks}]")

    final = AudioSegment.silent(duration=0)
    for i, c in enumerate(chunks):
        final += built[c.idx]
        if i != len(chunks) - 1:
            final += AudioSegment.silent(duration=INTER_CHUNK_BREAK_MS)

    final_path = outputs_dir / f"{stem}.mp3"
    final.export(
        str(final_path),
        format="mp3",
        bitrate="192k",
        tags={
            "title": stem,
            "artist": "LearnLangs Youtube Converter",
            "album": "LearnLangs French",
        },
    )
    print(f"\nDone! {len(final) / 1000:.1f}s → {final_path}")


if __name__ == "__main__":
    main()
