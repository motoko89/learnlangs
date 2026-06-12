#!/usr/bin/env python3
"""Shared orchestration for the per-language YouTube → Vocab Listening-Practice
generators.

This module owns the entire CLI pipeline (argparse + the 6 stages: download,
transcribe, vocab, translate, chunk, synth/assemble). The per-language scripts
in src/<lang>/ytconverter/ytconverter.py are thin wrappers that build a
:class:`LangConfig` and call :func:`run_pipeline`; the only language-specific
behaviour is injected via the config and the optional ``word_postprocess`` hook
(e.g. Mandarin's OpenCC s2tw conversion). The language-agnostic library helpers
live in :mod:`common.ytcommon`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment

from .ytcommon import (
    INTER_CHUNK_BREAK_MS,
    INTRA_GROUP_BREAK_MS,
    Chunk,
    LangConfig,
    WordRec,
    assemble_chunk,
    build_explanation_clip,
    build_sentences,
    build_vocab_ssml_by_sentence,
    chunk_sentences,
    chunk_sentences_by_boundaries,
    download_youtube,
    ensure_dirs,
    extract_vocab,
    load_keys,
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


def run_pipeline(
    cfg: LangConfig,
    script_dir: Path,
    *,
    downloader: Callable[[str, Path], Path] = download_youtube,
    url_label: str = "YouTube URL",
    word_postprocess: Callable[[list[WordRec]], list[WordRec]] | None = None,
    description: str | None = None,
) -> None:
    """Run the full <source> → study-MP3 pipeline for one target language.

    `cfg` carries every language-specific setting (voices, language codes, vocab
    params, album tag). `script_dir` locates the per-language key.json /
    jumeau-gc.json defaults. `downloader` resolves the input `url` to a local MP3
    (default: YouTube via yt-dlp; the Apple Podcasts converter passes
    :func:`common.applepodcast.download_apple_podcast`), and `url_label` is the
    human label for that URL in --help / the prompt. `word_postprocess`, if given,
    maps the raw STT word records before sentence segmentation (e.g. Mandarin's
    OpenCC s2tw). `description` is shown in --help (the caller's module
    docstring)."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help=f"{url_label} (prompted if omitted)")
    parser.add_argument("--keys", default=str(script_dir / "key.json"))
    parser.add_argument("--gc", default=str(script_dir / "jumeau-gc.json"))
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
    parser.add_argument("--vocab-number", type=int, default=40, help="Number of vocab words/phrases for OpenAI to extract (default: 40)")
    args = parser.parse_args()

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
    ai_key = keys.get("cApi")
    if not ai_key:
        print("key.json must contain 'cApi' (OpenAI API key).", file=sys.stderr)
        sys.exit(1)
    ai_base_url = keys.get("cApiBaseUrl")
    if not ai_base_url:
        print("key.json must contain 'cApiBaseUrl' (Azure OpenAI base URL).", file=sys.stderr)
        sys.exit(1)

    gc_path = Path(args.gc).resolve()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gc_path)
    with open(gc_path, encoding="utf-8") as f:
        project_id = json.load(f)["project_id"]

    url = args.url or input(f"{url_label}: ").strip()
    if not url:
        print("No URL provided.", file=sys.stderr)
        sys.exit(1)

    # ── 1. Download ─────────────────────────────────────────────────────────
    print("\n[1/6] download")
    mp3_path = downloader(url, inputs_dir)
    stem = sanitize_stem(mp3_path.stem)
    print(f"  → {mp3_path}")

    inter_dir = intermediates_root / stem
    tts_cache = inter_dir / "tts"
    chunks_cache = inter_dir / "chunks"
    transcript_json_path = inter_dir / "transcript.json"
    transcript_srt_path = inter_dir / "transcript.srt"
    transcript_txt_path = inter_dir / "transcript.txt"
    boundaries_json_path = inter_dir / "chunk_boundaries.json"
    vocab_json_path = inter_dir / "vocab.json"
    vocab_tsv_path = inter_dir / "vocab.tsv"
    for stale in (tts_cache, chunks_cache):
        if stale.exists():
            shutil.rmtree(stale)
    ensure_dirs(inter_dir, tts_cache, chunks_cache)

    chunk_target_ms = int(args.chunk_minutes * 60 * 1000)

    # ── 2. Transcribe (cached) ──────────────────────────────────────────────
    print("\n[2/6] transcribe")
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
                raw_words = transcribe_mai(
                    flac_chunks, stt_endpoint, az_key, args.stt_model, cfg.mai_locale, args.workers,
                )
            else:
                min_speakers = max(1, args.min_speakers)
                max_speakers = args.max_speakers if args.max_speakers is not None else min_speakers
                max_speakers = max(min_speakers, max_speakers)
                speaker_split = min_speakers > 1
                print(f"  → diarization: {min_speakers}-{max_speakers} speaker(s)")
                timestamp = int(time.time())
                uploaded: list[tuple[str, int, object]] = []  # (uri, offset_ms, blob)
                for chunk_path, offset_ms, _end_ms in flac_chunks:
                    blob_name = f"stt-staging/{stem}-{timestamp}-{chunk_path.stem}.flac"
                    print(f"  → upload to gs://{gcs_bucket}/{blob_name} (+{offset_ms}ms)")
                    gcs_uri, blob = upload_to_gcs(chunk_path, gcs_bucket, blob_name)
                    uploaded.append((gcs_uri, offset_ms, blob))
                try:
                    raw_words = transcribe(
                        [(u, o) for u, o, _ in uploaded],
                        project_id,
                        cfg.chirp_location,
                        cfg.chirp_model,
                        cfg.language_code,
                        min_speakers,
                        max_speakers,
                    )
                finally:
                    for _, _, blob in uploaded:
                        try:
                            blob.delete()
                        except Exception as e:
                            print(f"  (warning: failed to delete staged blob: {e})", file=sys.stderr)
        print(f"  → {len(raw_words)} word records")
        words = word_postprocess(raw_words) if word_postprocess else raw_words
        sentences = build_sentences(words, cfg, split_on_speaker_change=speaker_split)
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

    transcript_text = cfg.word_joiner.join(s.text for s in sentences)

    # ── 3. Vocab extraction (OpenAI, cached) ────────────────────────────────
    print("\n[3/6] vocab (OpenAI)")
    vocab: list[dict] = []
    if vocab_json_path.exists() and vocab_json_path.stat().st_size > 0:
        try:
            vocab = json.loads(vocab_json_path.read_text(encoding="utf-8"))
            print(f"  cached: {vocab_json_path.name} ({len(vocab)} items)")
        except json.JSONDecodeError:
            vocab = []
    if not vocab:
        vocab = extract_vocab(
            transcript_text,
            ai_key,
            ai_base_url,
            native_voice=cfg.native_voice,
            break_ms=INTRA_GROUP_BREAK_MS,
            vocab_number=args.vocab_number,
            extra_field=cfg.vocab_extra_field,
            extra_explain=cfg.vocab_extra_explain,
        )
        vocab_json_path.write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  → {len(vocab)} vocab items → {vocab_json_path.name}")
    if not vocab:
        print("OpenAI returned no vocab items; nothing to synthesize. Exiting.")
        sys.exit(0)

    # vocab.tsv: text, [extra field], longExplain, shortExplain
    with open(vocab_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for v in vocab:
            row = [v.get("text", "")]
            if cfg.vocab_extra_field:
                row.append(v.get(cfg.vocab_extra_field, ""))
            row += [v.get("longExplain", ""), v.get("shortExplain", "")]
            writer.writerow(row)
    print(f"  → wrote {vocab_tsv_path.name}")

    # ── 4. Sentence translation (for playback pairs) ────────────────────────
    print("\n[4/6] translate sentences")
    vocab_by_sentence = build_vocab_ssml_by_sentence(vocab, sentences, cfg)
    print(f"  → {len(vocab_by_sentence)} sentences contain ≥1 vocab item")
    all_sentence_texts = list({s.text for s in sentences})
    print(f"  → translating {len(all_sentence_texts)} sentence(s)")
    try:
        sentence_translations_raw = translate_batch(
            all_sentence_texts, project_id, cfg.translate_source, batch_size=50
        )
    except Exception as e:
        print(f"  (sentence translate failed: {e})", file=sys.stderr)
        sentence_translations_raw = {}

    # ── 5. Load source audio + chunk ────────────────────────────────────────
    print("\n[5/6] chunk audio")
    print(f"  → loading {mp3_path.name}")
    original_audio = AudioSegment.from_mp3(str(mp3_path))
    if boundaries:
        chunks = chunk_sentences_by_boundaries(sentences, boundaries)
        print(f"  → {len(chunks)} chunks (reused from silence-aware split)")
    else:
        chunks = chunk_sentences(sentences, target_ms=chunk_target_ms)
        print(f"  → {len(chunks)} chunks (target {args.chunk_minutes:.1f} min each)")

    # ── 6. Build explanation clips, assemble chunks, concat ─────────────────
    print("\n[6/6] synth + assemble")
    sentence_index: dict[int, int] = {id(s): idx for idx, s in enumerate(sentences)}
    explanations: dict[int, AudioSegment] = {}
    for n, (idx, ssmls) in enumerate(vocab_by_sentence.items(), 1):
        s = sentences[idx]
        translation = sentence_translations_raw.get(s.text, "")
        explanations[idx] = build_explanation_clip(
            sentence=s,
            vocab_ssml_in_order=ssmls,
            sentence_translation=translation,
            original_audio=original_audio,
            tts_cache=tts_cache,
            az_key=az_key,
            az_region=az_region,
            cfg=cfg,
        )
        if n % 5 == 0 or n == len(vocab_by_sentence):
            print(f"  expl [{n}/{len(vocab_by_sentence)}]")

    total_chunks = len(chunks)

    def _build_one_chunk(c: Chunk) -> tuple[int, AudioSegment]:
        announcement = render_tts(
            ssml_part_announcement(c.idx, total_chunks, cfg),
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
            cfg=cfg,
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
            "album": cfg.album,
        },
    )
    print(f"\nDone! {len(final) / 1000:.1f}s → {final_path}")
