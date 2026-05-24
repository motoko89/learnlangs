#!/usr/bin/env python3
"""
Mandarin YT Podcast Vocabulary Extractor

Takes an MP3 path, transcribes it with Google Cloud Speech-to-Text
(cmn-Hans-CN — the only Mandarin variant that supports diarization on Chirp),
converts the Simplified output to Traditional (zh-TW) with OpenCC,
segments with jieba, identifies important vocabulary (frequent / rare / proper
nouns / NER entities), enriches with pinyin (Azure) + English (Google Translate),
and writes a TSV next to the MP3.

Dependencies:
  pip install -r requirements.txt
  brew install ffmpeg   # macOS — pydub MP3 decode

Credentials (reused from adhocscripts/):
  key.json       - {"key": "<Google API key>", "azDictKey": "<Azure key>"}
  jumeau-gc.json - Google Cloud service account JSON

Usage:
  python3 extract_vocab.py mp3/episode.mp3 --gcs-bucket jumeau-stt-staging
  python3 extract_vocab.py mp3/MCP-003-SufferDepressionChina.mp3 --gcs-bucket kzsadhoclanguagelearning 
  python3 extract_vocab.py mp3/episode.mp3 --gcs-bucket jumeau-stt-staging --min-count 5
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import requests
from pydub import AudioSegment

LANGUAGE_CODE = "cmn-Hans-CN"  # only Hans-CN supports diarization on Chirp; output is post-converted to Hant-TW
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
CHIRP_LOCATION = "us"

PROPER_NOUN_POS = {"nr", "ns", "nt", "nz", "nrfg", "nrt"}
SKIP_POS = {"u", "uj", "ul", "ud", "uv", "uz", "ug", "p", "c", "y", "e", "o", "x", "w", "m"}
KEEP_ENTITY_TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "EVENT", "WORK_OF_ART"}

PUNCT_OR_DIGIT_RE = re.compile(r"^[\W\d_]+$", re.UNICODE)
HAN_ONLY_RE = re.compile(r"^[㐀-䶿一-鿿豈-﫿]+$")


def load_keys(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def convert_mp3_to_flac(mp3_path: Path, out_path: Path) -> None:
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)
    audio.export(str(out_path), format="flac")


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str):
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}", client, blob


def transcribe(gcs_uri: str, project_id: str) -> tuple[str, str]:
    """Transcribe GCS audio with Chirp (Speech v2) + diarization.

    Returns (plain_transcript, speaker_labeled_transcript).
    """
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
        model="chirp_3",  # Chirp 3 supports diarization for cmn-Hans-CN via BatchRecognize
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=2,
                max_speaker_count=2,
            ),
        ),
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer,
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    print("  → STT v2 (chirp_3) BatchRecognize submitted; waiting (may take several minutes)...")
    operation = client.batch_recognize(request=request)
    response = operation.result(timeout=3600)

    plain_parts = []
    speaker_parts = []
    for file_result in response.results.values():
        for result in file_result.transcript.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            plain_parts.append(alt.transcript.strip())
            if alt.words and alt.words[0].speaker_label:
                seg_words = []
                current_speaker = alt.words[0].speaker_label
                for word_info in alt.words:
                    if word_info.speaker_label != current_speaker:
                        speaker_parts.append(f"[{current_speaker}] {''.join(seg_words)}")
                        seg_words = []
                        current_speaker = word_info.speaker_label
                    seg_words.append(word_info.word)
                if seg_words:
                    speaker_parts.append(f"[{current_speaker}] {''.join(seg_words)}")
            else:
                speaker_parts.append(alt.transcript.strip())

    return " ".join(plain_parts), "\n".join(speaker_parts)


_S2T_CONVERTER = None


def s2t(text: str) -> str:
    """Simplified (cmn-Hans-CN STT output) → Traditional (zh-TW) via OpenCC."""
    global _S2T_CONVERTER
    if _S2T_CONVERTER is None:
        from opencc import OpenCC

        _S2T_CONVERTER = OpenCC("s2tw")
    return _S2T_CONVERTER.convert(text)


def tokenize(text: str):
    """Returns list of (word, pos). Filters punctuation, digits, particles, most 1-char tokens."""
    import jieba.posseg as pseg

    out = []
    for w, pos in pseg.cut(text, HMM=True):
        w = w.strip()
        if not w or PUNCT_OR_DIGIT_RE.match(w):
            continue
        if pos in SKIP_POS:
            continue
        # Drop most 1-char words; keep 1-char proper nouns (some surnames/places are 1 char)
        if len(w) < 2 and pos not in PROPER_NOUN_POS:
            continue
        out.append((w, pos))
    return out


def load_hsk(path: Path) -> set:
    if not path.exists():
        print(f"  (HSK list not found at {path.name}; 'rare' flagging disabled)")
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def extract_entities(text: str) -> dict:
    """{word -> set of 'entity:TYPE' strings} via Google NL API."""
    from google.cloud import language_v1

    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language="zh",
    )
    response = client.analyze_entities(request={"document": document})
    out = {}
    for entity in response.entities:
        type_name = language_v1.Entity.Type(entity.type_).name
        if type_name not in KEEP_ENTITY_TYPES:
            continue
        out.setdefault(entity.name, set()).add(f"entity:{type_name}")
        for mention in entity.mentions:
            out.setdefault(mention.text.content, set()).add(f"entity:{type_name}")
    return {k: sorted(v) for k, v in out.items()}


def transliterate(text: str, az_key: str) -> str:
    """zh-Hant → Pinyin via Azure Cognitive Services Translator."""
    url = (
        "https://api.cognitive.microsofttranslator.com/transliterate"
        "?api-version=3.0&language=zh-Hant&fromScript=Hant&toScript=Latn"
    )
    resp = requests.post(
        url,
        headers={"Ocp-Apim-Subscription-Key": az_key, "Content-Type": "application/json"},
        json=[{"Text": text}],
    )
    resp.raise_for_status()
    result = resp.json()
    if result and "text" in result[0]:
        return result[0]["text"]
    return ""


def translate_batch(words: list, translate_url: str, batch_size: int = 100) -> dict:
    """Translate a list of words to English in batches. Returns {word: english}."""
    out = {}
    for i in range(0, len(words), batch_size):
        chunk = words[i : i + batch_size]
        resp = requests.post(
            translate_url,
            json={"q": chunk, "target": "en", "source": "zh-TW", "format": "text"},
        )
        resp.raise_for_status()
        translations = resp.json().get("data", {}).get("translations", [])
        for word, t in zip(chunk, translations):
            out[word] = t.get("translatedText", "")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract important Mandarin vocabulary from a YT podcast MP3."
    )
    parser.add_argument("mp3_path", help="Path to the MP3 file")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket for STT staging (e.g. jumeau-stt-staging)")
    parser.add_argument("--top", type=int, default=50, help="Cap on rows in output TSV (default: 50; 0 = no cap)")
    parser.add_argument(
        "--keys",
        default=str(Path(__file__).parent.parent / "key.json"),
        help="Path to key.json (default: ../key.json)",
    )
    parser.add_argument(
        "--gc",
        default=str(Path(__file__).parent.parent / "jumeau-gc.json"),
        help="Path to Google Cloud service account JSON (default: ../jumeau-gc.json)",
    )
    parser.add_argument(
        "--hsk",
        default=str(Path(__file__).parent / "hsk1to4_zh-TW.txt"),
        help="Path to HSK 1-4 word list, one traditional word per line",
    )
    parser.add_argument("--output", help="Output TSV path (default: <mp3>.vocab.tsv)")
    parser.add_argument("--skip-ner", action="store_true", help="Skip Google Natural Language API entity pass")
    parser.add_argument("--least-occur", action="store_true", help="Iterate words from least to most occurrences (default is most to least)")
    args = parser.parse_args()

    mp3_path = Path(args.mp3_path).resolve()
    if not mp3_path.exists():
        print(f"MP3 not found: {mp3_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else mp3_path.with_suffix(".vocab.tsv")
    transcript_path = mp3_path.with_suffix(".transcript.txt")
    speaker_transcript_path = mp3_path.with_suffix(".speakers.txt")

    keys = load_keys(Path(args.keys))
    az_key = keys["azDictKey"]
    translate_url = "https://translation.googleapis.com/language/translate/v2?key=" + keys["key"]
    gc_path = Path(args.gc).resolve()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gc_path)
    with open(gc_path, encoding="utf-8") as f:
        project_id = json.load(f)["project_id"]

    # ---- 1. Transcribe (cached) ---------------------------------------------
    if transcript_path.exists() and transcript_path.stat().st_size > 0:
        print(f"[1/5] transcript cached: {transcript_path.name}")
        transcript = transcript_path.read_text(encoding="utf-8")
    else:
        print(f"[1/5] transcribe: {mp3_path.name}")
        with tempfile.TemporaryDirectory() as td:
            flac_path = Path(td) / (mp3_path.stem + ".flac")
            print(f"  → MP3 → FLAC ({TARGET_SAMPLE_RATE}Hz mono)")
            convert_mp3_to_flac(mp3_path, flac_path)
            blob_name = f"stt-staging/{mp3_path.stem}-{int(time.time())}.flac"
            print(f"  → upload to gs://{args.gcs_bucket}/{blob_name}")
            gcs_uri, gcs_client, blob = upload_to_gcs(flac_path, args.gcs_bucket, blob_name)
            try:
                transcript, speaker_transcript = transcribe(gcs_uri, project_id)
            finally:
                try:
                    blob.delete()
                except Exception as e:
                    print(f"  (warning: failed to delete staged blob: {e})", file=sys.stderr)
        print("  → OpenCC s2tw: Hans-CN → Hant-TW")
        transcript = s2t(transcript)
        speaker_transcript = s2t(speaker_transcript)
        transcript_path.write_text(transcript, encoding="utf-8")
        speaker_transcript_path.write_text(speaker_transcript, encoding="utf-8")
        print(f"  → wrote {transcript_path.name} ({len(transcript)} chars)")
        print(f"  → wrote {speaker_transcript_path.name}")

    if not transcript.strip():
        print("Empty transcript; nothing to extract.", file=sys.stderr)
        sys.exit(1)

    # ---- 2. Tokenize --------------------------------------------------------
    print("[2/5] tokenize")
    tokens = tokenize(transcript)
    counts = Counter(w for w, _ in tokens)
    pos_for = {}
    for w, pos in tokens:
        pos_for.setdefault(w, pos)
    print(f"  → {sum(counts.values())} kept tokens, {len(counts)} unique")

    # ---- 3. Score importance ------------------------------------------------
    print("[3/5] score importance")
    hsk = load_hsk(Path(args.hsk))

    entities = {}
    if args.skip_ner:
        print("  (--skip-ner)")
    elif len(transcript) > 100_000:
        print(f"  (transcript {len(transcript)} chars > 100k; skipping NL API)")
    else:
        try:
            entities = extract_entities(transcript)
            print(f"  → Google NL API: {len(entities)} distinct entity surface forms")
        except Exception as e:
            print(f"  (NL API failed: {e})", file=sys.stderr)

    important = {}
    hsk_path = Path(args.hsk)
    if hsk_path.exists():
        data = hsk_path.read_bytes()
        if data and not data.endswith(b"\n"):
            with open(hsk_path, "ab") as _f:
                _f.write(b"\n")
    with open(hsk_path, "a", encoding="utf-8") as hsk_file:
        count_sign = 1 if args.least_occur else -1
        for word, count in sorted(counts.items(), key=lambda x: (count_sign * x[1], -len(x[0]), x[0])):
            if hsk and word in hsk:
                continue
            if not HAN_ONLY_RE.match(word):
                continue
            ans = input(f"  New word: {word!r} (count={count}) — Enter=accept, any char+Enter=reject: ")
            if ans:
                hsk.add(word)
                hsk_file.write(word + "\n")
                hsk_file.flush()
                continue
            cats = []
            if pos_for.get(word) in PROPER_NOUN_POS:
                cats.append("proper_noun")
            if word in entities:
                cats.extend(entities[word])
            if not cats:
                cats.append("user_accepted")
            important[word] = (count, cats)
            if args.top and args.top > 0 and len(important) == args.top:
                break

    sorted_words = list(important.items())
    print(f"  → kept {len(sorted_words)} important words")

    if not sorted_words:
        print("No important vocabulary found. Try lowering --min-count.", file=sys.stderr)
        sys.exit(0)

    # ---- 4. Enrich ----------------------------------------------------------
    print("[4/5] enrich (pinyin + English)")
    words_only = [w for w, _ in sorted_words]
    try:
        translations = translate_batch(words_only, translate_url)
    except Exception as e:
        print(f"  (translate batch failed: {e}; rows will have empty english)", file=sys.stderr)
        translations = {}

    pinyin_map = {}
    for i, w in enumerate(words_only, 1):
        try:
            pinyin_map[w] = transliterate(w, az_key)
        except Exception as e:
            print(f"    pinyin failed for {w!r}: {e}", file=sys.stderr)
            pinyin_map[w] = ""
        if i % 25 == 0:
            print(f"    pinyin [{i}/{len(words_only)}]")

    # ---- 5. Write TSV -------------------------------------------------------
    print(f"[5/5] write TSV: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for word, (count, cats) in sorted_words:
            writer.writerow([
                word,
                pinyin_map.get(word, ""),
                translations.get(word, ""),
                ";".join(cats),
                count,
            ])

    print(f"\nDone! {len(sorted_words)} rows → {output_path}")
    print(f"      transcript → {transcript_path}")


if __name__ == "__main__":
    main()
