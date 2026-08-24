#!/usr/bin/env python3
"""Per-language store of vocabulary the learner already knows.

python3 ./vocabstore.py import --lang mandarin /Users/hunghho/Documents/Anki-TWMandarin.txt --dry-run

The ytconverter / applepodcastconverter pipeline extracts N lexical units per
episode and writes them to ``vocab.tsv`` for import into Anki. Without memory
across episodes an ever-growing share of those N are words the learner already
has, wasting card slots and padding the study MP3 with re-explanations. This
module holds that memory: one ``known_vocab.tsv`` per language, seeded from an
Anki export and rolled forward automatically by each pipeline run.

The store is a TSV with the header ``key<TAB>text<TAB>added<TAB>source``:

    key     normalized comparison key (see :func:`normalize_key`; tab-free)
    text    the item as written, whitespace-collapsed
    added   local ISO-8601 timestamp
    source  "anki:<basename>" for imports, the episode stem for pipeline runs

``source`` carrying the episode stem is what makes ``forget <stem>`` possible,
so it must never be a generic label. There is no ``lang`` column — the file is
already per-language.

Deliberately stdlib-only with no intra-package imports, so this runs as a script
from the repo root (``python3 src/common/vocabstore.py import --lang french …``)
*and* imports cleanly as ``from .vocabstore import …`` from the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable, NamedTuple

try:
    import fcntl
except ImportError:  # non-POSIX; fall back to an unlocked append
    fcntl = None  # type: ignore[assignment]


STORE_FILENAME = "known_vocab.tsv"
STORE_HEADER = ("key", "text", "added", "source")


class KnownRow(NamedTuple):
    key: str
    text: str
    added: str
    source: str


# ─── Normalization ────────────────────────────────────────────────────────────

# Every apostrophe-ish and backtick-ish codepoint folds to ASCII "'" before the
# punctuation pass, so a curly-quoted export and a straight-quoted transcript
# reach the same key.
_QUOTES = {ord(c): "'" for c in "‘’ʼ′´`"}


def _is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def normalize_key(text: str) -> str:
    """Fold a vocab string to its store comparison key.

    NFKC (composes accents, folds full-width forms) → curly quotes to ASCII →
    every punctuation/symbol/separator/control character to a space → collapse
    whitespace → casefold → drop spaces sitting between two East-Asian wide
    characters.

    Punctuation becomes a space rather than being deleted: "l'ordre" must agree
    with "l ordre", and deleting the apostrophe would give "lordre" instead.
    The CJK pass then undoes that for scripts written without spaces, so
    "你好，世界" and "你好世界" agree.

    Diacritics are deliberately preserved: folding them would merge "año"/"ano"
    and "papá"/"papa", and a false match silently suppresses a genuinely new
    word — invisible and unrecoverable, where a missed match only costs one
    duplicate card.

    Because ``str.split`` is used, a key can never contain a tab or newline,
    which is what makes the TSV store safe to parse by splitting on tabs.
    """
    s = unicodedata.normalize("NFKC", text or "")
    s = s.translate(_QUOTES)
    s = "".join(" " if unicodedata.category(ch)[0] in "PSZC" else ch for ch in s)
    s = " ".join(s.split()).casefold()
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch == " " and out and _is_wide(out[-1]) and i + 1 < len(s) and _is_wide(s[i + 1]):
            continue
        out.append(ch)
    return "".join(out)


def _collapse(text: str) -> str:
    """Whitespace-collapse a value so it is safe in a TSV column."""
    return " ".join((text or "").split())


def _item_text(item) -> str:
    """Accept either a raw string or a vocab dict as produced by extract_vocab."""
    if isinstance(item, dict):
        return str(item.get("text") or "")
    return str(item or "")


# ─── Store paths ──────────────────────────────────────────────────────────────

def default_store_path(lang_dir: Path) -> Path:
    """The store for a language, given its ``src/<lang>`` directory.

    Anchored to the language directory rather than the cwd so ytconverter and
    applepodcastconverter share one store per language, matching how the
    FRENCH / SPANISH / MANDARIN LangConfigs are already shared."""
    return Path(lang_dir) / STORE_FILENAME


def store_for_lang(lang: str) -> Path:
    return default_store_path(Path(__file__).resolve().parents[1] / lang)


def known_languages() -> list[str]:
    src = Path(__file__).resolve().parents[1]
    return sorted(
        p.name for p in src.iterdir()
        if p.is_dir() and p.name != "common" and (p / "common" / "langconfig.py").exists()
    )


# ─── Read / write ─────────────────────────────────────────────────────────────

def load_known(path: Path | str) -> dict[str, KnownRow]:
    """Read the store into {key: row}, first-wins on duplicate keys.

    Tolerates hand editing: blank lines, ``#`` comments and the header row are
    skipped, and a line with no tabs is read as a bare vocab item whose key is
    derived on the fly. A missing file is an empty store, not an error."""
    out: dict[str, KnownRow] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "key" and len(parts) > 1 and parts[1] == "text":
                continue
            if len(parts) == 1:
                text = parts[0].strip()
                row = KnownRow(normalize_key(text), text, "", "manual")
            else:
                key = parts[0].strip() or normalize_key(parts[1])
                row = KnownRow(
                    key,
                    parts[1],
                    parts[2] if len(parts) > 2 else "",
                    parts[3] if len(parts) > 3 else "",
                )
            if not row.key or row.key in out:
                continue
            out[row.key] = row
    return out


def _lock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)


def append_known(path: Path | str, items: Iterable, source: str) -> tuple[int, int]:
    """Append the items not already present. Returns (n_added, n_total).

    Opens O_APPEND under an exclusive flock and re-reads the existing keys while
    holding it, so two pipeline runs of the same language in parallel both
    survive. Note this deliberately does *not* use the mkstemp + os.replace
    pattern from ytcommon.render_tts: that is right for an immutable
    one-blob-per-SHA cache and wrong for a shared append-only log, where a
    whole-file replace would clobber rows another process appended after the
    read. Dedup by key makes re-recording the same episode a no-op."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    texts = [_item_text(i) for i in items]
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    fd = os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        _lock(fd)
        existing = load_known(p)
        lines: list[str] = []
        fresh: set[str] = set()
        for text in texts:
            key = normalize_key(text)
            if not key or key in existing or key in fresh:
                continue
            fresh.add(key)
            lines.append("\t".join((key, _collapse(text), now, _collapse(source))))
        if lines and os.fstat(fd).st_size == 0:
            lines.insert(0, "\t".join(STORE_HEADER))
        if lines:
            os.write(fd, ("\n".join(lines) + "\n").encode("utf-8"))
        return len(fresh), len(existing) + len(fresh)
    finally:
        os.close(fd)


def rewrite_known(path: Path | str, rows: Iterable[KnownRow]) -> None:
    """Replace the whole store atomically. For maintenance commands only —
    append_known is the hot path and must stay append-only."""
    p = Path(path)
    tmp = p.with_name(f".{p.name}.tmp")
    body = ["\t".join(STORE_HEADER)]
    body += ["\t".join((r.key, _collapse(r.text), r.added, r.source)) for r in rows]
    tmp.write_text("\n".join(body) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ─── Anki export / word-list parsing ──────────────────────────────────────────

_SEP_NAMES = {
    "tab": "\t", "comma": ",", "semicolon": ";",
    "space": " ", "pipe": "|", "colon": ":",
}
_META_COLUMN_KEYS = ("guid column", "notetype column", "deck column", "tags column")

_SOUND_RE = re.compile(r"\[sound:[^\]]*\]")
_CLOZE_RE = re.compile(r"\{\{c\d+::(.*?)(?:::[^}]*)?\}\}", re.S)
_BREAKISH_RE = re.compile(r"(?i)<\s*(?:br|/div|/p|/li|/tr|/td)\s*/?\s*>")
_TAG_RE = re.compile(r"<[^>]+>")


def clean_field(value: str) -> str:
    """Strip the markup Anki puts inside fields when exported with #html:true."""
    v = _SOUND_RE.sub(" ", value or "")
    v = _CLOZE_RE.sub(r"\1", v)
    v = _BREAKISH_RE.sub(" ", v)
    v = _TAG_RE.sub("", v)
    v = html.unescape(v)
    v = v.replace("\u00a0", " ")
    return " ".join(v.split())


def _read_meta(path: Path) -> tuple[dict[str, str], int]:
    """Parse the leading contiguous block of ``#key:value`` lines.

    Only the leading block — a note field can legitimately start with '#', and
    Anki writes its header solely at the top."""
    meta: dict[str, str] = {}
    n = 0
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("#"):
                break
            n += 1
            key, sep, val = line[1:].strip().partition(":")
            if sep:
                meta[key.strip().lower()] = val.strip()
    return meta, n


def parse_export(path: Path, field: int | None = None) -> tuple[list[str], dict]:
    """Read an Anki "Notes in Plain Text" export or a one-item-per-line list.

    Returns (texts, info) where info describes what was detected, for --dry-run.
    """
    meta, n_meta = _read_meta(path)

    delimiter: str | None = None
    raw_sep = meta.get("separator")
    if raw_sep:
        cand = _SEP_NAMES.get(raw_sep.lower(), raw_sep)
        delimiter = cand if len(cand) == 1 else "\t"

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    data = lines[n_meta:]
    if delimiter is None:
        first = next((ln for ln in data if ln.strip() and not ln.startswith("#")), "")
        delimiter = "\t" if "\t" in first else None

    claimed: set[int] = set()
    for key in _META_COLUMN_KEYS:
        if key in meta:
            try:
                claimed.add(int(meta[key]))
            except ValueError:
                pass

    if delimiter is None:
        # Plain word list: the learntwords.txt / hsk1to4_zh-TW.txt convention.
        texts = [
            ln.strip() for ln in data
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        info = {
            "format": "plain word list", "separator": None,
            "column": 1, "claimed": sorted(claimed), "rows": len(texts),
        }
        return [t for t in (clean_field(t) for t in texts) if t], info

    if field:
        col = field - 1
    else:
        col = 1
        while col in claimed:
            col += 1
        col -= 1

    texts: list[str] = []
    for row in csv.reader(data, delimiter=delimiter):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if col >= len(row):
            continue
        cleaned = clean_field(row[col])
        if cleaned:
            texts.append(cleaned)
    info = {
        "format": "delimited export", "separator": delimiter,
        "column": col + 1, "claimed": sorted(claimed), "rows": len(texts),
    }
    return texts, info


def _opencc_convert(texts: list[str], config: str) -> list[str]:
    from opencc import OpenCC

    converter = OpenCC(config)
    return [converter.convert(t) for t in texts]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_store(args) -> Path:
    return Path(args.store) if args.store else store_for_lang(args.lang)


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", choices=known_languages(), help="resolve src/<lang>/known_vocab.tsv")
    group.add_argument("--store", help="explicit path to a known_vocab.tsv")


def cmd_import(args) -> int:
    store = _resolve_store(args)
    src = Path(args.file)
    if not src.exists():
        print(f"No such file: {src}", file=sys.stderr)
        return 1

    texts, info = parse_export(src, field=args.field)
    if args.opencc:
        texts = _opencc_convert(texts, args.opencc)

    sep_label = {"\t": "tab", None: "(none)"}.get(info["separator"], repr(info["separator"]))
    print(f"  format:    {info['format']}")
    print(f"  separator: {sep_label}")
    print(f"  column:    {info['column']}" + (" (--field override)" if args.field else " (auto)"))
    if info["claimed"]:
        print(f"  skipped meta columns: {info['claimed']}")
    if args.opencc:
        print(f"  opencc:    {args.opencc}")
    print(f"  parsed:    {len(texts)} row(s)")

    known = load_known(store)
    preview = texts[:10]
    print(f"  first {len(preview)} parsed:")
    for t in preview:
        key = normalize_key(t)
        mark = "KNOWN" if key in known else "new"
        print(f"    [{mark:>5}] {key!r} ← {t!r}")

    if args.dry_run:
        would = len({normalize_key(t) for t in texts if normalize_key(t)} - set(known))
        print(f"\n  dry run: would add {would} new item(s) to {store} ({len(known)} present)")
        return 0

    added, total = append_known(store, texts, source=f"anki:{src.name}")
    print(f"\n  → added {added} new item(s) → {store} ({total} total)")
    return 0


def cmd_check(args) -> int:
    store = _resolve_store(args)
    target = Path(args.file)
    if not target.exists():
        print(f"No such file: {target}", file=sys.stderr)
        return 1
    known = load_known(store)
    n_known = 0
    n_total = 0
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        text = line.split("\t")[0].strip()
        if not text:
            continue
        n_total += 1
        row = known.get(normalize_key(text))
        if row:
            n_known += 1
            origin = f"  ({row.source})" if row.source else ""
            print(f"  KNOWN  {text}{origin}")
        else:
            print(f"  new    {text}")
    print(f"\n  {n_known}/{n_total} already known — store {store} ({len(known)} items)")
    return 0


def cmd_forget(args) -> int:
    store = _resolve_store(args)
    rows = list(load_known(store).values())
    keep = [r for r in rows if r.source != args.source]
    dropped = len(rows) - len(keep)
    if not dropped:
        sources = sorted({r.source for r in rows if r.source})
        print(f"  no rows with source {args.source!r} in {store}")
        print(f"  known sources: {', '.join(sources) if sources else '(none)'}")
        return 1
    if args.dry_run:
        print(f"  dry run: would drop {dropped} row(s) with source {args.source!r}")
        return 0
    rewrite_known(store, keep)
    print(f"  → dropped {dropped} row(s) with source {args.source!r} → {store} ({len(keep)} remain)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vocabstore",
        description="Manage the per-language store of already-known vocabulary.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_import = sub.add_parser("import", help="seed/merge from an Anki export or a word list")
    _add_store_args(p_import)
    p_import.add_argument("file", help="Anki 'Notes in Plain Text' export, or one item per line")
    p_import.add_argument("--field", type=int, help="1-based column holding the vocab (default: auto-detect)")
    p_import.add_argument("--opencc", help="convert via OpenCC on the way in, e.g. s2tw")
    p_import.add_argument("--dry-run", action="store_true", help="report what would be imported and stop")
    p_import.set_defaults(func=cmd_import)

    p_check = sub.add_parser("check", help="report which rows of a vocab.tsv are already known")
    _add_store_args(p_check)
    p_check.add_argument("file", help="a vocab.tsv (first column is the vocab text)")
    p_check.set_defaults(func=cmd_check)

    p_forget = sub.add_parser("forget", help="drop the rows recorded by one episode")
    _add_store_args(p_forget)
    p_forget.add_argument("source", help="episode stem as recorded in the source column")
    p_forget.add_argument("--dry-run", action="store_true", help="report what would be dropped and stop")
    p_forget.set_defaults(func=cmd_forget)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
