#!/usr/bin/env python3
"""Apple Podcasts episode URL → downloaded MP3.

Drop-in replacement for :func:`common.ytcommon.download_youtube` used by the
per-language ``applepodcastconverter`` scripts: same ``(url, inputs_dir) -> Path``
contract, returning a path to an MP3 in ``inputs_dir``.

Resolution strategy (no scraping, no text search):

  1. Parse the episode URL into the *podcast collection* id (the ``id<digits>``
     path segment) and the *episode* id (the ``i`` query param).
  2. Query Apple's official iTunes Lookup API for the collection's episodes
     (``entity=podcastEpisode``) and match the episode by exact ``trackId``
     (== the ``i`` value). The matched object carries a direct CDN ``episodeUrl``.
  3. Download that direct URL with yt-dlp (handles redirects / UA / MP3 transcode).

The lookup only returns recent episodes (capped at ``limit=200``). When the
episode is not present (very old), fall back to yt-dlp's maintained
``ApplePodcasts`` extractor run on the page URL.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .ytcommon import sanitize_stem, ytdlp_download

ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"
LOOKUP_LIMIT = 200
_PODCAST_ID_RE = re.compile(r"/id(\d+)")


def _parse_apple_url(url: str) -> tuple[str, str]:
    """Extract (podcast_collection_id, episode_id) from an Apple Podcasts episode
    URL, e.g. ``.../podcast/<slug>/id1521247617?i=1000765888764`` → ("1521247617",
    "1000765888764"). Raises ValueError if either id is missing."""
    parsed = urlparse(url)
    m = _PODCAST_ID_RE.search(parsed.path)
    if not m:
        raise ValueError(
            f"Not an Apple Podcasts URL (no '/id<digits>' segment): {url!r}"
        )
    podcast_id = m.group(1)
    episode_id = (parse_qs(parsed.query).get("i") or [""])[0]
    if not episode_id:
        raise ValueError(
            "Apple Podcasts URL is missing the '?i=<episode id>' query parameter "
            f"(this looks like a show URL, not an episode URL): {url!r}"
        )
    return podcast_id, episode_id


def _lookup_episode(podcast_id: str, episode_id: str) -> dict | None:
    """Return the iTunes Lookup result for the episode whose trackId == episode_id,
    or None if it is not among the collection's recent episodes."""
    import requests

    resp = requests.get(
        ITUNES_LOOKUP_URL,
        params={"id": podcast_id, "entity": "podcastEpisode", "limit": LOOKUP_LIMIT},
        timeout=30,
    )
    resp.raise_for_status()
    target = int(episode_id)
    for r in resp.json().get("results", []):
        if r.get("wrapperType") == "podcastEpisode" and r.get("trackId") == target:
            return r
    return None


def download_apple_podcast(url: str, inputs_dir: Path) -> Path:
    """Download an Apple Podcasts episode's audio to ``inputs_dir`` and return the
    MP3 path (same contract as :func:`common.ytcommon.download_youtube`)."""
    podcast_id, episode_id = _parse_apple_url(url)
    print(f"  → iTunes lookup: podcast {podcast_id}, episode {episode_id}")
    episode = _lookup_episode(podcast_id, episode_id)

    audio_url = (episode or {}).get("episodeUrl")
    if episode and audio_url:
        title = sanitize_stem(episode.get("trackName") or f"episode-{episode_id}")
        print(f"  → episode: {title!r} → {audio_url}")
        return ytdlp_download(audio_url, inputs_dir, title=title)

    print("  → episode not found via iTunes API; falling back to yt-dlp ApplePodcasts extractor")
    return ytdlp_download(url, inputs_dir)
