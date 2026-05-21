#!/usr/bin/env python3
"""Download a local cache of animated emoji GIFs for podcast video effects."""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin


DEFAULT_SOURCE_URL = "https://animatemojis.com/"
DEFAULT_ASSET_DIR = Path(__file__).with_name("assets") / "animated_emojis"
ENTRY_RE = re.compile(
    r'\{\\"slug\\":\\"(?P<slug>[^"]+)\\".*?'
    r'\\"name\\":\\"(?P<name>[^"]+)\\".*?'
    r'\\"category\\":\\"(?P<category>[^"]+)\\".*?'
    r'\\"keywords\\":\[(?P<keywords>.*?)\].*?'
    r'\\"formats\\":\{\\"webp\\":\\"(?P<webp>[^"]*)\\",\\"gif\\":\\"(?P<gif>[^"]*)\\"',
    re.S,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the emoji cache downloader."""

    parser = argparse.ArgumentParser(
        description="Cache animated emoji GIFs from AnimatEmojis for local video rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--limit", type=int, default=260, help="Maximum number of GIFs to download.")
    parser.add_argument("--sleep-sec", type=float, default=0.02, help="Short delay between downloads.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise download progress logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def fetch_text(url: str) -> str:
    """Fetch a UTF-8 web page through curl to avoid local Python CA issues."""

    result = subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", url],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def parse_emoji_entries(page_html: str, source_url: str) -> list[dict[str, object]]:
    """Extract downloadable animated emoji metadata from the AnimatEmojis page."""

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for match in ENTRY_RE.finditer(page_html):
        gif_url = html.unescape(match.group("gif"))
        if not gif_url:
            continue
        slug = html.unescape(match.group("slug"))
        if slug in seen:
            continue
        seen.add(slug)
        keywords = [
            html.unescape(item.strip().strip('\\"'))
            for item in match.group("keywords").split(",")
            if item.strip()
        ]
        entries.append(
            {
                "slug": slug,
                "name": html.unescape(match.group("name")),
                "category": html.unescape(match.group("category")),
                "keywords": keywords,
                "gif_url": urljoin(source_url, gif_url),
                "webp_url": urljoin(source_url, html.unescape(match.group("webp"))),
                "file": f"{slug}.gif",
            }
        )
    return entries


def download_bytes(url: str) -> bytes:
    """Download one binary asset through curl to match manual fetch behavior."""

    result = subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", url],
        check=True,
        capture_output=True,
    )
    return result.stdout


def download_entries(entries: list[dict[str, object]], asset_dir: Path, limit: int, overwrite: bool, sleep_sec: float) -> list[dict[str, object]]:
    """Download selected emoji GIFs and return manifest entries for successful files."""

    asset_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, object]] = []
    for index, entry in enumerate(entries[:limit], start=1):
        target = asset_dir / str(entry["file"])
        if target.exists() and not overwrite:
            entry["bytes"] = target.stat().st_size
            downloaded.append(entry)
            continue
        try:
            payload = download_bytes(str(entry["gif_url"]))
        except Exception as exc:  # noqa: BLE001 - keep cache script resilient for flaky sources.
            logging.warning("Skip %s: %s", entry["slug"], exc)
            continue
        target.write_bytes(payload)
        entry["bytes"] = len(payload)
        downloaded.append(entry)
        logging.info("Downloaded %s/%s %s", index, min(limit, len(entries)), target.name)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return downloaded


def write_manifest(asset_dir: Path, entries: list[dict[str, object]], source_url: str) -> None:
    """Write a compact manifest so the cache can be searched offline."""

    payload = {
        "source_url": source_url,
        "count": len(entries),
        "entries": entries,
    }
    (asset_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Download animated emojis into the local podcast asset directory."""

    configure_logging()
    args = parse_args()
    page_html = fetch_text(args.source_url)
    entries = parse_emoji_entries(page_html, args.source_url)
    logging.info("Found %s downloadable GIF entries", len(entries))
    downloaded = download_entries(entries, args.asset_dir, args.limit, args.overwrite, args.sleep_sec)
    write_manifest(args.asset_dir, downloaded, args.source_url)
    logging.info("Cached %s GIFs in %s", len(downloaded), args.asset_dir)


if __name__ == "__main__":
    main()
