#!/usr/bin/env python3
"""
Collect a small paired dataset (audio + lyrics + optional transcripts) from a YouTube playlist.
- Downloads top N tracks as WAV with yt-dlp.
- Fetches lyrics from Genius using lyricsgenius.
- Optionally transcribes with Faster-Whisper for timestamps.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\-\s_]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    return value or "untitled"


class YDLLogger:
    def debug(self, msg):
        logging.getLogger("yt_dlp").debug(msg)

    def warning(self, msg):
        logging.getLogger("yt_dlp").warning(msg)

    def error(self, msg):
        logging.getLogger("yt_dlp").error(msg)


def download_playlist(
    playlist_url: str,
    output_dir: Path,
    limit: int,
) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "logger": YDLLogger(),
        "outtmpl": str(output_dir / "%(playlist_index)03d_%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": False,
        "playlistend": limit,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    entries: List[Dict[str, str]] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=True)
        if not info or not info.get("entries"):
            raise RuntimeError("No entries extracted; check the playlist URL.")

        for entry in info["entries"]:
            if not entry:
                continue
            idx = entry.get("playlist_index")
            title = entry.get("title") or "untitled"
            yt_id = entry.get("id") or "unknown"
            base = f"{int(idx):03d}_{slugify(title)}" if idx else slugify(title)

            guessed = output_dir / f"{int(idx):03d}_{yt_id}.wav" if idx else output_dir / f"{yt_id}.wav"
            if guessed.exists():
                target = output_dir / f"{base}.wav"
                guessed.rename(target)
                audio_path = target
            else:
                matches = list(output_dir.glob(f"{int(idx):03d}_{yt_id}.*")) if idx else []
                audio_path = matches[0] if matches else None

            entries.append(
                {
                    "title": title,
                    "id": yt_id,
                    "playlist_index": idx,
                    "base": base,
                    "audio_path": str(audio_path) if audio_path else None,
                    "webpage_url": entry.get("webpage_url"),
                }
            )

    return entries


def fetch_lyrics(
    entries: List[Dict[str, str]],
    output_dir: Path,
    artist_name: Optional[str],
    token: Optional[str],
    timeout: int,
    retries: int,
) -> None:
    """Fetch lyrics from Genius API using requests (raw HTTP)."""
    if not token or not requests:
        logging.info("Skipping lyrics fetch (missing GENIUS_ACCESS_TOKEN or requests library).")
        return

    genius_base = "https://api.genius.com"
    headers = {"Authorization": f"Bearer {token}"}

    for entry in entries:
        base = entry["base"]
        title = entry["title"]
        logging.info("Fetching lyrics for: %s", title)

        try:
            # Search for song
            params = {"q": f"{title} {artist_name or ''}"}
            resp = requests.get(
                f"{genius_base}/search",
                headers=headers,
                params=params,
                timeout=timeout,
            )
            resp.raise_for_status()
            hits = resp.json().get("response", {}).get("hits", [])

            if not hits:
                logging.warning("No lyrics found on Genius for: %s", title)
                continue

            # Get the song URL
            song_url = hits[0]["result"]["url"]
            logging.debug("Song URL: %s", song_url)

            # Fetch the page and parse lyrics (simple regex-based approach)
            page_resp = requests.get(song_url, timeout=timeout)
            page_resp.raise_for_status()
            html = page_resp.text

            # Extract text between <div> with data-lyrics-container
            import re
            match = re.search(r'<div[^>]*data-lyrics-container[^>]*>(.*?)</div>', html, re.DOTALL)
            if not match:
                logging.warning("Could not extract lyrics HTML for: %s", title)
                continue

            lyrics_html = match.group(1)
            # Replace <br> with newlines and strip HTML tags
            lyrics_text = re.sub(r'<br\s*/?>', '\n', lyrics_html)
            lyrics_text = re.sub(r'<[^>]+>', '', lyrics_text)
            lyrics_text = lyrics_text.strip()

            if not lyrics_text:
                logging.warning("Extracted empty lyrics for: %s", title)
                continue

            lyrics_path = output_dir / f"{base}.lyrics.txt"
            lyrics_path.write_text(lyrics_text, encoding="utf-8")
            entry["lyrics_path"] = str(lyrics_path)
            logging.info("Saved lyrics: %s", lyrics_path)

        except requests.RequestException as exc:  # pragma: no cover
            logging.warning("Lyrics fetch failed for %s: %s", title, exc)
        except Exception as exc:  # pragma: no cover
            logging.warning("Error processing lyrics for %s: %s", title, exc)


def transcribe_audios(
    entries: List[Dict[str, str]],
    model_name: str,
    device: str,
    beam_size: int,
    compute_type: str,
) -> None:
    if not WhisperModel:
        logging.info("Skipping transcription (faster-whisper not installed).")
        return

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    for entry in entries:
        audio_path = entry.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            logging.warning("Missing audio for transcription: %s", entry.get("title"))
            continue
        base = entry["base"]
        logging.info("Transcribing: %s", entry["title"])
        segments, info = model.transcribe(audio_path, beam_size=beam_size)

        text_lines: List[str] = []
        segments_out: List[Dict[str, float]] = []
        for seg in segments:
            text_lines.append(seg.text.strip())
            segments_out.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                }
            )

        txt_path = Path(audio_path).with_name(f"{base}.transcript.txt")
        json_path = Path(audio_path).with_name(f"{base}.transcript.json")
        txt_path.write_text("\n".join(text_lines), encoding="utf-8")
        json_path.write_text(json.dumps({"language": info.language, "segments": segments_out}, indent=2), encoding="utf-8")
        entry["transcript_txt"] = str(txt_path)
        entry["transcript_json"] = str(json_path)


def build_index(entries: List[Dict[str, str]], output_dir: Path) -> None:
    index_path = output_dir / "dataset_index.json"
    index_data = []
    for entry in entries:
        index_data.append(
            {
                "title": entry.get("title"),
                "id": entry.get("id"),
                "playlist_index": entry.get("playlist_index"),
                "audio_path": entry.get("audio_path"),
                "lyrics_path": entry.get("lyrics_path"),
                "transcript_txt": entry.get("transcript_txt"),
                "transcript_json": entry.get("transcript_json"),
                "webpage_url": entry.get("webpage_url"),
            }
        )
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    logging.info("Wrote index: %s", index_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect audio + lyrics + transcripts from a YouTube playlist.")
    parser.add_argument("--playlist-url", required=True, help="YouTube playlist URL")
    parser.add_argument("--output-dir", default="data/raw", help="Where to store outputs")
    parser.add_argument("--artist-name", default=None, help="Hint for lyrics search")
    parser.add_argument("--limit", type=int, default=10, help="Max tracks to fetch from playlist")
    parser.add_argument("--genius-token", default=os.getenv("GENIUS_ACCESS_TOKEN"), help="Genius API token (or set GENIUS_ACCESS_TOKEN)")
    parser.add_argument("--lyrics-timeout", type=int, default=15, help="Lyrics request timeout seconds")
    parser.add_argument("--lyrics-retries", type=int, default=3, help="Lyrics request retries")
    parser.add_argument("--transcribe", action="store_true", help="Enable Faster-Whisper transcription")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size/name")
    parser.add_argument("--whisper-device", default="cuda", help="Device for Whisper (cuda, cpu, mps)")
    parser.add_argument("--whisper-beam", type=int, default=5, help="Beam size for Whisper decoding")
    parser.add_argument("--whisper-compute-type", default="float16", help="faster-whisper compute type")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(args.output_dir)
    entries = download_playlist(args.playlist_url, output_dir, args.limit)

    fetch_lyrics(
        entries,
        output_dir,
        artist_name=args.artist_name,
        token=args.genius_token,
        timeout=args.lyrics_timeout,
        retries=args.lyrics_retries,
    )

    if args.transcribe:
        transcribe_audios(
            entries,
            model_name=args.whisper_model,
            device=args.whisper_device,
            beam_size=args.whisper_beam,
            compute_type=args.whisper_compute_type,
        )

    build_index(entries, output_dir)
    logging.info("Done. Tracks processed: %d", len(entries))


if __name__ == "__main__":
    main()
