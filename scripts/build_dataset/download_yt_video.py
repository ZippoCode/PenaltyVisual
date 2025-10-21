import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import yt_dlp
from yt_dlp.utils import DownloadError

from penalty_vision.utils.logger import logger


class PenaltyVideoDownloader:

    def __init__(self, output_dir: str = "dataset/raw_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.downloaded_file = self.output_dir / "downloaded_urls.json"
        self.downloaded_urls = self._load_downloaded()
        self.video_count = self._get_next_video_number()

    def _load_downloaded(self) -> Dict:
        if self.downloaded_file.exists():
            with open(self.downloaded_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_downloaded(self):
        with open(self.downloaded_file, 'w') as f:
            json.dump(self.downloaded_urls, f, indent=2)

    def _get_next_video_number(self) -> int:
        existing = list(self.output_dir.glob("penalty_*.mp4"))
        if not existing:
            return 1

        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split('_')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue

        return max(numbers) + 1 if numbers else 1

    def _progress_hook(self, d):
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            eta = d.get('_eta_str', 'N/A')
            logger.info(f"  Downloading: {percent} at {speed} ETA: {eta}")
        elif d['status'] == 'finished':
            logger.info(f"  Download completed, processing...")

    def _get_ydl_opts(self, output_path: Path) -> Dict:
        return {
            'format': 'bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height>=720][ext=mp4]/best',
            'format_sort': ['res:720', 'fps:30'],
            'outtmpl': str(output_path),
            'restrictfilenames': True,
            'max_filesize': 100 * 1024 * 1024,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'writeinfojson': True,
            'writethumbnail': True,
            'ratelimit': 2 * 1024 * 1024,
            'sleep_interval': 2,
            'max_sleep_interval': 5,
            'retries': 3,
            'fragment_retries': 3,
            'noplaylist': True,
            'quiet': False,
            'no_warnings': False,
            'progress_hooks': [self._progress_hook],
            'ignoreerrors': False,
            'nocheckcertificate': False,
        }

    def _validate_video(self, info_dict: Dict) -> tuple[bool, str]:
        duration = info_dict.get('duration', 0)
        height = info_dict.get('height', 0)

        if duration < 3:
            return False, f"too_short ({duration}s)"

        if height < 720:
            return False, f"low_quality ({height}p)"

        return True, "ok"

    def download_video(self, url: str) -> bool:
        if url in self.downloaded_urls:
            logger.info(f"Video already downloaded: {url}")
            return False

        output_filename = f"penalty_{self.video_count:03d}"
        output_path = self.output_dir / output_filename

        logger.info(f"\n[{self.video_count}] Downloading: {url}")

        try:
            ydl_opts = self._get_ydl_opts(output_path)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                is_valid, reason = self._validate_video(info)

                if not is_valid:
                    logger.warning(f"✗ Invalid video: {reason}")
                    return False

                info = ydl.extract_info(url, download=True)

                video_file = output_path.with_suffix('.mp4')
                if not video_file.exists():
                    logger.error(f"✗ File not found after download")
                    return False

                self.downloaded_urls[url] = {
                    'filename': video_file.name,
                    'duration': info.get('duration'),
                    'resolution': f"{info.get('width')}x{info.get('height')}",
                    'fps': info.get('fps'),
                    'title': info.get('title'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'download_date': datetime.now().isoformat(),
                }
                self._save_downloaded()

                logger.info(f"✓ Downloaded: {video_file.name} ({info.get('duration')}s, {info.get('height')}p)")
                self.video_count += 1
                return True

        except DownloadError as e:
            logger.error(f"✗ Download error: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False

    def download_from_file(self, urls_file: str, delay: int = 3):
        urls_path = Path(urls_file)
        if not urls_path.exists():
            logger.error(f"File not found: {urls_file}")
            return

        with open(urls_path, 'r') as f:
            urls = [line.strip() for line in f
                    if line.strip() and not line.startswith('#')]

        logger.info(f"Found {len(urls)} URLs to process")
        logger.info("=" * 50)

        for i, url in enumerate(urls, 1):
            logger.info(f"\n[{i}/{len(urls)}] Processing...")
            self.download_video(url)

            if i < len(urls):
                logger.info(f"Sleeping {delay}s...")
                time.sleep(delay)

    def search_and_download(self, query: str, max_results: int = 50, delay: int = 5):
        logger.info(f"Searching: '{query}' (max {max_results} results)")

        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_url = f"ytsearch{max_results}:{query}"
                info = ydl.extract_info(search_url, download=False)

                if 'entries' not in info:
                    logger.warning("No results found")
                    return

                videos = info['entries']
                logger.info(f"Found {len(videos)} videos")
                logger.info("=" * 50)

                for i, video in enumerate(videos, 1):
                    if video is None:
                        continue

                    url = f"https://www.youtube.com/watch?v={video['id']}"
                    logger.info(f"\n[{i}/{len(videos)}] {video.get('title', 'N/A')}")
                    self.download_video(url)

                    if i < len(videos):
                        time.sleep(delay)

        except Exception as e:
            logger.error(f"Search error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download penalty kick videos from YouTube')
    parser.add_argument('--output', type=str, default='data/raw_videos',
                        help='Output directory for downloaded videos')
    parser.add_argument('--urls', type=str, default='penalty_urls.txt',
                        help='File containing YouTube URLs (one per line)')
    args = parser.parse_args()

    downloader = PenaltyVideoDownloader(output_dir=args.output)

    if os.path.exists(args.urls):
        logger.info(f"📄 Found URL file: {args.urls}")
        downloader.download_from_file(args.urls, delay=3)
    else:
        logger.info("🔍 No URL file found, starting automatic search...\n")
        queries = [
            "Serie A penalty compilation",
        ]

        for query in queries:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Query: {query}")
            logger.info('=' * 50)
            downloader.search_and_download(query, max_results=20, delay=5)

            logger.info("\nSleeping 10s before next search...")
            time.sleep(10)


if __name__ == "__main__":
    main()