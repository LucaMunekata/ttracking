#!/usr/bin/env python3
"""Download a tennis match video from YouTube.

Usage:
    uv run python scripts/download_video.py URL [--output-dir DIR] [--max-height N]
"""

import argparse
import sys

from tennis_tracking.video.downloader import download_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a tennis match video")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir", default="data/videos", help="Output directory (default: data/videos)"
    )
    parser.add_argument(
        "--max-height", type=int, default=720, help="Max video height in px (default: 720)"
    )
    parser.add_argument(
        "--start", type=float, default=None, help="Start time in seconds"
    )
    parser.add_argument(
        "--end", type=float, default=None, help="End time in seconds"
    )
    args = parser.parse_args()

    print(f"Downloading: {args.url}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Max height:  {args.max_height}p")
    if args.start is not None or args.end is not None:
        print(f"Section:     {args.start or 0}s - {args.end or 'end'}s")

    try:
        path = download_video(
            args.url, args.output_dir, args.max_height, args.start, args.end
        )
        print(f"Saved to: {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
