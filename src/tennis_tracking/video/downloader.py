"""Download tennis match videos from YouTube using yt-dlp."""

import subprocess
from pathlib import Path

# Default download directory (relative to project root)
DEFAULT_OUTPUT_DIR = Path("data/videos")


def download_video(
    url: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_height: int = 720,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Download a video from YouTube.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the video.
        max_height: Maximum video height in pixels (720 = 720p).
        start: Start time in seconds (None = from beginning).
        end: End time in seconds (None = to end).

    Returns:
        Path to the downloaded video file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output template: video title as filename
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--print", "after_move:filepath",  # Print final path to stdout
        "--no-simulate",
    ]

    if start is not None or end is not None:
        s = start if start is not None else 0
        e = end if end is not None else "inf"
        cmd += ["--download-sections", f"*{s}-{e}"]

    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Last non-empty line of stdout is the filepath
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp did not return a file path. stderr: {result.stderr}")

    return Path(lines[-1])
