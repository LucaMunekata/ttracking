"""Video frame reader wrapping OpenCV's VideoCapture."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata about a video file."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Reads frames from a video file using OpenCV.

    Usage:
        reader = VideoReader("match.mp4")
        print(reader.metadata)

        # Iterate every 30th frame (e.g., 1 per second at 30fps)
        for frame_num, frame in reader.frames(step=30):
            process(frame)

        reader.close()

    Or as a context manager:
        with VideoReader("match.mp4") as reader:
            for frame_num, frame in reader.frames():
                process(frame)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

        self.metadata = VideoMetadata(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def seek(self, frame_number: int) -> None:
        """Seek to a specific frame number."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def read_frame(self) -> np.ndarray | None:
        """Read the next frame. Returns None at end of video."""
        ret, frame = self._cap.read()
        return frame if ret else None

    def frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_number, frame) tuples.

        Args:
            start: First frame to read.
            stop: Stop before this frame (None = end of video).
            step: Read every Nth frame. step=30 at 30fps gives ~1 frame/sec.
        """
        if stop is None:
            stop = self.metadata.frame_count

        self.seek(start)
        current = start

        while current < stop:
            ret, frame = self._cap.read()
            if not ret:
                break

            if (current - start) % step == 0:
                yield current, frame

            current += 1

    def close(self) -> None:
        """Release the video capture."""
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
