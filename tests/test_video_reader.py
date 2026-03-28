"""Tests for VideoReader using a synthetic test video."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tennis_tracking.video.reader import VideoReader


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    """Create a small synthetic video (60 frames at 30fps = 2 seconds)."""
    path = tmp_path / "test.mp4"
    width, height, fps, num_frames = 320, 240, 30.0, 60

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Each frame has a unique solid color so we can verify which frame we read
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path


def test_metadata(test_video: Path):
    with VideoReader(test_video) as reader:
        assert reader.metadata.width == 320
        assert reader.metadata.height == 240
        assert reader.metadata.fps == pytest.approx(30.0, abs=1.0)
        assert reader.metadata.frame_count == 60
        assert reader.metadata.duration == pytest.approx(2.0, abs=0.1)


def test_read_frame(test_video: Path):
    with VideoReader(test_video) as reader:
        frame = reader.read_frame()
        assert frame is not None
        assert frame.shape == (240, 320, 3)


def test_read_frame_returns_none_at_end(test_video: Path):
    with VideoReader(test_video) as reader:
        reader.seek(60)
        frame = reader.read_frame()
        assert frame is None


def test_seek(test_video: Path):
    with VideoReader(test_video) as reader:
        reader.seek(30)
        frame = reader.read_frame()
        assert frame is not None


def test_frames_all(test_video: Path):
    with VideoReader(test_video) as reader:
        all_frames = list(reader.frames())
        assert len(all_frames) == 60
        # Check frame numbers are sequential
        assert [f[0] for f in all_frames] == list(range(60))


def test_frames_with_step(test_video: Path):
    with VideoReader(test_video) as reader:
        frames = list(reader.frames(step=10))
        assert len(frames) == 6
        assert [f[0] for f in frames] == [0, 10, 20, 30, 40, 50]


def test_frames_with_start_stop(test_video: Path):
    with VideoReader(test_video) as reader:
        frames = list(reader.frames(start=10, stop=30))
        assert len(frames) == 20
        assert frames[0][0] == 10
        assert frames[-1][0] == 29


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        VideoReader("/nonexistent/video.mp4")
