#!/usr/bin/env python3
"""Interactive parameter tuning for court detection using OpenCV trackbars.

Usage:
    uv run python scripts/tune_params.py <image_or_video_path> [--frame N]

Controls:
    - Trackbars adjust preprocessing and Hough parameters in real time
    - Press 'q' to quit
    - Press 's' to save current parameters to a JSON config file

This script helps you find good detection parameters for different
video sources / camera angles before running the full pipeline.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tennis_tracking.config import CourtDetectionConfig
from tennis_tracking.court.classical import Line, detect_lines, preprocess_frame


def nothing(_: int) -> None:
    """No-op callback for trackbars."""
    pass


LEGEND_LINES = [
    "Blur Kernel     - Smoothing before edges (higher = less noise)",
    "Canny Low/High  - Edge sensitivity (lower = more edges)",
    "White S Max     - Max color saturation (court lines are white)",
    "White V Min     - Min brightness (court lines are bright)",
    "Hough Threshold - Min votes for a line (higher = fewer lines)",
    "Min Line Length  - Ignore short segments (pixels)",
    "Max Line Gap    - Bridge gaps in a line (pixels)",
    "ROI Top/Bot/L/R - Crop edges to ignore scoreboard/ads (%)",
]


def draw_legend(display: np.ndarray) -> np.ndarray:
    """Draw a parameter legend at the bottom of the display."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    color = (200, 200, 200)
    line_height = 18
    padding = 8

    legend_h = len(LEGEND_LINES) * line_height + 2 * padding
    h, w = display.shape[:2]

    # Semi-transparent dark background
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - legend_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    for i, text in enumerate(LEGEND_LINES):
        y = h - legend_h + padding + (i + 1) * line_height
        cv2.putText(display, text, (10, y), font, scale, color, 1, cv2.LINE_AA)

    return display


def draw_lines_on_frame(frame: np.ndarray, lines: list[Line]) -> np.ndarray:
    """Draw detected lines in green on a copy of the frame."""
    vis = frame.copy()
    for line in lines:
        cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 2)
    return vis


def load_frame(path: str, frame_number: int = 0) -> np.ndarray:
    """Load a frame from an image file or video."""
    # Try as image first
    img = cv2.imread(path)
    if img is not None:
        return img

    # Try as video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open {path}", file=sys.stderr)
        sys.exit(1)

    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {frame_number} from {path}", file=sys.stderr)
        sys.exit(1)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive court detection parameter tuning")
    parser.add_argument("path", help="Path to image or video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number if using a video")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Load initial parameters from a JSON config file",
    )
    parser.add_argument(
        "--save-to", type=str, default="data/court_config.json",
        help="Where to save config when pressing 's' (default: data/court_config.json)",
    )
    args = parser.parse_args()

    frame = load_frame(args.path, args.frame)

    # Resize if very large (for display)
    max_dim = 1280
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    window = "Court Detection Tuning"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Load initial config from file or use defaults
    if args.config:
        config = CourtDetectionConfig.model_validate_json(Path(args.config).read_text())
        print(f"Loaded config from {args.config}")
    else:
        config = CourtDetectionConfig()
    cv2.createTrackbar("Blur Kernel", window, config.blur_kernel_size, 31, nothing)
    cv2.createTrackbar("Canny Low", window, config.canny_low, 255, nothing)
    cv2.createTrackbar("Canny High", window, config.canny_high, 255, nothing)
    cv2.createTrackbar("White Mask S Max", window, config.white_mask.max_saturation, 255, nothing)
    cv2.createTrackbar("White Mask V Min", window, config.white_mask.min_value, 255, nothing)
    cv2.createTrackbar("Hough Threshold", window, config.hough_threshold, 200, nothing)
    cv2.createTrackbar("Min Line Length", window, config.hough_min_line_length, 300, nothing)
    cv2.createTrackbar("Max Line Gap", window, config.hough_max_line_gap, 50, nothing)
    cv2.createTrackbar("ROI Top %", window, int(config.roi_top), 50, nothing)
    cv2.createTrackbar("ROI Bottom %", window, int(config.roi_bottom), 50, nothing)
    cv2.createTrackbar("ROI Left %", window, int(config.roi_left), 50, nothing)
    cv2.createTrackbar("ROI Right %", window, int(config.roi_right), 50, nothing)

    print("Controls: 'q' = quit, 's' = print current params")

    while True:
        # Read trackbar values
        blur = cv2.getTrackbarPos("Blur Kernel", window)
        if blur < 1:
            blur = 1
        if blur % 2 == 0:
            blur += 1

        config = CourtDetectionConfig(
            blur_kernel_size=blur,
            canny_low=cv2.getTrackbarPos("Canny Low", window),
            canny_high=cv2.getTrackbarPos("Canny High", window),
            white_mask=config.white_mask.model_copy(
                update={
                    "max_saturation": cv2.getTrackbarPos("White Mask S Max", window),
                    "min_value": cv2.getTrackbarPos("White Mask V Min", window),
                }
            ),
            hough_threshold=max(1, cv2.getTrackbarPos("Hough Threshold", window)),
            hough_min_line_length=max(1, cv2.getTrackbarPos("Min Line Length", window)),
            hough_max_line_gap=max(1, cv2.getTrackbarPos("Max Line Gap", window)),
            roi_top=float(cv2.getTrackbarPos("ROI Top %", window)),
            roi_bottom=float(cv2.getTrackbarPos("ROI Bottom %", window)),
            roi_left=float(cv2.getTrackbarPos("ROI Left %", window)),
            roi_right=float(cv2.getTrackbarPos("ROI Right %", window)),
        )

        # Run pipeline
        edges = preprocess_frame(frame, config)
        lines = detect_lines(edges, config)

        # Build display: original with lines on left, edges on right
        lines_vis = draw_lines_on_frame(frame, lines)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        display = np.hstack([lines_vis, edges_color])
        cv2.putText(
            display, f"Lines: {len(lines)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        display = draw_legend(display)

        cv2.imshow(window, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_path = Path(args.save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(config.model_dump_json(indent=2))
            print(f"\nConfig saved to {save_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
