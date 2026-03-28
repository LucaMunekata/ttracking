#!/usr/bin/env python3
"""Run court detection on a video or image and display/save results.

Usage:
    # Process a single image
    uv run python scripts/detect_court.py frame.jpg

    # Process a video (samples every Nth frame)
    uv run python scripts/detect_court.py match.mp4 --step 30

    # Save output instead of displaying
    uv run python scripts/detect_court.py match.mp4 --output data/outputs/result.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tennis_tracking.config import CourtDetectionConfig
from tennis_tracking.court.classical import detect_lines, preprocess_frame
from tennis_tracking.court.court_model import CourtModel
from tennis_tracking.court.filters import classify_lines, merge_similar_lines
from tennis_tracking.court.homography import (
    HomographyResult,
    build_correspondences,
    estimate_homography,
    identify_lines,
    warp_court_overlay,
)
from tennis_tracking.video.reader import VideoReader
from tennis_tracking.viz.draw import draw_classified_lines, draw_points


def process_frame(
    frame: np.ndarray,
    config: CourtDetectionConfig,
    court: CourtModel,
) -> tuple[np.ndarray, HomographyResult | None]:
    """Run the full court detection pipeline on a single frame.

    Returns:
        (annotated_frame, homography_result or None)
    """
    h, w = frame.shape[:2]

    # Step 1: Preprocessing and edge detection
    edges = preprocess_frame(frame, config)

    # Step 2: Hough line detection
    raw_lines = detect_lines(edges, config)

    # Step 3: Merge and classify
    merged = merge_similar_lines(
        raw_lines,
        angle_threshold_deg=config.merge_angle_threshold,
        distance_threshold=config.merge_distance_threshold,
    )
    horizontal, vertical = classify_lines(merged)

    # Step 4: Identify which lines are which court features
    identified = identify_lines(horizontal, vertical)

    # Step 5: Build known correspondences and estimate homography
    pairs = build_correspondences(identified, court, w, h)
    result = estimate_homography(pairs)

    # Build annotated frame
    vis = draw_classified_lines(frame, horizontal, vertical)

    # Show intersection points only from identified correspondences
    corr_points = [tuple(p[0]) for p in pairs]
    vis = draw_points(vis, corr_points)

    if result is not None and result.confidence > 0.1:
        vis = warp_court_overlay(vis, result.H, court)

    # Add info text
    info = f"Lines: {len(merged)} (H:{len(horizontal)} V:{len(vertical)})"
    info += f" | Matched: {len(pairs)}"
    if result:
        info += f" | Conf: {result.confidence:.2f} | Err: {result.reprojection_error:.1f}"
    cv2.putText(vis, info, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis, result


def process_image(path: str, config: CourtDetectionConfig, output: str | None) -> None:
    """Process a single image."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: Cannot read image {path}", file=sys.stderr)
        sys.exit(1)

    court = CourtModel()
    vis, result = process_frame(frame, config, court)

    if result:
        print(f"Homography confidence: {result.confidence:.2f}")
        print(f"Inliers: {result.inlier_count}/{result.total_points}")
        print(f"Reprojection error: {result.reprojection_error:.2f}")
    else:
        print("Could not estimate homography (not enough matched points)")

    if output:
        cv2.imwrite(output, vis)
        print(f"Saved to: {output}")
    else:
        cv2.imshow("Court Detection", vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    path: str,
    config: CourtDetectionConfig,
    step: int,
    output: str | None,
    delay: int = 0,
) -> None:
    """Process a video, sampling every `step` frames."""
    court = CourtModel()
    writer = None

    with VideoReader(path) as reader:
        print(f"Video: {reader.metadata.width}x{reader.metadata.height}")
        print(f"FPS: {reader.metadata.fps}, Frames: {reader.metadata.frame_count}")
        print(f"Sampling every {step} frames")
        print()

        for frame_num, frame in reader.frames(step=step):
            vis, result = process_frame(frame, config, court)

            conf = result.confidence if result else 0.0
            status = f"Frame {frame_num:>6d} | Confidence: {conf:.2f}"
            print(f"\r{status}", end="", flush=True)

            if output:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, reader.metadata.fps / step, (w, h))
                writer.write(vis)
            else:
                cv2.imshow("Court Detection", vis)
                # delay=0 pauses until keypress, otherwise waits N ms
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    print("\nStopped by user")
                    break

    if writer:
        writer.release()
        print(f"\nSaved to: {output}")
    else:
        cv2.destroyAllWindows()

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect tennis court lines in video/images")
    parser.add_argument("path", help="Path to video or image file")
    parser.add_argument("--step", type=int, default=30, help="Frame step for video (default: 30)")
    parser.add_argument("--output", "-o", help="Output path (image or video)")
    parser.add_argument(
        "--delay", type=int, default=0,
        help="Ms between frames (0 = pause until keypress, 500 = half second)",
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Load parameters from a JSON config file (e.g., from tune_params.py)",
    )

    args = parser.parse_args()

    if args.config:
        config = CourtDetectionConfig.model_validate_json(Path(args.config).read_text())
        print(f"Loaded config from {args.config}")
    else:
        config = CourtDetectionConfig()

    # Determine if input is image or video
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    ext = Path(args.path).suffix.lower()
    if ext in image_extensions:
        process_image(args.path, config, args.output)
    else:
        process_video(args.path, config, args.step, args.output, args.delay)


if __name__ == "__main__":
    main()
