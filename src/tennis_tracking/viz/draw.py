"""Visualization utilities for drawing detections on video frames."""

import cv2
import numpy as np

from tennis_tracking.court.classical import Line

# Color palette (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)


def draw_lines(
    frame: np.ndarray,
    lines: list[Line],
    color: tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 2,
) -> np.ndarray:
    """Draw line segments on a frame.

    Args:
        frame: BGR image (will be copied, not modified in place).
        lines: Lines to draw.
        color: BGR color tuple.
        thickness: Line thickness in pixels.

    Returns:
        Copy of frame with lines drawn.
    """
    vis = frame.copy()
    for line in lines:
        cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
    return vis


def draw_classified_lines(
    frame: np.ndarray,
    horizontal: list[Line],
    vertical: list[Line],
    h_color: tuple[int, int, int] = COLOR_BLUE,
    v_color: tuple[int, int, int] = COLOR_RED,
    thickness: int = 2,
) -> np.ndarray:
    """Draw horizontal and vertical lines in different colors.

    Args:
        frame: BGR image (will be copied).
        horizontal: Horizontal lines (baselines, service lines).
        vertical: Vertical lines (sidelines).
        h_color: Color for horizontal lines.
        v_color: Color for vertical lines.
        thickness: Line thickness.

    Returns:
        Annotated frame.
    """
    vis = frame.copy()
    for line in horizontal:
        cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), h_color, thickness)
    for line in vertical:
        cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), v_color, thickness)

    # Legend
    cv2.putText(vis, f"H: {len(horizontal)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, h_color, 2)
    cv2.putText(vis, f"V: {len(vertical)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, v_color, 2)
    return vis


def draw_points(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int] = COLOR_YELLOW,
    radius: int = 5,
) -> np.ndarray:
    """Draw points (e.g., line intersections) on a frame.

    Args:
        frame: BGR image (will be copied).
        points: List of (x, y) coordinates.
        color: BGR color.
        radius: Circle radius in pixels.

    Returns:
        Annotated frame.
    """
    vis = frame.copy()
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis
