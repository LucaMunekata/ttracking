"""Classical CV pipeline for court line detection.

Pipeline: preprocess → edge detection → Hough lines → filter/merge → classify.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from tennis_tracking.config import CourtDetectionConfig


@dataclass
class Line:
    """A detected line segment."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def angle(self) -> float:
        """Angle in degrees from horizontal (0-180)."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    @property
    def length(self) -> float:
        """Length in pixels."""
        return math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def endpoints_array(self) -> np.ndarray:
        """Return as [[x1, y1], [x2, y2]]."""
        return np.array([[self.x1, self.y1], [self.x2, self.y2]], dtype=np.float64)


def preprocess_frame(frame: np.ndarray, config: CourtDetectionConfig) -> np.ndarray:
    """Convert a BGR frame to an edge image suitable for Hough line detection.

    Steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. (Optional) Mask for white pixels using HSV thresholds
        4. Run Canny edge detection

    Args:
        frame: BGR image from OpenCV.
        config: Detection parameters.

    Returns:
        Binary edge image (uint8, values 0 or 255).
    """
    # Apply ROI mask: black out regions outside the court area.
    # This prevents scoreboards, ad boards, and crowd from producing edges.
    working = frame.copy()
    h, w = working.shape[:2]
    if config.roi_top > 0 or config.roi_bottom > 0 or config.roi_left > 0 or config.roi_right > 0:
        top = int(h * config.roi_top / 100)
        bottom = int(h * config.roi_bottom / 100)
        left = int(w * config.roi_left / 100)
        right = int(w * config.roi_right / 100)
        # Black out margins
        if top > 0:
            working[:top, :] = 0
        if bottom > 0:
            working[h - bottom:, :] = 0
        if left > 0:
            working[:, :left] = 0
        if right > 0:
            working[:, w - right:] = 0

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    # Gaussian blur: reduces noise so Canny doesn't fire on texture
    k = config.blur_kernel_size
    if k % 2 == 0:
        k += 1  # kernel must be odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if config.use_white_mask:
        # White court lines: low saturation, high value in HSV
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        wm = config.white_mask
        lower = np.array([0, wm.min_saturation, wm.min_value])
        upper = np.array([180, wm.max_saturation, wm.max_value])
        mask = cv2.inRange(hsv, lower, upper)

        # Apply mask to the blurred grayscale image
        blurred = cv2.bitwise_and(blurred, blurred, mask=mask)

    edges = cv2.Canny(blurred, config.canny_low, config.canny_high)
    return edges


def detect_lines(edges: np.ndarray, config: CourtDetectionConfig) -> list[Line]:
    """Run probabilistic Hough line transform on an edge image.

    Args:
        edges: Binary edge image from preprocess_frame().
        config: Detection parameters.

    Returns:
        List of detected Line segments.
    """
    theta = np.deg2rad(config.hough_theta_degrees)

    raw = cv2.HoughLinesP(
        edges,
        rho=config.hough_rho,
        theta=theta,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )

    if raw is None:
        return []

    return [
        Line(x1=int(seg[0]), y1=int(seg[1]), x2=int(seg[2]), y2=int(seg[3]))
        for seg in raw[:, 0]
    ]
