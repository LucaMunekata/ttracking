"""Homography estimation: maps between image pixels and real-world court coordinates.

The homography is a 3x3 matrix H such that:
    court_point = H @ image_point  (in homogeneous coordinates)

The approach:
1. Classify detected lines into court roles (baseline, sideline, service line)
   based on their relative positions and angles.
2. Compute intersections between identified lines — since we know which lines
   are which, we know exactly which court keypoint each intersection maps to.
3. Use these known correspondences to compute the homography via RANSAC.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from tennis_tracking.court.classical import Line
from tennis_tracking.court.court_model import CourtModel, Point2D


@dataclass
class HomographyResult:
    """Result of homography estimation."""

    H: np.ndarray  # 3x3 homography matrix (image -> court)
    inlier_count: int  # Number of RANSAC inliers
    total_points: int  # Total matched points used
    reprojection_error: float  # Mean reprojection error in pixels

    @property
    def confidence(self) -> float:
        """Rough confidence score: inlier ratio * (1 if error is low)."""
        if self.total_points == 0:
            return 0.0
        ratio = self.inlier_count / self.total_points
        # Penalize high reprojection error
        error_factor = max(0.0, 1.0 - self.reprojection_error / 10.0)
        return ratio * error_factor


@dataclass
class IdentifiedLines:
    """Court lines identified by their role.

    In a broadcast view looking at the near half of the court:
    - Horizontal lines (sorted top to bottom in image):
      service line (far), then possibly center, then baseline (near/bottom)
    - Vertical lines (sorted left to right in image):
      left doubles sideline, left singles sideline,
      right singles sideline, right doubles sideline
    """

    baseline: Line | None = None
    service_line: Line | None = None
    left_sideline: Line | None = None
    right_sideline: Line | None = None
    # Optional — may not always be detected
    left_singles: Line | None = None
    right_singles: Line | None = None
    center_service: Line | None = None


def find_intersections(
    h_lines: list[Line],
    v_lines: list[Line],
    frame_width: int,
    frame_height: int,
    margin: float = 0.05,
) -> list[tuple[float, float]]:
    """Find intersection points between horizontal and vertical line groups.

    Only returns points that fall within the frame (with a small margin).
    """
    points = []
    x_margin = frame_width * margin
    y_margin = frame_height * margin

    for h in h_lines:
        for v in v_lines:
            pt = _line_intersection(h, v)
            if pt is None:
                continue

            x, y = pt
            if (-x_margin <= x <= frame_width + x_margin
                    and -y_margin <= y <= frame_height + y_margin):
                points.append((x, y))

    return points


def _line_intersection(line1: Line, line2: Line) -> tuple[float, float] | None:
    """Compute intersection of two lines (extended to infinite lines).

    Uses cross-product in homogeneous coordinates.
    Returns None if lines are parallel.
    """
    p1 = np.array([line1.x1, line1.y1, 1.0])
    p2 = np.array([line1.x2, line1.y2, 1.0])
    p3 = np.array([line2.x1, line2.y1, 1.0])
    p4 = np.array([line2.x2, line2.y2, 1.0])

    l1 = np.cross(p1, p2)
    l2 = np.cross(p3, p4)
    intersection = np.cross(l1, l2)

    if abs(intersection[2]) < 1e-8:
        return None

    x = intersection[0] / intersection[2]
    y = intersection[1] / intersection[2]
    return (x, y)


def identify_lines(
    horizontal: list[Line],
    vertical: list[Line],
) -> IdentifiedLines:
    """Identify which detected lines correspond to which court features.

    Uses spatial reasoning based on the broadcast camera perspective:
    - The camera looks from behind/above one baseline toward the other.
    - In the image, the near baseline is at the bottom, far features at top.
    - Sidelines converge toward the top (perspective).

    Horizontal lines (sorted by vertical midpoint position in image):
    - Bottom-most = near baseline
    - Top-most = service line (since we cropped the far half)

    Vertical lines (sorted by horizontal midpoint position):
    - Outermost left/right = doubles sidelines
    - Inner left/right = singles sidelines (if detected)
    """
    result = IdentifiedLines()

    # --- Horizontal lines ---
    if horizontal:
        # Sort by Y midpoint: smallest Y = highest in image = farthest from camera
        h_sorted = sorted(horizontal, key=lambda ln: ln.midpoint[1])

        if len(h_sorted) >= 1:
            # Bottom-most horizontal line = near baseline
            result.baseline = h_sorted[-1]
        if len(h_sorted) >= 2:
            # Top-most horizontal line = service line
            result.service_line = h_sorted[0]

    # --- Vertical lines ---
    if vertical:
        # Sort by X midpoint: smallest X = leftmost in image
        v_sorted = sorted(vertical, key=lambda ln: ln.midpoint[0])

        if len(v_sorted) >= 2:
            # Outermost = doubles sidelines
            result.left_sideline = v_sorted[0]
            result.right_sideline = v_sorted[-1]

        if len(v_sorted) >= 4:
            # Inner pair = singles sidelines
            result.left_singles = v_sorted[1]
            result.right_singles = v_sorted[-2]
        elif len(v_sorted) == 3:
            # Three vertical lines: outermost two are doubles, middle is ambiguous.
            # Check which singles sideline it's closer to by comparing distances
            # from left and right doubles sidelines.
            mid = v_sorted[1]
            left_dist = abs(mid.midpoint[0] - v_sorted[0].midpoint[0])
            right_dist = abs(mid.midpoint[0] - v_sorted[-1].midpoint[0])
            if left_dist < right_dist:
                result.left_singles = mid
            else:
                result.right_singles = mid

    return result


def build_correspondences(
    identified: IdentifiedLines,
    court: CourtModel,
    frame_width: int,
    frame_height: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build image-to-court point correspondences from identified lines.

    For each pair of identified lines that intersect, we know exactly
    which court keypoint that intersection corresponds to.
    """
    kp = court.get_keypoints()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    # Define which line pairs produce which keypoints.
    # Each entry: (line_a, line_b, court_keypoint_name)
    # We use "near_" keypoints since we're looking at the near half.
    correspondences = [
        # Baseline x sidelines (all 4 corners exist on a real court)
        ("baseline", "left_sideline", "near_baseline_left"),
        ("baseline", "right_sideline", "near_baseline_right"),
        ("baseline", "left_singles", "near_singles_left"),
        ("baseline", "right_singles", "near_singles_right"),
        # Service line x singles sidelines (painted corners)
        ("service_line", "left_singles", "near_service_left"),
        ("service_line", "right_singles", "near_service_right"),
        # Service line x doubles sidelines (not painted, but the lines
        # cross here and we know the court coordinates)
        ("service_line", "left_sideline", "near_service_left_doubles"),
        ("service_line", "right_sideline", "near_service_right_doubles"),
    ]

    for line_a_name, line_b_name, kp_name in correspondences:
        line_a = getattr(identified, line_a_name, None)
        line_b = getattr(identified, line_b_name, None)

        if line_a is None or line_b is None:
            continue

        if kp_name not in kp:
            continue

        pt = _line_intersection(line_a, line_b)
        if pt is None:
            continue

        # Check that the intersection is roughly within the frame
        x, y = pt
        margin = 0.1
        if (x < -frame_width * margin or x > frame_width * (1 + margin)
                or y < -frame_height * margin or y > frame_height * (1 + margin)):
            continue

        img_pt = np.array([x, y], dtype=np.float64)
        court_pt = np.array(kp[kp_name].as_tuple(), dtype=np.float64)
        pairs.append((img_pt, court_pt))

    return pairs


def estimate_homography(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    ransac_threshold: float = 5.0,
) -> HomographyResult | None:
    """Estimate homography from matched point pairs.

    Args:
        pairs: List of (image_point, court_point) arrays, each shape (2,).
        ransac_threshold: RANSAC inlier distance threshold in pixels.

    Returns:
        HomographyResult, or None if estimation fails (< 4 pairs).
    """
    if len(pairs) < 4:
        return None

    src = np.array([p[0] for p in pairs], dtype=np.float64)  # image points
    dst = np.array([p[1] for p in pairs], dtype=np.float64)  # court points

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)

    if H is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())

    # Compute reprojection error for inliers
    src_h = np.hstack([src, np.ones((len(src), 1))])  # to homogeneous
    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]  # from homogeneous

    errors = np.linalg.norm(projected - dst, axis=1)
    mean_error = float(errors[inlier_mask].mean()) if inlier_count > 0 else float("inf")

    return HomographyResult(
        H=H,
        inlier_count=inlier_count,
        total_points=len(pairs),
        reprojection_error=mean_error,
    )


def warp_court_overlay(
    frame: np.ndarray,
    H: np.ndarray,
    court: CourtModel,
    near_half_only: bool = True,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the court model wireframe warped onto the video frame.

    Uses the inverse homography (court -> image) to project court lines
    onto the frame.

    Args:
        near_half_only: If True, only draw the near half of the court.
            This avoids wild projections when the homography is estimated
            from near-side points only.
    """
    vis = frame.copy()

    H_inv = np.linalg.inv(H)  # court -> image

    def project_point(court_pt: Point2D) -> tuple[int, int] | None:
        """Project a court coordinate to image pixel."""
        pt = np.array([court_pt.x, court_pt.y, 1.0])
        img_pt = H_inv @ pt
        if abs(img_pt[2]) < 1e-8:
            return None
        return (int(img_pt[0] / img_pt[2]), int(img_pt[1] / img_pt[2]))

    lines = court.get_near_half_lines() if near_half_only else court.get_lines()
    for line_seg in lines:
        p1 = project_point(line_seg.start)
        p2 = project_point(line_seg.end)
        if p1 is not None and p2 is not None:
            cv2.line(vis, p1, p2, color, thickness)

    return vis
