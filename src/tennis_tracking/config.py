"""Configuration models for all tunable CV parameters."""

from pydantic import BaseModel


class WhiteMaskConfig(BaseModel):
    """HSV thresholds for isolating white court lines.

    In HSV space, white pixels have:
    - Low saturation (S is small — little color)
    - High value (V is large — bright)
    - Hue can be anything (white has no hue)
    """

    min_saturation: int = 0
    max_saturation: int = 50
    min_value: int = 200
    max_value: int = 255


class CourtDetectionConfig(BaseModel):
    """All tunable parameters for the classical court detection pipeline."""

    # Preprocessing
    blur_kernel_size: int = 5  # Gaussian blur kernel (must be odd)
    use_white_mask: bool = True  # Whether to mask for white lines before edge detection
    white_mask: WhiteMaskConfig = WhiteMaskConfig()

    # Region of interest — percentage of frame to ignore from each edge.
    # For example, roi_top=15 blacks out the top 15% of the frame (scoreboard area).
    roi_top: float = 0.0
    roi_bottom: float = 0.0
    roi_left: float = 0.0
    roi_right: float = 0.0

    # Canny edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Hough line transform (probabilistic)
    hough_rho: float = 1.0  # Distance resolution in pixels
    hough_theta_degrees: float = 1.0  # Angle resolution in degrees
    hough_threshold: int = 50  # Minimum votes (intersections)
    hough_min_line_length: int = 50  # Minimum line length in pixels
    hough_max_line_gap: int = 10  # Maximum gap between line segments to merge

    # Line filtering
    merge_angle_threshold: float = 5.0  # Max angle difference (degrees) to merge lines
    merge_distance_threshold: float = 20.0  # Max perpendicular distance (pixels) to merge
