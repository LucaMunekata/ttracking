"""Standard tennis court geometry based on ITF specifications.

All dimensions are in meters. The coordinate system has the origin at the
center of the court (where the net meets the center service line).

    Y axis: along the net (left-right from camera perspective)
    X axis: baseline-to-baseline (top-bottom from camera perspective)

Key ITF dimensions:
    - Full court length: 23.77m (baseline to baseline)
    - Doubles width: 10.97m
    - Singles width: 8.23m
    - Service line distance from net: 6.40m
    - Net height at center: 0.914m
    - Net height at posts: 1.07m
"""

from dataclasses import dataclass

# Half-dimensions (distance from center)
HALF_LENGTH = 23.77 / 2  # 11.885m from center to baseline
HALF_DOUBLES_WIDTH = 10.97 / 2  # 5.485m from center to doubles sideline
HALF_SINGLES_WIDTH = 8.23 / 2  # 4.115m from center to singles sideline
SERVICE_LINE_DIST = 6.40  # from net to service line


@dataclass(frozen=True)
class Point2D:
    """A 2D point in court coordinates (meters)."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class LineSegment:
    """A named line segment on the court."""

    name: str
    start: Point2D
    end: Point2D


class CourtModel:
    """ITF standard tennis court geometry.

    Provides keypoints (line intersections) and line segments for the court.
    All coordinates in meters, origin at court center (net/center-line crossing).
    """

    # Court dimensions
    FULL_LENGTH = 23.77
    DOUBLES_WIDTH = 10.97
    SINGLES_WIDTH = 8.23
    SERVICE_LINE_DISTANCE = 6.40
    NET_HEIGHT_CENTER = 0.914
    NET_HEIGHT_POSTS = 1.07

    def get_keypoints(self) -> dict[str, Point2D]:
        """Return named court intersection points.

        Returns a dict mapping descriptive names to 2D coordinates.
        Names follow the pattern: {near/far}_{left/right}_{feature}
        where "near" = negative X (bottom of screen), "far" = positive X (top).
        """
        hl = HALF_LENGTH
        dw = HALF_DOUBLES_WIDTH
        sw = HALF_SINGLES_WIDTH
        sl = SERVICE_LINE_DIST

        return {
            # Baselines (4 corners of doubles court)
            "far_baseline_left": Point2D(hl, -dw),
            "far_baseline_right": Point2D(hl, dw),
            "near_baseline_left": Point2D(-hl, -dw),
            "near_baseline_right": Point2D(-hl, dw),
            # Singles sideline / baseline intersections
            "far_singles_left": Point2D(hl, -sw),
            "far_singles_right": Point2D(hl, sw),
            "near_singles_left": Point2D(-hl, -sw),
            "near_singles_right": Point2D(-hl, sw),
            # Service line / singles sideline intersections
            "far_service_left": Point2D(sl, -sw),
            "far_service_right": Point2D(sl, sw),
            "near_service_left": Point2D(-sl, -sw),
            "near_service_right": Point2D(-sl, sw),
            # Service line extended to doubles sideline (not painted, but the
            # lines cross here — useful for homography when singles lines
            # aren't detected)
            "near_service_left_doubles": Point2D(-sl, -dw),
            "near_service_right_doubles": Point2D(-sl, dw),
            "far_service_left_doubles": Point2D(sl, -dw),
            "far_service_right_doubles": Point2D(sl, dw),
            # Center service line endpoints (on service lines)
            "far_service_center": Point2D(sl, 0),
            "near_service_center": Point2D(-sl, 0),
            # Net / sideline intersections
            "net_left_doubles": Point2D(0, -dw),
            "net_right_doubles": Point2D(0, dw),
            "net_left_singles": Point2D(0, -sw),
            "net_right_singles": Point2D(0, sw),
            "net_center": Point2D(0, 0),
        }

    def get_lines(self) -> list[LineSegment]:
        """Return all court line segments."""
        kp = self.get_keypoints()

        return [
            # Baselines
            LineSegment("far_baseline", kp["far_baseline_left"], kp["far_baseline_right"]),
            LineSegment("near_baseline", kp["near_baseline_left"], kp["near_baseline_right"]),
            # Doubles sidelines
            LineSegment(
                "left_doubles_sideline", kp["near_baseline_left"], kp["far_baseline_left"]
            ),
            LineSegment(
                "right_doubles_sideline", kp["near_baseline_right"], kp["far_baseline_right"]
            ),
            # Singles sidelines
            LineSegment(
                "left_singles_sideline", kp["near_singles_left"], kp["far_singles_left"]
            ),
            LineSegment(
                "right_singles_sideline", kp["near_singles_right"], kp["far_singles_right"]
            ),
            # Service lines
            LineSegment("far_service_line", kp["far_service_left"], kp["far_service_right"]),
            LineSegment(
                "near_service_line", kp["near_service_left"], kp["near_service_right"]
            ),
            # Center service line
            LineSegment(
                "center_service_line",
                kp["near_service_center"],
                kp["far_service_center"],
            ),
            # Net (not a painted line, but useful for visualization)
            LineSegment("net", kp["net_left_doubles"], kp["net_right_doubles"]),
        ]

    def get_near_half_lines(self) -> list[LineSegment]:
        """Return only the near-half court lines (baseline to net).

        Useful when the camera only sees the near side clearly.
        """
        kp = self.get_keypoints()

        return [
            LineSegment("near_baseline", kp["near_baseline_left"], kp["near_baseline_right"]),
            # Sidelines from baseline to net
            LineSegment(
                "left_doubles_sideline", kp["near_baseline_left"], kp["net_left_doubles"]
            ),
            LineSegment(
                "right_doubles_sideline", kp["near_baseline_right"], kp["net_right_doubles"]
            ),
            LineSegment(
                "left_singles_sideline", kp["near_singles_left"], kp["net_left_singles"]
            ),
            LineSegment(
                "right_singles_sideline", kp["near_singles_right"], kp["net_right_singles"]
            ),
            # Near service line
            LineSegment(
                "near_service_line", kp["near_service_left"], kp["near_service_right"]
            ),
            # Center service line
            LineSegment(
                "center_service_line",
                kp["near_service_center"],
                kp["net_center"],
            ),
        ]
