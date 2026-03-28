"""Line filtering, merging, and classification for court detection.

After Hough line detection, we typically get many duplicate/fragmented lines.
This module groups similar lines and merges each group into a single
representative line, then classifies them as horizontal or vertical.
"""

import math

from tennis_tracking.court.classical import Line


def _line_polar_params(line: Line) -> tuple[float, float]:
    """Convert a line to polar form (theta, rho).

    theta: angle of the line's normal in radians [0, pi)
    rho: signed perpendicular distance from origin to the line

    This representation makes it easier to compare and group lines,
    since parallel lines have similar theta regardless of position,
    and rho captures how far apart they are.
    """
    dx = line.x2 - line.x1
    dy = line.y2 - line.y1

    # Angle of the line itself
    angle = math.atan2(dy, dx)
    # Normal angle (perpendicular to line direction)
    theta = angle + math.pi / 2
    # Normalize theta to [0, pi)
    theta = theta % math.pi

    # rho = x * cos(theta) + y * sin(theta), using the midpoint
    mx, my = line.midpoint
    rho = mx * math.cos(theta) + my * math.sin(theta)

    return theta, rho


def merge_similar_lines(
    lines: list[Line],
    angle_threshold_deg: float = 5.0,
    distance_threshold: float = 20.0,
) -> list[Line]:
    """Merge lines that are approximately the same into single representatives.

    Lines are grouped when both:
    - Their angles differ by less than angle_threshold_deg
    - Their perpendicular distance (rho) differs by less than distance_threshold pixels

    Each group is merged into one line by averaging endpoints, weighted by length.

    Args:
        lines: Detected line segments.
        angle_threshold_deg: Max angle difference (degrees) to consider lines similar.
        distance_threshold: Max perpendicular distance (pixels) to consider lines similar.

    Returns:
        Merged list of representative lines.
    """
    if not lines:
        return []

    angle_threshold = math.radians(angle_threshold_deg)

    # Compute polar params for each line
    polar = [_line_polar_params(line) for line in lines]

    # Greedy clustering: assign each line to the first compatible group
    groups: list[list[int]] = []  # each group is a list of line indices
    group_theta: list[float] = []
    group_rho: list[float] = []

    for i, (theta, rho) in enumerate(polar):
        assigned = False
        for g_idx in range(len(groups)):
            # Check angle similarity (handle wraparound near 0/pi)
            dtheta = abs(theta - group_theta[g_idx])
            dtheta = min(dtheta, math.pi - dtheta)

            drho = abs(rho - group_rho[g_idx])

            if dtheta < angle_threshold and drho < distance_threshold:
                groups[g_idx].append(i)
                # Update group center as weighted average
                n = len(groups[g_idx])
                group_theta[g_idx] = (group_theta[g_idx] * (n - 1) + theta) / n
                group_rho[g_idx] = (group_rho[g_idx] * (n - 1) + rho) / n
                assigned = True
                break

        if not assigned:
            groups.append([i])
            group_theta.append(theta)
            group_rho.append(rho)

    # Merge each group into one representative line
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(lines[group[0]])
            continue

        # Weight by line length: longer lines are more reliable
        total_weight = 0.0
        x1_acc, y1_acc, x2_acc, y2_acc = 0.0, 0.0, 0.0, 0.0

        for idx in group:
            line = lines[idx]
            w = line.length
            total_weight += w

            # Ensure consistent direction: longest line sets the direction
            x1_acc += line.x1 * w
            y1_acc += line.y1 * w
            x2_acc += line.x2 * w
            y2_acc += line.y2 * w

        merged.append(Line(
            x1=int(x1_acc / total_weight),
            y1=int(y1_acc / total_weight),
            x2=int(x2_acc / total_weight),
            y2=int(y2_acc / total_weight),
        ))

    return merged


def classify_lines(
    lines: list[Line],
    angle_cutoff: float = 45.0,
) -> tuple[list[Line], list[Line]]:
    """Separate lines into approximately horizontal and approximately vertical groups.

    In a broadcast tennis view:
    - Horizontal lines (~0°): baselines, service lines
    - Vertical lines (~90°): sidelines

    Args:
        lines: Detected/merged lines.
        angle_cutoff: Lines below this angle (from horizontal) are "horizontal",
                     above are "vertical". Default 45° splits evenly.

    Returns:
        (horizontal_lines, vertical_lines) tuple.
    """
    horizontal = []
    vertical = []

    for line in lines:
        if line.angle < angle_cutoff:
            horizontal.append(line)
        else:
            vertical.append(line)

    return horizontal, vertical
