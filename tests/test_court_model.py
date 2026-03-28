"""Tests for CourtModel verifying ITF dimensions."""

import math

import pytest

from tennis_tracking.court.court_model import CourtModel, Point2D


@pytest.fixture
def court() -> CourtModel:
    return CourtModel()


def _distance(a: Point2D, b: Point2D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def test_court_length(court: CourtModel):
    kp = court.get_keypoints()
    # Baseline to baseline along a sideline
    length = _distance(kp["near_baseline_left"], kp["far_baseline_left"])
    assert length == pytest.approx(23.77, abs=0.01)


def test_doubles_width(court: CourtModel):
    kp = court.get_keypoints()
    width = _distance(kp["far_baseline_left"], kp["far_baseline_right"])
    assert width == pytest.approx(10.97, abs=0.01)


def test_singles_width(court: CourtModel):
    kp = court.get_keypoints()
    width = _distance(kp["far_singles_left"], kp["far_singles_right"])
    assert width == pytest.approx(8.23, abs=0.01)


def test_service_line_distance_from_net(court: CourtModel):
    kp = court.get_keypoints()
    # Service line to net center
    dist = _distance(kp["far_service_center"], kp["net_center"])
    assert dist == pytest.approx(6.40, abs=0.01)
    dist = _distance(kp["near_service_center"], kp["net_center"])
    assert dist == pytest.approx(6.40, abs=0.01)


def test_court_is_symmetric(court: CourtModel):
    kp = court.get_keypoints()
    # Near and far baselines should be equidistant from net
    assert abs(kp["near_baseline_left"].x) == pytest.approx(
        abs(kp["far_baseline_left"].x), abs=0.001
    )
    # Left and right should be symmetric about Y=0
    assert kp["far_baseline_left"].y == pytest.approx(
        -kp["far_baseline_right"].y, abs=0.001
    )


def test_keypoint_count(court: CourtModel):
    kp = court.get_keypoints()
    assert len(kp) == 23


def test_line_count(court: CourtModel):
    lines = court.get_lines()
    assert len(lines) == 10


def test_net_at_origin(court: CourtModel):
    kp = court.get_keypoints()
    assert kp["net_center"].x == pytest.approx(0.0)
    assert kp["net_center"].y == pytest.approx(0.0)
