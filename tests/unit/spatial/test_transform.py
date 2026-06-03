"""Spatial transform tests.

These verify the pixel-to-pitch math against known homographies.
"""

from __future__ import annotations

import pytest

from datum.cv.pitch.schemas import Homography
from datum.spatial.transform import (
    pixel_box_to_pitch_anchors,
    pixel_to_pitch,
)


def _identity() -> Homography:
    return Homography(
        matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        method="test",
        inlier_count=4,
        reprojection_error_px=0.0,
    )


def _translation(tx: float, ty: float) -> Homography:
    return Homography(
        matrix=[[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        method="test",
        inlier_count=4,
        reprojection_error_px=0.0,
    )


def _scale(sx: float, sy: float) -> Homography:
    return Homography(
        matrix=[[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        method="test",
        inlier_count=4,
        reprojection_error_px=0.0,
    )


def test_identity_is_identity() -> None:
    assert pixel_to_pitch((10.0, 20.0), _identity()) == (10.0, 20.0)


def test_translation() -> None:
    assert pixel_to_pitch((0.0, 0.0), _translation(5.0, -3.0)) == (5.0, -3.0)
    assert pixel_to_pitch((10.0, 10.0), _translation(5.0, -3.0)) == (15.0, 7.0)


def test_scale() -> None:
    # A scale homography maps pixels to pitch-meter coordinates.
    assert pixel_to_pitch((100.0, 50.0), _scale(0.1, 0.2)) == (10.0, 10.0)


def test_degenerate_homography_raises_zero_division() -> None:
    degenerate = Homography(
        matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        method="test",
        inlier_count=0,
        reprojection_error_px=99.0,
    )
    with pytest.raises(ZeroDivisionError):
        pixel_to_pitch((1.0, 1.0), degenerate)


def test_box_anchors_under_identity() -> None:
    box = (100.0, 200.0, 200.0, 400.0)
    anchors = pixel_box_to_pitch_anchors(box, _identity())
    assert anchors["center"] == (150.0, 300.0)
    assert anchors["foot"] == (150.0, 400.0)
    assert anchors["top_left"] == (100.0, 200.0)
    assert anchors["bottom_right"] == (200.0, 400.0)


def test_box_foot_is_below_centroid_in_image_space() -> None:
    """Foot anchor should sit at the bottom edge midpoint, not the centroid."""
    box = (50.0, 100.0, 150.0, 300.0)
    anchors = pixel_box_to_pitch_anchors(box, _identity())
    center_y = anchors["center"][1]
    foot_y = anchors["foot"][1]
    assert foot_y > center_y, "foot should sit lower in pixel space than centroid"
