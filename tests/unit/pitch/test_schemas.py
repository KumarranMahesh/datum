"""Pitch schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datum.cv.pitch.schemas import (
    FramePitch,
    Homography,
    PitchSolverConfig,
)


def _identity_matrix() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_homography_accepts_identity_matrix() -> None:
    h = Homography(
        matrix=_identity_matrix(),
        method="noop",
        inlier_count=0,
        reprojection_error_px=0.0,
    )
    assert h.matrix[1][1] == 1.0


def test_homography_rejects_non_3x3_matrix() -> None:
    with pytest.raises(ValidationError):
        Homography(
            matrix=[[1.0, 0.0], [0.0, 1.0]],
            method="noop",
            inlier_count=0,
            reprojection_error_px=0.0,
        )
    with pytest.raises(ValidationError):
        Homography(
            matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # 2x3
            method="noop",
            inlier_count=0,
            reprojection_error_px=0.0,
        )


def test_homography_rejects_negative_inlier_count() -> None:
    with pytest.raises(ValidationError):
        Homography(
            matrix=_identity_matrix(),
            method="noop",
            inlier_count=-1,
            reprojection_error_px=0.0,
        )


def test_homography_rejects_empty_method() -> None:
    with pytest.raises(ValidationError):
        Homography(
            matrix=_identity_matrix(),
            method="",
            inlier_count=0,
            reprojection_error_px=0.0,
        )


def test_pitch_solver_config_requires_solver() -> None:
    with pytest.raises(ValidationError):
        PitchSolverConfig()  # type: ignore[call-arg]


def test_pitch_solver_config_defaults() -> None:
    cfg = PitchSolverConfig(solver="pitch-noop")
    assert cfg.solver == "pitch-noop"
    assert cfg.solver_config == {}
    assert cfg.pitch_only is True


def test_frame_pitch_allows_none_homography() -> None:
    record = FramePitch(
        frame_idx=0, source_frame_idx=0, homography=None, solver_wall_ms=1.5
    )
    assert record.homography is None
