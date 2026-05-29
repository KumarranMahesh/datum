"""CV schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datum.cv import CvConfig, DetectionBox


def test_cv_config_requires_detector() -> None:
    with pytest.raises(ValidationError):
        CvConfig()  # type: ignore[call-arg]


def test_cv_config_defaults() -> None:
    cfg = CvConfig(detector="noop")
    assert cfg.detector == "noop"
    assert cfg.batch_size == 8
    assert cfg.pitch_only is True
    assert cfg.detector_config == {}


def test_cv_config_rejects_zero_batch() -> None:
    with pytest.raises(ValidationError):
        CvConfig(detector="noop", batch_size=0)


def test_detection_box_validates_confidence_range() -> None:
    with pytest.raises(ValidationError):
        DetectionBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            confidence=1.5,
            class_id=0,
            class_name="person",
        )


def test_detection_box_rejects_empty_class_name() -> None:
    with pytest.raises(ValidationError):
        DetectionBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            confidence=0.5,
            class_id=0,
            class_name="",
        )
