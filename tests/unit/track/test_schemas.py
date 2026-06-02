"""Track schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datum.cv.track.schemas import FrameTracks, TrackedDetection, TrackerConfig


def test_tracker_config_requires_tracker() -> None:
    with pytest.raises(ValidationError):
        TrackerConfig()  # type: ignore[call-arg]


def test_tracker_config_defaults() -> None:
    cfg = TrackerConfig(tracker="bytetrack")
    assert cfg.tracker == "bytetrack"
    assert cfg.tracker_config == {}
    assert cfg.reset_on_scene_cut is False


def test_tracked_detection_extends_detection_box() -> None:
    td = TrackedDetection(
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
        confidence=0.9,
        class_id=0,
        class_name="person",
        track_id=42,
    )
    assert td.track_id == 42
    assert td.class_name == "person"


def test_tracked_detection_rejects_negative_track_id() -> None:
    with pytest.raises(ValidationError):
        TrackedDetection(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            confidence=0.5,
            class_id=0,
            class_name="person",
            track_id=-1,
        )


def test_frame_tracks_allows_empty_tracks() -> None:
    ft = FrameTracks(frame_idx=0, source_frame_idx=0, tracks=[])
    assert ft.tracks == []
