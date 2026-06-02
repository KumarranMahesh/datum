"""Noop tracker tests.

Confirms the registry resolves the noop tracker and the contract holds:
every detection becomes its own track.
"""

from __future__ import annotations

from datum.cv.detect import Detection
from datum.cv.track import get


def _det(x: float = 100.0, conf: float = 0.9) -> Detection:
    return Detection(
        x1=x,
        y1=100.0,
        x2=x + 20.0,
        y2=200.0,
        confidence=conf,
        class_id=0,
    )


def test_noop_is_registered() -> None:
    cls = get("track-noop")
    assert cls.name == "track-noop"


def test_noop_returns_one_track_per_detection() -> None:
    tracker = get("track-noop")()
    out = tracker.update([_det(100), _det(200), _det(300)], frame_idx=0)
    assert len(out) == 3
    assert {o.track_id for o in out} == {0, 1, 2}


def test_noop_track_ids_keep_incrementing_across_frames() -> None:
    tracker = get("track-noop")()
    a = tracker.update([_det(100)], frame_idx=0)
    b = tracker.update([_det(100)], frame_idx=1)
    # Even at the same position on the next frame, noop emits a new id.
    assert a[0].track_id != b[0].track_id


def test_noop_reset_restarts_ids_from_zero() -> None:
    tracker = get("track-noop")()
    tracker.update([_det(100)], frame_idx=0)
    tracker.reset()
    out = tracker.update([_det(100)], frame_idx=0)
    assert out[0].track_id == 0


def test_noop_handles_empty_input() -> None:
    tracker = get("track-noop")()
    out = tracker.update([], frame_idx=0)
    assert out == []
