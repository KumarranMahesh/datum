"""ByteTrack tracker tests.

These exercise the association algorithm against synthetic detection
streams so behaviour is deterministic and the tests stay GPU-free.
"""

from __future__ import annotations

from datum.cv.detect import Detection
from datum.cv.track import Tracker, get
from datum.cv.track.bytetrack import ByteTrackTracker


def _det(
    x1: float = 100.0,
    y1: float = 100.0,
    w: float = 50.0,
    h: float = 100.0,
    conf: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(
        x1=x1,
        y1=y1,
        x2=x1 + w,
        y2=y1 + h,
        confidence=conf,
        class_id=class_id,
    )


def _make() -> Tracker:
    return get("bytetrack")()


def test_bytetrack_is_registered() -> None:
    cls = get("bytetrack")
    assert cls.name == "bytetrack"


def test_stationary_detection_keeps_single_track_id() -> None:
    tracker = _make()
    ids: list[int] = []
    for frame_idx in range(10):
        out = tracker.update([_det(x1=100, y1=100)], frame_idx=frame_idx)
        assert len(out) == 1
        ids.append(out[0].track_id)
    assert len(set(ids)) == 1


def test_slowly_moving_detection_keeps_single_track_id() -> None:
    tracker = _make()
    ids: list[int] = []
    for frame_idx in range(10):
        # Drift 5 pixels per frame on a 50x100 box. IoU(prev, next) stays
        # high regardless of match_thresh; the camera-shift estimator also
        # locks onto the linear motion within a couple of frames.
        out = tracker.update(
            [_det(x1=100 + 5 * frame_idx, y1=100)], frame_idx=frame_idx
        )
        assert len(out) == 1
        ids.append(out[0].track_id)
    assert len(set(ids)) == 1


def test_two_disjoint_detections_get_distinct_track_ids() -> None:
    tracker = _make()
    out = tracker.update(
        [_det(x1=100, y1=100), _det(x1=500, y1=400)], frame_idx=0
    )
    assert len(out) == 2
    assert out[0].track_id != out[1].track_id


def test_low_confidence_alone_does_not_create_new_track() -> None:
    # Default new_track_thresh is 0.4; a single conf=0.15 detection should
    # not start a track on a frame with no existing tracks.
    tracker = _make()
    out = tracker.update([_det(conf=0.15)], frame_idx=0)
    assert out == []


def test_borderline_high_confidence_creates_track() -> None:
    # conf=0.45 sits just above the default new_track_thresh of 0.4.
    tracker = _make()
    out = tracker.update([_det(conf=0.45)], frame_idx=0)
    assert len(out) == 1


def test_low_confidence_can_extend_existing_track() -> None:
    # ByteTrack's defining behaviour: low-conf detections cannot start
    # tracks, but they can keep an existing one alive.
    tracker = _make()
    a = tracker.update([_det(x1=100, y1=100, conf=0.9)], frame_idx=0)
    b = tracker.update([_det(x1=100, y1=100, conf=0.15)], frame_idx=1)
    assert len(a) == len(b) == 1
    assert a[0].track_id == b[0].track_id


def test_track_dies_after_max_age_then_new_detection_makes_new_id() -> None:
    # Use a small max_age so the test does not depend on the default,
    # which is tuned for broadcast and may change with footage.
    tracker = get("bytetrack")(max_age=5)
    first = tracker.update([_det(x1=100, y1=100)], frame_idx=0)
    # Provide zero detections for max_age + 1 frames so the original track
    # is reaped before it can be matched.
    for frame_idx in range(1, 10):
        tracker.update([], frame_idx=frame_idx)
    revived = tracker.update([_det(x1=100, y1=100)], frame_idx=10)
    assert first[0].track_id != revived[0].track_id


def test_reset_clears_state() -> None:
    tracker = _make()
    a = tracker.update([_det(x1=100, y1=100)], frame_idx=0)
    tracker.reset()
    b = tracker.update([_det(x1=100, y1=100)], frame_idx=0)
    # After reset the id counter restarts; the same first detection should
    # match track_id == 0 again.
    assert a[0].track_id == b[0].track_id == 0


def test_camera_motion_compensation_keeps_panning_scene_together() -> None:
    """Simulate a steady pan small enough that frame 1 still has IoU overlap.

    Three "players" sit at fixed world positions. Each frame, every
    detection shifts by 15 px in x. That is enough to defeat a tracker
    with no motion model after a couple of frames, but small enough that
    the first frame transition still has IoU > match_thresh, so the
    matched-pair estimator bootstraps itself from frame 1's matches and
    holds identities from frame 2 onwards.

    Larger pans (frame-to-frame motion > box width) currently fragment
    the first transition after every reset. See the LIMITATION block at
    the top of bytetrack.py.
    """
    tracker = _make()

    world_positions = [(200, 200), (500, 300), (900, 250)]
    camera_per_frame = 15  # px in x; well below box_width / 2 = 25

    ids_per_frame: list[set[int]] = []
    for frame_idx in range(8):
        dets = [
            _det(x1=wx + camera_per_frame * frame_idx, y1=wy, conf=0.9)
            for (wx, wy) in world_positions
        ]
        out = tracker.update(dets, frame_idx=frame_idx)
        ids_per_frame.append({o.track_id for o in out})

    # Three tracks should be present every frame.
    for frame_idx in range(8):
        assert len(ids_per_frame[frame_idx]) == 3, (
            f"frame {frame_idx} lost a track: {ids_per_frame[frame_idx]}"
        )

    # IDs are stable across the whole sequence.
    steady_state_ids = ids_per_frame[0]
    for frame_idx in range(1, 8):
        assert ids_per_frame[frame_idx] == steady_state_ids, (
            f"frame {frame_idx} swapped IDs: was {steady_state_ids}, "
            f"now {ids_per_frame[frame_idx]}"
        )


def test_camera_shift_estimate_initialises_to_zero() -> None:
    """A freshly constructed tracker has no shift estimate yet."""
    tracker = ByteTrackTracker()
    # Private attribute peek is intentional; this is a property of the
    # algorithm worth pinning.
    assert tracker._camera_shift.tolist() == [0.0, 0.0]  # noqa: SLF001
    tracker.update([_det()], frame_idx=0)
    # Single match, below min_matches_for_shift=3, so estimate stays zero.
    assert tracker._camera_shift.tolist() == [0.0, 0.0]  # noqa: SLF001


def test_reset_clears_camera_shift() -> None:
    tracker = ByteTrackTracker()
    # Build up a non-zero shift estimate.
    for frame_idx in range(4):
        tracker.update(
            [
                _det(x1=100 + 30 * frame_idx, y1=100, conf=0.9),
                _det(x1=400 + 30 * frame_idx, y1=200, conf=0.9),
                _det(x1=700 + 30 * frame_idx, y1=300, conf=0.9),
            ],
            frame_idx=frame_idx,
        )
    assert tracker._camera_shift[0] > 0  # noqa: SLF001
    tracker.reset()
    assert tracker._camera_shift.tolist() == [0.0, 0.0]  # noqa: SLF001
