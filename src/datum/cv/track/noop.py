"""Noop tracker.

Assigns a fresh `track_id` to every detection on every frame. No identity
persists across frames. Used for:

  * unit-testing the track pipeline without ByteTrack's complexity
  * sanity-checking the writer, reader, and idempotency paths
  * establishing a worst-case baseline that any real tracker should beat
"""

from __future__ import annotations

from datum.cv.detect import Detection
from datum.cv.track import TrackObservation, Tracker, register


@register("track-noop")
class NoopTracker(Tracker):
    """Assigns a fresh track_id to every detection on every frame.

    Use only for plumbing tests. `unique_tracks` will roughly equal
    `tracked_observations`, which is the worst possible tracker.
    """

    def __init__(self) -> None:
        self._next_id: int = 0

    def update(
        self,
        detections: list[Detection],
        *,
        frame_idx: int,
    ) -> list[TrackObservation]:
        result: list[TrackObservation] = []
        for det in detections:
            result.append(
                TrackObservation(
                    track_id=self._next_id,
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    confidence=det.confidence,
                    class_id=det.class_id,
                )
            )
            self._next_id += 1
        return result

    def reset(self) -> None:
        self._next_id = 0


__all__ = ["NoopTracker"]
