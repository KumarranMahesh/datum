"""Noop detector.

Returns no detections for any frame. Used for:

  * unit-testing the CV pipeline without downloading model weights
  * sanity-checking the writer, reader, and idempotency paths
  * benchmarking pipeline overhead exclusive of model inference
"""

from __future__ import annotations

from datum.cv.detect import (
    DetectionBatch,
    Detector,
    FrameBatch,
    register,
)


@register("noop")
class NoopDetector(Detector):
    """Returns an empty detection list for every frame in the batch."""

    @property
    def class_map(self) -> dict[int, str]:
        return {}

    def detect(self, frames: FrameBatch) -> DetectionBatch:
        return DetectionBatch(per_frame=[[] for _ in range(len(frames))])


__all__ = ["NoopDetector"]
