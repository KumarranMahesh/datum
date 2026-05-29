"""Scene cut detection and per-frame shot classification.

Baseline (`histogram-v1`) is intentionally simple:

  * cuts: hue-histogram Bhattacharyya distance threshold
  * class: green-pixel ratio. > 30% green is PITCH, everything else
    UNKNOWN. CROWD / BENCH / GRAPHIC / REPLAY are reserved for future
    classifiers and not emitted by this baseline.

Downstream stages only need PITCH vs not-PITCH for filtering, so the
sparse class space is sufficient for 0.1. Better classifiers register
under new `scene_classifier` identifiers (e.g. 'cnn-v1') and the pipeline
dispatches by identifier.

Cost is roughly 1-2 ms per 1080p frame on a modern CPU. No model state,
no GPU. The segmenter is a single class because the cut detector and the
per-segment classifier share state (the running vote counter).
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from datum.ingest.reader import DecodedFrame
from datum.ingest.schemas import SceneKind, SceneSegment

# Hue range for grass on standard broadcasts. OpenCV's HSV hue is 0..180.
# Saturation and value floors filter out near-grey shadow regions that
# decay into the green band under low-light conditions (night matches).
_GREEN_HUE_LO: int = 35
_GREEN_HUE_HI: int = 85
_GREEN_SAT_MIN: int = 40
_GREEN_VAL_MIN: int = 30

# Above this fraction of green pixels the frame is overwhelmingly likely
# to be a wide pitch shot. Lower bound is permissive; broadcasts often
# pan the camera off the pitch briefly.
_PITCH_GREEN_RATIO: float = 0.30

_HUE_BINS: int = 30  # 30 bins across 0..180 hue is plenty for cut detection


def _hue_histogram(image: np.ndarray) -> np.ndarray:
    """Return a normalised 30-bin hue histogram for a BGR frame.

    Saturation and value are dropped on purpose. Cut detection cares about
    *colour composition*, not lighting. S/V vary within a single shot more
    than across cuts.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [_HUE_BINS], [0, 180])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def classify_frame(image: np.ndarray) -> tuple[SceneKind, float]:
    """Classify a single BGR frame.

    Returns (kind, confidence). The baseline only emits PITCH and UNKNOWN;
    everything else is reserved for future classifiers.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    green_mask = (
        (h >= _GREEN_HUE_LO)
        & (h <= _GREEN_HUE_HI)
        & (s >= _GREEN_SAT_MIN)
        & (v >= _GREEN_VAL_MIN)
    )
    green_ratio = float(green_mask.mean())

    if green_ratio > _PITCH_GREEN_RATIO:
        # Confidence rises with how far above the threshold the frame sits.
        # Capped at 1.0 for the high-end edge case of a fully-grass frame.
        confidence = min(1.0, 0.6 + (green_ratio - _PITCH_GREEN_RATIO) / _PITCH_GREEN_RATIO)
        return SceneKind.PITCH, confidence

    # Not enough green to call it pitch. The baseline does not try to
    # distinguish CROWD vs BENCH vs GRAPHIC; that needs a real classifier.
    return SceneKind.UNKNOWN, 0.5


class HistogramSceneSegmenter:
    """Streaming scene cut detector and per-segment classifier.

    Use:

        seg = HistogramSceneSegmenter(cut_threshold=0.35)
        for frame, scene_id, kind in seg.process(frames):
            ...
        segments = seg.finalize()

    The segmenter must be drained before `finalize()` is called; otherwise
    the trailing segment is not emitted.
    """

    def __init__(self, *, cut_threshold: float) -> None:
        if not 0.0 <= cut_threshold <= 1.0:
            raise ValueError(f"cut_threshold must be in [0, 1]; got {cut_threshold}")
        self._cut_threshold = cut_threshold
        self._prev_hist: np.ndarray | None = None
        self._current_scene_id: int = 0
        self._segment_start_src_idx: int = 0
        self._last_src_idx: int = -1
        # vote counter for the modal-class-per-segment decision
        self._kind_votes: dict[SceneKind, int] = {}
        self._segments: list[SceneSegment] = []

    def process(
        self, frames: Iterator[DecodedFrame]
    ) -> Iterator[tuple[DecodedFrame, int, SceneKind]]:
        """Yield (frame, scene_id, frame_kind) for every input frame.

        `frame_kind` is the per-frame classifier output. The per-segment
        modal kind is computed in `finalize()`.
        """
        for frame in frames:
            hist = _hue_histogram(frame.image)
            kind, _ = classify_frame(frame.image)

            if self._prev_hist is not None and self._is_cut(hist):
                self._close_segment(self._last_src_idx)
                self._current_scene_id += 1
                self._segment_start_src_idx = frame.source_frame_idx
                self._kind_votes = {}

            self._prev_hist = hist
            self._kind_votes[kind] = self._kind_votes.get(kind, 0) + 1
            self._last_src_idx = frame.source_frame_idx

            yield frame, self._current_scene_id, kind

    def finalize(self) -> list[SceneSegment]:
        """Emit the trailing segment and return all segments."""
        if self._last_src_idx >= 0 and self._kind_votes:
            self._close_segment(self._last_src_idx)
        return list(self._segments)

    def _is_cut(self, hist: np.ndarray) -> bool:
        # Bhattacharyya is in [0, 1]; 0 is identical, 1 is no overlap.
        # Type guard for mypy: _prev_hist is checked non-None by caller.
        assert self._prev_hist is not None
        dist = float(cv2.compareHist(hist, self._prev_hist, cv2.HISTCMP_BHATTACHARYYA))
        return dist > self._cut_threshold

    def _close_segment(self, end_src_idx: int) -> None:
        if not self._kind_votes:
            return
        modal = max(self._kind_votes, key=lambda k: self._kind_votes[k])
        total = sum(self._kind_votes.values())
        confidence = self._kind_votes[modal] / total
        self._segments.append(
            SceneSegment(
                scene_id=self._current_scene_id,
                start_source_frame_idx=self._segment_start_src_idx,
                end_source_frame_idx=end_src_idx,
                kind=modal,
                confidence=confidence,
            )
        )


__all__ = [
    "HistogramSceneSegmenter",
    "classify_frame",
]
