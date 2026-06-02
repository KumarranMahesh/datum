"""ByteTrack tracker with camera-motion compensation.

Implements ByteTrack's association algorithm:

  1. Predict each active track's bbox by applying the last per-frame
     camera shift estimate.
  2. Split detections into high-confidence (>= track_high_thresh) and
     low-confidence (>= track_low_thresh but < track_high_thresh).
  3. First association: high-confidence detections vs all active tracks
     via IoU + Hungarian.
  4. Second association: still-unmatched tracks vs low-confidence
     detections. Low-conf detections cannot spawn new tracks but can keep
     existing tracks alive through brief occlusions.
  5. Unmatched high-confidence detections above `new_track_thresh` become
     new tracks.
  6. Tracks unmatched for `max_age` frames are removed.
  7. Re-estimate per-frame camera shift from the matched pairs in this
     frame. Median of (detection_centroid - previous_track_centroid) over
     all matches, EMA-smoothed across frames.

The camera-shift estimator helps within a scene once it has bootstrapped
from a few matched pairs. Tracks survive normal smooth broadcast pans
cleanly. The current implementation has two known weaknesses on real
broadcast video:

  * Bootstrap. The very first frame transition inside a scene (initial
    or post-reset) has shift=0 and no prior matches, so a pan larger
    than the box width loses every track. An earlier attempt at bootstrap
    via raw centroid medians was reverted: the median is contaminated by
    independent player motion on real footage and produced *more* ID
    fragmentation than no bootstrap at all.
  * Player-motion contamination. The matched-pair shift estimate is
    `median(detection_centroid - track_centroid)`, which is the camera
    shift only when player motion averages to zero. On a counter-attack
    or set piece the median drifts away from the true camera motion.

The principled fix for both is optical flow on background features,
which needs the source image plumbed into the tracker. Tracked as a
future improvement; the current behaviour is acceptable for the v0.1
"naive tracking" milestone.

Reference: "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box" (Zhang et al., ECCV 2022).

License note: the upstream ByteTrack code is MIT. This is a from-scratch
reimplementation that uses no upstream code, so the same MIT-compatible
terms of this repository apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

from datum.cv.detect import Detection
from datum.cv.track import TrackObservation, Tracker, register


@dataclass(slots=True)
class _Track:
    """One active or lost track.

    Bbox carried internally in [cx, cy, w, h] for centroid math; converted
    to [x1, y1, x2, y2] at the I/O boundary.
    """

    track_id: int
    class_id: int
    bbox: np.ndarray  # (4,) [cx, cy, w, h]
    last_confidence: float
    last_frame_idx: int
    hits: int = 1
    state: str = "active"  # "active" | "lost"

    def predict(self, current_frame_idx: int, camera_shift: np.ndarray) -> np.ndarray:
        """Predict bbox at current_frame_idx, accounting for camera motion.

        `camera_shift` is the per-frame (dx, dy) of the camera as estimated
        from recent matches. Applied linearly over the elapsed frame count.

        The pipeline passes contiguous tracking-frame indices (0, 1, 2, ...)
        that reset at scene boundaries, so dt is naturally small: 1 for
        active tracks, up to max_age for temporarily lost tracks.  No
        clamping is needed.

        Width and height are not extrapolated.
        """
        dt = current_frame_idx - self.last_frame_idx
        predicted = self.bbox.copy()
        if dt > 0:
            predicted[0] += camera_shift[0] * dt
            predicted[1] += camera_shift[1] * dt
        return predicted

    def update(
        self,
        det_bbox_cwh: np.ndarray,
        *,
        confidence: float,
        frame_idx: int,
    ) -> None:
        self.bbox = det_bbox_cwh
        self.last_confidence = confidence
        self.last_frame_idx = frame_idx
        self.hits += 1
        self.state = "active"


def _xyxy_to_cwh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64
    )


def _cwh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64
    )


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _iou_matrix(
    track_boxes: list[np.ndarray], det_boxes: list[np.ndarray]
) -> np.ndarray:
    if not track_boxes or not det_boxes:
        return np.zeros((len(track_boxes), len(det_boxes)))
    mat = np.zeros((len(track_boxes), len(det_boxes)))
    for i, t in enumerate(track_boxes):
        for j, d in enumerate(det_boxes):
            mat[i, j] = _iou(t, d)
    return mat


def _associate(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    n_rows, n_cols = iou_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    cost = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_ind, col_ind, strict=True):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


@register("bytetrack")
class ByteTrackTracker(Tracker):
    """ByteTrack association with median-pair camera motion compensation.

    Constructor knobs (override-able via tracker_config in YAML):

      * track_high_thresh:    high-confidence detection cutoff  (0.4)
      * track_low_thresh:     low-confidence detection floor    (0.1)
      * new_track_thresh:     min confidence to start a track   (0.4)
      * match_thresh:         IoU floor for any association     (0.1)
      * max_age:              frames lost tracks survive        (60)
      * min_hits:             min hits before emitting          (1)
      * camera_motion_alpha:  EMA factor for camera-shift smoothing.
                              0 = no smoothing (use this frame's raw estimate)
                              1 = no update (always last frame's estimate)
                              (0.5)
      * min_matches_for_shift: matched pairs needed before the camera-shift
                               estimate is updated. Small numbers are noisy;
                               2 is the floor.                  (3)
    """

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.4,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.4,
        match_thresh: float = 0.1,
        max_age: int = 60,
        min_hits: int = 1,
        camera_motion_alpha: float = 0.5,
        min_matches_for_shift: int = 3,
    ) -> None:
        if not 0.0 <= track_low_thresh <= track_high_thresh <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= low <= high <= 1; "
                f"got low={track_low_thresh} high={track_high_thresh}"
            )
        if not 0.0 <= camera_motion_alpha <= 1.0:
            raise ValueError(
                f"camera_motion_alpha must be in [0, 1]; got {camera_motion_alpha}"
            )

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.camera_motion_alpha = camera_motion_alpha
        self.min_matches_for_shift = min_matches_for_shift

        self._tracks: list[_Track] = []
        self._id_counter = count(0)
        self._camera_shift = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self._tracks = []
        self._id_counter = count(0)
        self._camera_shift = np.zeros(2, dtype=np.float64)

    def update(
        self,
        detections: list[Detection],
        *,
        frame_idx: int,
    ) -> list[TrackObservation]:
        high_dets = [
            d for d in detections if d.confidence >= self.track_high_thresh
        ]
        low_dets = [
            d
            for d in detections
            if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        # Capture per-track state BEFORE the in-place update, so the camera
        # shift estimator can compare matched detection centroids to the
        # track's actual previous-frame centroid.
        prev_bboxes = [t.bbox.copy() for t in self._tracks]
        prev_frame_indices = [t.last_frame_idx for t in self._tracks]

        # Predictions for matching: bbox + camera_shift * dt. The shift is
        # zero for the very first frame after a reset, which means the
        # first transition after every scene boundary loses some tracks to
        # camera motion. See the LIMITATION note at the top of this file.
        predicted_xyxy = [
            _cwh_to_xyxy(t.predict(frame_idx, self._camera_shift))
            for t in self._tracks
        ]
        high_xyxy = [np.array([d.x1, d.y1, d.x2, d.y2]) for d in high_dets]
        low_xyxy = [np.array([d.x1, d.y1, d.x2, d.y2]) for d in low_dets]

        # First association: high-confidence detections vs all tracks.
        iou1 = _iou_matrix(predicted_xyxy, high_xyxy)
        matches_h, unmatched_tracks_h, unmatched_high = _associate(
            iou1, self.match_thresh
        )

        for track_idx, det_idx in matches_h:
            det = high_dets[det_idx]
            self._tracks[track_idx].update(
                _xyxy_to_cwh(np.array([det.x1, det.y1, det.x2, det.y2])),
                confidence=det.confidence,
                frame_idx=frame_idx,
            )

        # Second association: still-unmatched tracks vs low-confidence dets.
        remaining_track_xyxy = [predicted_xyxy[i] for i in unmatched_tracks_h]
        iou2 = _iou_matrix(remaining_track_xyxy, low_xyxy)
        matches_l, unmatched_tracks_l_local, _ = _associate(
            iou2, self.match_thresh
        )

        # Collect (global_track_idx, Detection) for all stage-2 matches so
        # the camera-shift estimator below can read the matched detection
        # without re-translating local indices.
        stage2_matches: list[tuple[int, Detection]] = []
        for local_idx, det_idx in matches_l:
            global_idx = unmatched_tracks_h[local_idx]
            det = low_dets[det_idx]
            stage2_matches.append((global_idx, det))
            self._tracks[global_idx].update(
                _xyxy_to_cwh(np.array([det.x1, det.y1, det.x2, det.y2])),
                confidence=det.confidence,
                frame_idx=frame_idx,
            )

        # Mark tracks that matched in neither pass as lost.
        truly_unmatched = [unmatched_tracks_h[i] for i in unmatched_tracks_l_local]
        for track_idx in truly_unmatched:
            self._tracks[track_idx].state = "lost"

        # Estimate the per-frame camera shift from this frame's matches.
        # delta = (new_centroid - prev_centroid) / dt, in pixels per frame.
        deltas_x: list[float] = []
        deltas_y: list[float] = []
        for track_idx, det_idx in matches_h:
            prev = prev_bboxes[track_idx]
            det = high_dets[det_idx]
            det_cwh = _xyxy_to_cwh(
                np.array([det.x1, det.y1, det.x2, det.y2])
            )
            dt = max(1, frame_idx - prev_frame_indices[track_idx])
            deltas_x.append((det_cwh[0] - prev[0]) / dt)
            deltas_y.append((det_cwh[1] - prev[1]) / dt)
        for global_idx, det in stage2_matches:
            prev = prev_bboxes[global_idx]
            det_cwh = _xyxy_to_cwh(
                np.array([det.x1, det.y1, det.x2, det.y2])
            )
            dt = max(1, frame_idx - prev_frame_indices[global_idx])
            deltas_x.append((det_cwh[0] - prev[0]) / dt)
            deltas_y.append((det_cwh[1] - prev[1]) / dt)

        if len(deltas_x) >= self.min_matches_for_shift:
            new_shift_x = float(np.median(deltas_x))
            new_shift_y = float(np.median(deltas_y))
            a = self.camera_motion_alpha
            self._camera_shift = np.array(
                [
                    a * self._camera_shift[0] + (1 - a) * new_shift_x,
                    a * self._camera_shift[1] + (1 - a) * new_shift_y,
                ],
                dtype=np.float64,
            )

        # New tracks from confident unmatched high-confidence detections.
        for det_idx in unmatched_high:
            det = high_dets[det_idx]
            if det.confidence < self.new_track_thresh:
                continue
            self._tracks.append(
                _Track(
                    track_id=next(self._id_counter),
                    class_id=det.class_id,
                    bbox=_xyxy_to_cwh(
                        np.array([det.x1, det.y1, det.x2, det.y2])
                    ),
                    last_confidence=det.confidence,
                    last_frame_idx=frame_idx,
                )
            )

        # Drop tracks too old to keep.
        self._tracks = [
            t
            for t in self._tracks
            if (frame_idx - t.last_frame_idx) <= self.max_age
        ]

        # Emit observations for tracks matched this frame with enough hits.
        observations: list[TrackObservation] = []
        for track in self._tracks:
            if track.last_frame_idx != frame_idx:
                continue
            if track.hits < self.min_hits:
                continue
            xyxy = _cwh_to_xyxy(track.bbox)
            observations.append(
                TrackObservation(
                    track_id=track.track_id,
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=track.last_confidence,
                    class_id=track.class_id,
                )
            )
        return observations


__all__ = ["ByteTrackTracker"]
