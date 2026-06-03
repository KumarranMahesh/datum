"""Pixel-to-pitch coordinate transforms.

Pure functions that consume a 3x3 image-to-pitch homography and apply it
to pixel-space points or bboxes. Lives in `datum.spatial` because it is
the operation every downstream stage (features, embeddings, search) will
reach for once pitch homographies start landing.

Coordinate conventions match `datum.cv.pitch.schemas.Homography`: pitch
origin at the centre spot, x along the long axis, y along the short axis,
metres throughout.
"""

from __future__ import annotations

import numpy as np

from datum.cv.pitch.schemas import Homography


def _to_numpy(homography: Homography) -> np.ndarray:
    return np.asarray(homography.matrix, dtype=np.float64)


def pixel_to_pitch(
    point: tuple[float, float],
    homography: Homography,
) -> tuple[float, float]:
    """Map a single pixel (u, v) to pitch coordinates (x, y) in metres.

    Standard projective transform: H @ [u, v, 1].T -> [X, Y, W].T,
    divided by W to recover the pitch-plane coordinate.

    Raises ZeroDivisionError if W collapses to zero (point at infinity).
    """
    h = _to_numpy(homography)
    u, v = point
    vec = np.array([u, v, 1.0], dtype=np.float64)
    x_w, y_w, w = h @ vec
    if w == 0.0:
        raise ZeroDivisionError(
            f"projective divisor collapsed to zero for pixel {point}; "
            "this point projects to infinity under the given homography"
        )
    return float(x_w / w), float(y_w / w)


def pixel_box_to_pitch_anchors(
    box: tuple[float, float, float, float],
    homography: Homography,
) -> dict[str, tuple[float, float]]:
    """Map a pixel-space xyxy box to pitch-space reference points.

    Returns a dict with four named anchors so callers can pick the one
    that makes sense for the consumer:

      * 'center'      box centroid in pitch coords
      * 'foot'        midpoint of the bottom edge (player's feet, the
                      anatomically correct anchor for grounding a
                      detection onto the pitch plane)
      * 'top_left'    pitch coord of the bbox's top-left corner
      * 'bottom_right' pitch coord of the bbox's bottom-right corner

    For player detections, 'foot' is almost always the right anchor:
    bounding-box centres sit at the player's torso, which is above the
    pitch plane and not consistent with the homography assumption that
    everything lies on a single plane.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    foot_x = (x1 + x2) / 2.0
    foot_y = y2  # bottom of the bbox = player's feet for a standing person
    return {
        "center": pixel_to_pitch((cx, cy), homography),
        "foot": pixel_to_pitch((foot_x, foot_y), homography),
        "top_left": pixel_to_pitch((x1, y1), homography),
        "bottom_right": pixel_to_pitch((x2, y2), homography),
    }


__all__ = ["pixel_box_to_pitch_anchors", "pixel_to_pitch"]
