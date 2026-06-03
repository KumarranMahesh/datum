"""Pixel-to-pitch coordinate transforms.

Pure-math helpers that consume a `datum.cv.pitch.schemas.Homography` and
apply it to pixel-space points or bboxes. No I/O, no model state.
"""

from datum.spatial.transform import pixel_box_to_pitch_anchors, pixel_to_pitch

__all__ = ["pixel_box_to_pitch_anchors", "pixel_to_pitch"]
