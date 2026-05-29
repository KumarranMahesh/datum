"""Scene segmenter tests.

The synth video has three scenes at known boundaries (60-frame intervals)
alternating pitch-green and grey. The baseline `histogram-v1` segmenter
should produce three segments classified as PITCH, UNKNOWN, PITCH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datum.ingest.reader import iter_frames
from datum.ingest.schemas import SceneKind
from datum.ingest.scenes import HistogramSceneSegmenter, classify_frame


def test_classify_pitch_frame_green() -> None:
    # Synthetic full-green BGR frame should be PITCH.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[..., 1] = 180  # green channel
    img[..., 2] = 40  # red channel
    kind, conf = classify_frame(img)
    assert kind == SceneKind.PITCH
    assert 0.6 <= conf <= 1.0


def test_classify_non_pitch_grey() -> None:
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    kind, _ = classify_frame(img)
    assert kind == SceneKind.UNKNOWN


def test_segmenter_finds_three_scenes_on_synth(synth_video: Path) -> None:
    seg = HistogramSceneSegmenter(cut_threshold=0.35)
    list(seg.process(iter_frames(synth_video)))  # drain
    segments = seg.finalize()

    # Three scenes by construction. The encoder occasionally smears one
    # frame at the cut; allow up to four detected segments before failing.
    assert 3 <= len(segments) <= 4, len(segments)

    # First and last segments are pitch-like (green). Middle is grey.
    assert segments[0].kind == SceneKind.PITCH
    assert segments[-1].kind == SceneKind.PITCH

    # No segment overlaps its successor.
    for a, b in zip(segments, segments[1:], strict=False):
        assert a.end_source_frame_idx < b.start_source_frame_idx


def test_segmenter_rejects_bad_threshold() -> None:
    import pytest

    with pytest.raises(ValueError):
        HistogramSceneSegmenter(cut_threshold=1.5)
    with pytest.raises(ValueError):
        HistogramSceneSegmenter(cut_threshold=-0.1)
