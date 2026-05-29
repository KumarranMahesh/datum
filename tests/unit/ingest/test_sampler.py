"""Sampler tests.

These exercise the resampling math in isolation. DecodedFrame is
constructed by hand to avoid pulling in a real decoder for unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from datum.ingest.reader import DecodedFrame
from datum.ingest.sampler import sample_stream


def _fake_frames(n: int) -> list[DecodedFrame]:
    """Cheap DecodedFrames. Pixel content is irrelevant to the sampler."""
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)
    return [
        DecodedFrame(source_frame_idx=i, pts_us=i * 1000, image=dummy)
        for i in range(n)
    ]


def test_integer_ratio_keeps_every_nth() -> None:
    out = list(sample_stream(_fake_frames(60), source_fps=30.0, sample_fps=5.0))
    # 60 frames at 30 fps to 5 fps -> 10 frames at indices 0, 6, 12, ...
    assert [f.source_frame_idx for f in out] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


def test_source_lower_than_target_is_passthrough() -> None:
    src = _fake_frames(10)
    out = list(sample_stream(src, source_fps=5.0, sample_fps=30.0))
    assert [f.source_frame_idx for f in out] == [f.source_frame_idx for f in src]


def test_equal_rates_is_passthrough() -> None:
    src = _fake_frames(10)
    out = list(sample_stream(src, source_fps=25.0, sample_fps=25.0))
    assert len(out) == len(src)


def test_non_integer_ratio_converges_to_target_rate() -> None:
    # NTSC-ish: 24000/1001 to 5 fps. Over a long enough stream the count
    # should be close to duration * sample_fps. 10 seconds of source is
    # 240 source frames, expect ~50 sampled.
    out = list(
        sample_stream(_fake_frames(240), source_fps=24000.0 / 1001.0, sample_fps=5.0)
    )
    assert 49 <= len(out) <= 51, len(out)


def test_invalid_rates_raise() -> None:
    with pytest.raises(ValueError):
        list(sample_stream(_fake_frames(1), source_fps=30.0, sample_fps=0.0))
    with pytest.raises(ValueError):
        list(sample_stream(_fake_frames(1), source_fps=0.0, sample_fps=5.0))


def test_deterministic() -> None:
    """Same input + config produces the same output. No hidden state."""
    a = list(sample_stream(_fake_frames(100), source_fps=30.0, sample_fps=7.0))
    b = list(sample_stream(_fake_frames(100), source_fps=30.0, sample_fps=7.0))
    assert [f.source_frame_idx for f in a] == [f.source_frame_idx for f in b]
