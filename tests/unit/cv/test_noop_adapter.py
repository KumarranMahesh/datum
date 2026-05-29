"""Noop detector adapter tests.

The noop adapter is a smoke harness, so the tests are correspondingly
shallow. The important thing is that the registry resolves it and the
shape of DetectionBatch is honoured.
"""

from __future__ import annotations

import numpy as np

from datum.cv.detect import get


def test_noop_is_registered() -> None:
    cls = get("noop")
    assert cls.name == "noop"


def test_noop_returns_empty_per_frame() -> None:
    cls = get("noop")
    detector = cls()

    frames = np.zeros((3, 100, 100, 3), dtype=np.uint8)
    result = detector.detect(frames)

    assert len(result.per_frame) == 3
    assert all(per == [] for per in result.per_frame)


def test_noop_class_map_is_empty() -> None:
    cls = get("noop")
    detector = cls()
    assert detector.class_map == {}


def test_empty_batch_returns_empty_result() -> None:
    cls = get("noop")
    detector = cls()
    frames = np.zeros((0, 100, 100, 3), dtype=np.uint8)
    result = detector.detect(frames)
    assert result.per_frame == []
