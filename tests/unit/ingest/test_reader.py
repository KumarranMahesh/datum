"""Reader tests.

Exercise PyAV-backed probing and decoding on the synthetic mp4. The synth
video is fully controlled (720p / 30 fps / 180 frames) so expected values
are exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datum.ingest.reader import (
    UnsupportedSourceError,
    iter_frames,
    probe_source,
    sha256_file,
)


def test_probe_source_reports_expected_metadata(synth_video: Path) -> None:
    info = probe_source(synth_video, min_resolution=720)
    assert info.width == 1280
    assert info.height == 720
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert info.video_codec == "h264"
    assert len(info.sha256) == 64
    # 180 frames at 30 fps = 6 seconds. Allow some slack for container overhead.
    assert 5.5 <= info.duration_s <= 6.5


def test_probe_source_refuses_below_min_resolution(synth_video: Path) -> None:
    # The synth video is 720p; bumping the threshold to 1080 should refuse it.
    with pytest.raises(UnsupportedSourceError):
        probe_source(synth_video, min_resolution=1080)


def test_probe_source_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_source(tmp_path / "does-not-exist.mp4", min_resolution=720)


def test_sha256_file_deterministic(synth_video: Path) -> None:
    a = sha256_file(synth_video)
    b = sha256_file(synth_video)
    assert a == b
    assert len(a) == 64


def test_iter_frames_count_matches_source(synth_video: Path) -> None:
    decoded = list(iter_frames(synth_video))
    # 180 frames by construction. The encoder occasionally drops one at
    # the boundary; allow a tiny margin of error before failing.
    assert 178 <= len(decoded) <= 180, len(decoded)
    # PTS must be monotonically non-decreasing.
    pts = [f.pts_us for f in decoded]
    assert all(pts[i] <= pts[i + 1] for i in range(len(pts) - 1))
    # source_frame_idx must be 0..len-1 with no gaps.
    assert [f.source_frame_idx for f in decoded] == list(range(len(decoded)))
