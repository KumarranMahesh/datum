"""Track pipeline end-to-end tests.

Each test chains ingest -> cv -> track against the synth video. The CV
stage uses the noop detector, so the track stage receives empty input,
which is the right thing for testing wiring without committing GPU time.
"""

from __future__ import annotations

from pathlib import Path

from datum.cv import CvConfig
from datum.cv import run as cv_run
from datum.cv.track import run as track_run
from datum.cv.track.schemas import TrackerConfig, TrackManifest
from datum.ingest import IngestConfig
from datum.ingest import run as ingest_run


def _make_chain(synth_video: Path, tmp_path: Path) -> Path:
    """Run ingest -> cv (noop detector) and return the cv run directory."""
    runs_root = tmp_path / "runs"
    cv_runs_root = tmp_path / "cv_runs"

    ingest_manifest = ingest_run(
        synth_video, IngestConfig(sample_fps=5.0), runs_root=runs_root
    )
    ingest_dir = runs_root / ingest_manifest.run_id

    cv_manifest = cv_run(
        ingest_dir, CvConfig(detector="noop"), cv_runs_root=cv_runs_root
    )
    return cv_runs_root / cv_manifest.run_id


def test_noop_tracker_end_to_end(synth_video: Path, tmp_path: Path) -> None:
    cv_dir = _make_chain(synth_video, tmp_path)
    track_runs_root = tmp_path / "track_runs"

    manifest = track_run(
        cv_dir,
        TrackerConfig(tracker="track-noop"),
        track_runs_root=track_runs_root,
    )

    assert manifest.schema_version == 1
    assert manifest.run_id.startswith("track-")
    assert manifest.tracker_name == "track-noop"
    assert manifest.counters is not None
    # Noop detector produced no detections, so the tracker has nothing to do.
    assert manifest.counters.tracked_observations == 0
    assert manifest.counters.unique_tracks == 0

    run_dir = track_runs_root / manifest.run_id
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "tracks.jsonl").exists()


def test_bytetrack_end_to_end(synth_video: Path, tmp_path: Path) -> None:
    cv_dir = _make_chain(synth_video, tmp_path)
    manifest = track_run(
        cv_dir,
        TrackerConfig(tracker="bytetrack"),
        track_runs_root=tmp_path / "track_runs",
    )
    assert manifest.counters is not None
    # The synth video has no people; the chain exercises wiring only.
    assert manifest.counters.unique_tracks == 0


def test_pipeline_is_idempotent(synth_video: Path, tmp_path: Path) -> None:
    cv_dir = _make_chain(synth_video, tmp_path)
    cfg = TrackerConfig(tracker="track-noop")
    track_runs_root = tmp_path / "track_runs"

    a = track_run(cv_dir, cfg, track_runs_root=track_runs_root)
    b = track_run(cv_dir, cfg, track_runs_root=track_runs_root)

    assert a.run_id == b.run_id
    assert a.finished_at == b.finished_at  # cache hit returned the original


def test_different_trackers_produce_different_run_ids(
    synth_video: Path, tmp_path: Path
) -> None:
    cv_dir = _make_chain(synth_video, tmp_path)
    track_runs_root = tmp_path / "track_runs"

    a = track_run(
        cv_dir,
        TrackerConfig(tracker="track-noop"),
        track_runs_root=track_runs_root,
    )
    b = track_run(
        cv_dir,
        TrackerConfig(tracker="bytetrack"),
        track_runs_root=track_runs_root,
    )

    assert a.run_id != b.run_id


def test_manifest_round_trips(synth_video: Path, tmp_path: Path) -> None:
    cv_dir = _make_chain(synth_video, tmp_path)
    track_runs_root = tmp_path / "track_runs"

    manifest = track_run(
        cv_dir,
        TrackerConfig(tracker="track-noop"),
        track_runs_root=track_runs_root,
    )
    on_disk = (track_runs_root / manifest.run_id / "manifest.json").read_text(
        encoding="utf-8"
    )
    loaded = TrackManifest.model_validate_json(on_disk)
    assert loaded.run_id == manifest.run_id
    assert loaded.cv_run_id == manifest.cv_run_id


def test_track_manifest_binds_to_cv_run(synth_video: Path, tmp_path: Path) -> None:
    from datum.cv.schemas import CvManifest

    cv_dir = _make_chain(synth_video, tmp_path)
    cv_manifest = CvManifest.model_validate_json(
        (cv_dir / "manifest.json").read_text(encoding="utf-8")
    )

    track_manifest = track_run(
        cv_dir,
        TrackerConfig(tracker="track-noop"),
        track_runs_root=tmp_path / "track_runs",
    )
    assert track_manifest.cv_run_id == cv_manifest.run_id


def test_reset_on_scene_cut_requires_ingest_runs_root(
    synth_video: Path, tmp_path: Path
) -> None:
    import pytest

    cv_dir = _make_chain(synth_video, tmp_path)
    with pytest.raises(ValueError, match="ingest_runs_root"):
        track_run(
            cv_dir,
            TrackerConfig(tracker="track-noop", reset_on_scene_cut=True),
            track_runs_root=tmp_path / "track_runs",
            # ingest_runs_root deliberately omitted
        )


def test_reset_on_scene_cut_runs_end_to_end(
    synth_video: Path, tmp_path: Path
) -> None:
    # The synth video has 3 scenes; with reset_on_scene_cut=true the
    # pipeline should run to completion without raising.
    cv_dir = _make_chain(synth_video, tmp_path)
    manifest = track_run(
        cv_dir,
        TrackerConfig(tracker="track-noop", reset_on_scene_cut=True),
        track_runs_root=tmp_path / "track_runs",
        ingest_runs_root=tmp_path / "runs",
    )
    assert manifest.counters is not None
    # noop detector produced no detections, so no tracks. The point of this
    # test is that the scene-cut path runs cleanly.
    assert manifest.counters.tracked_observations == 0
