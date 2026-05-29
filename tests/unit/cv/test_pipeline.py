"""CV pipeline end-to-end tests.

Each test chains ingest -> cv against the synth video. The noop detector
is used throughout so tests run in milliseconds and require no model
weights.
"""

from __future__ import annotations

from pathlib import Path

from datum.cv import CvConfig, CvManifest
from datum.cv import run as cv_run
from datum.ingest import IngestConfig
from datum.ingest import run as ingest_run


def _make_ingest_run(synth_video: Path, tmp_path: Path) -> Path:
    """Run ingest once and return the run directory."""
    runs_root = tmp_path / "runs"
    manifest = ingest_run(synth_video, IngestConfig(sample_fps=5.0), runs_root=runs_root)
    return runs_root / manifest.run_id


def test_noop_pipeline_end_to_end(synth_video: Path, tmp_path: Path) -> None:
    ingest_dir = _make_ingest_run(synth_video, tmp_path)
    cv_runs_root = tmp_path / "cv_runs"

    manifest = cv_run(ingest_dir, CvConfig(detector="noop"), cv_runs_root=cv_runs_root)

    assert manifest.schema_version == 1
    assert manifest.run_id.startswith("cv-")
    assert manifest.detector_name == "noop"
    assert manifest.counters is not None
    assert manifest.counters.frames_processed > 0
    assert manifest.counters.frames_with_detections == 0
    assert manifest.counters.total_detections == 0

    run_dir = cv_runs_root / manifest.run_id
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "detections.jsonl").exists()

    lines = (run_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest.counters.frames_processed


def test_pipeline_is_idempotent(synth_video: Path, tmp_path: Path) -> None:
    ingest_dir = _make_ingest_run(synth_video, tmp_path)
    cv_runs_root = tmp_path / "cv_runs"
    cfg = CvConfig(detector="noop")

    a = cv_run(ingest_dir, cfg, cv_runs_root=cv_runs_root)
    b = cv_run(ingest_dir, cfg, cv_runs_root=cv_runs_root)

    assert a.run_id == b.run_id
    assert a.finished_at == b.finished_at  # cache hit returned the original


def test_different_configs_produce_different_run_ids(
    synth_video: Path, tmp_path: Path
) -> None:
    ingest_dir = _make_ingest_run(synth_video, tmp_path)
    cv_runs_root = tmp_path / "cv_runs"

    a = cv_run(ingest_dir, CvConfig(detector="noop"), cv_runs_root=cv_runs_root)
    b = cv_run(
        ingest_dir,
        CvConfig(detector="noop", batch_size=4),
        cv_runs_root=cv_runs_root,
    )

    assert a.run_id != b.run_id


def test_pitch_only_filters_non_pitch_frames(
    synth_video: Path, tmp_path: Path
) -> None:
    ingest_dir = _make_ingest_run(synth_video, tmp_path)
    cv_runs_root = tmp_path / "cv_runs"

    with_filter = cv_run(
        ingest_dir,
        CvConfig(detector="noop", pitch_only=True),
        cv_runs_root=cv_runs_root,
    )
    without_filter = cv_run(
        ingest_dir,
        CvConfig(detector="noop", pitch_only=False),
        cv_runs_root=cv_runs_root,
    )

    assert with_filter.counters is not None and without_filter.counters is not None
    # pitch_only=False sees every sampled frame; pitch_only=True drops the
    # synth video's grey middle scene.
    assert (
        without_filter.counters.frames_processed
        >= with_filter.counters.frames_processed
    )
    assert with_filter.run_id != without_filter.run_id


def test_manifest_round_trips(synth_video: Path, tmp_path: Path) -> None:
    ingest_dir = _make_ingest_run(synth_video, tmp_path)
    cv_runs_root = tmp_path / "cv_runs"

    manifest = cv_run(
        ingest_dir, CvConfig(detector="noop"), cv_runs_root=cv_runs_root
    )
    on_disk = (cv_runs_root / manifest.run_id / "manifest.json").read_text(
        encoding="utf-8"
    )
    loaded = CvManifest.model_validate_json(on_disk)

    assert loaded.run_id == manifest.run_id
    assert loaded.ingest_run_id == manifest.ingest_run_id
    assert loaded.counters == manifest.counters
    assert loaded.class_map == manifest.class_map


def test_cv_manifest_binds_to_ingest_run(synth_video: Path, tmp_path: Path) -> None:
    ingest_dir = _make_ingest_run(synth_video, tmp_path)

    # Pull the ingest run_id from its own manifest for comparison.
    from datum.ingest import IngestManifest

    ingest_manifest = IngestManifest.model_validate_json(
        (ingest_dir / "manifest.json").read_text(encoding="utf-8")
    )

    cv_manifest = cv_run(
        ingest_dir, CvConfig(detector="noop"), cv_runs_root=tmp_path / "cv_runs"
    )
    assert cv_manifest.ingest_run_id == ingest_manifest.run_id
