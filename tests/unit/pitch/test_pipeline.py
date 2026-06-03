"""Pitch pipeline end-to-end tests.

Each test chains ingest -> cv -> pitch against the synth video, using
the noop pitch solver so the chain is fast and reproducible. The synth
video has no real pitch lines, so the noop is the right tool for
exercising the wiring.
"""

from __future__ import annotations

from pathlib import Path

from datum.cv import CvConfig
from datum.cv import run as cv_run
from datum.cv.pitch import run as pitch_run
from datum.cv.pitch.schemas import PitchManifest, PitchSolverConfig
from datum.ingest import IngestConfig
from datum.ingest import run as ingest_run


def _make_cv_run(synth_video: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Run ingest -> cv. Returns (cv_run_dir, ingest_runs_root)."""
    runs_root = tmp_path / "runs"
    cv_runs_root = tmp_path / "cv_runs"

    ingest_manifest = ingest_run(
        synth_video, IngestConfig(sample_fps=5.0), runs_root=runs_root
    )
    ingest_dir = runs_root / ingest_manifest.run_id

    cv_manifest = cv_run(
        ingest_dir, CvConfig(detector="noop"), cv_runs_root=cv_runs_root
    )
    return cv_runs_root / cv_manifest.run_id, runs_root


def test_noop_pitch_end_to_end(synth_video: Path, tmp_path: Path) -> None:
    cv_dir, ingest_runs_root = _make_cv_run(synth_video, tmp_path)
    pitch_runs_root = tmp_path / "pitch_runs"

    manifest = pitch_run(
        cv_dir,
        PitchSolverConfig(solver="pitch-noop"),
        pitch_runs_root=pitch_runs_root,
        ingest_runs_root=ingest_runs_root,
    )

    assert manifest.schema_version == 1
    assert manifest.run_id.startswith("pitch-")
    assert manifest.solver_name == "pitch-noop"
    assert manifest.counters is not None
    assert manifest.counters.frames_processed > 0
    # Noop solver returns None for every frame.
    assert manifest.counters.frames_with_homography == 0
    assert manifest.counters.mean_reprojection_error_px is None

    run_dir = pitch_runs_root / manifest.run_id
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "pitch.jsonl").exists()

    lines = (run_dir / "pitch.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest.counters.frames_processed


def test_pipeline_is_idempotent(synth_video: Path, tmp_path: Path) -> None:
    cv_dir, ingest_runs_root = _make_cv_run(synth_video, tmp_path)
    cfg = PitchSolverConfig(solver="pitch-noop")
    pitch_runs_root = tmp_path / "pitch_runs"

    a = pitch_run(
        cv_dir, cfg, pitch_runs_root=pitch_runs_root, ingest_runs_root=ingest_runs_root
    )
    b = pitch_run(
        cv_dir, cfg, pitch_runs_root=pitch_runs_root, ingest_runs_root=ingest_runs_root
    )

    assert a.run_id == b.run_id
    assert a.finished_at == b.finished_at  # cache hit returned the original


def test_manifest_binds_to_cv_run(synth_video: Path, tmp_path: Path) -> None:
    from datum.cv.schemas import CvManifest

    cv_dir, ingest_runs_root = _make_cv_run(synth_video, tmp_path)
    cv_manifest = CvManifest.model_validate_json(
        (cv_dir / "manifest.json").read_text(encoding="utf-8")
    )

    manifest = pitch_run(
        cv_dir,
        PitchSolverConfig(solver="pitch-noop"),
        pitch_runs_root=tmp_path / "pitch_runs",
        ingest_runs_root=ingest_runs_root,
    )
    assert manifest.cv_run_id == cv_manifest.run_id


def test_manifest_round_trips(synth_video: Path, tmp_path: Path) -> None:
    cv_dir, ingest_runs_root = _make_cv_run(synth_video, tmp_path)
    pitch_runs_root = tmp_path / "pitch_runs"

    manifest = pitch_run(
        cv_dir,
        PitchSolverConfig(solver="pitch-noop"),
        pitch_runs_root=pitch_runs_root,
        ingest_runs_root=ingest_runs_root,
    )
    on_disk = (pitch_runs_root / manifest.run_id / "manifest.json").read_text(
        encoding="utf-8"
    )
    loaded = PitchManifest.model_validate_json(on_disk)
    assert loaded.run_id == manifest.run_id
    assert loaded.cv_run_id == manifest.cv_run_id
    assert loaded.counters == manifest.counters
