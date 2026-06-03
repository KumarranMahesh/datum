"""Top-level pitch pipeline.

`run(cv_run_dir, config, *, pitch_runs_root, ingest_runs_root)` is the
entrypoint. It loads the CV manifest, resolves the parent ingest run,
walks the ingest's images in chronological order, calls the configured
solver, and writes one FramePitch record per processed frame.

Determinism. `pitch_run_id` is `pitch-<12 hex>` derived from
`sha256(cv_run_id + ':' + pitch_config_hash)`. Identical inputs always
map to the same id; a complete prior run at that id is treated as a
cache hit and returned untouched.

The pipeline binds to a CV run for chain-audit consistency with the
track stage even though it does not consume any of the CV stage's
artifacts (only the parent ingest's images). The CV manifest's
`ingest_run_id` is the link.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog

from datum.cv.pitch import get as get_solver
from datum.cv.pitch.reader import (
    iter_frames,
    load_cv_manifest,
    resolve_ingest_run_dir,
)
from datum.cv.pitch.schemas import (
    SCHEMA_VERSION,
    FramePitch,
    PitchCounters,
    PitchManifest,
    PitchSolverConfig,
)
from datum.cv.pitch.writer import PitchWriter

_log = structlog.get_logger(__name__)


def _config_hash(config: PitchSolverConfig) -> str:
    parsed = json.loads(config.model_dump_json())
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_run_id(cv_run_id: str, config_hash: str) -> str:
    h = hashlib.sha256()
    h.update(cv_run_id.encode("ascii"))
    h.update(b":")
    h.update(config_hash.encode("ascii"))
    return f"pitch-{h.hexdigest()[:12]}"


def _is_complete_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("finished_at") is not None and data.get("counters") is not None


def _load_manifest(path: Path) -> PitchManifest:
    return PitchManifest.model_validate_json(path.read_text(encoding="utf-8"))


def run(
    cv_run_dir: Path,
    config: PitchSolverConfig,
    *,
    pitch_runs_root: Path,
    ingest_runs_root: Path,
    overwrite: bool = False,
) -> PitchManifest:
    """Run pitch homography solving against the parent ingest images.

    `cv_run_dir` provides the chain binding. `ingest_runs_root` is where
    the parent ingest run's image files live. Idempotent: complete prior
    runs at the same `pitch_run_id` are returned untouched unless
    `overwrite=True`.
    """
    cv_run_dir = cv_run_dir.expanduser().resolve()
    pitch_runs_root = pitch_runs_root.expanduser().resolve()
    ingest_runs_root = ingest_runs_root.expanduser().resolve()
    log = _log.bind(cv_run_dir=str(cv_run_dir))

    cv_manifest = load_cv_manifest(cv_run_dir)
    log = log.bind(cv_run_id=cv_manifest.run_id)

    ingest_run_dir = resolve_ingest_run_dir(cv_manifest, ingest_runs_root)
    log.info(
        "upstream.loaded",
        ingest_run_id=cv_manifest.ingest_run_id,
        detector=cv_manifest.detector_name,
    )

    config_hash_full = _config_hash(config)
    run_id = _derive_run_id(cv_manifest.run_id, config_hash_full)
    run_dir = pitch_runs_root / run_id
    log = log.bind(run_id=run_id)

    manifest_path = run_dir / "manifest.json"
    if not overwrite and _is_complete_manifest(manifest_path):
        log.info("run.cache-hit", path=str(run_dir))
        return _load_manifest(manifest_path)

    solver_cls = get_solver(config.solver)
    solver = solver_cls(**config.solver_config)
    log.info("solver.loaded", name=config.solver)

    started_at = datetime.now(tz=UTC)
    frames_processed = 0
    frames_with_homography = 0
    reprojection_errors: list[float] = []
    solver_wall_s = 0.0

    log.info("run.start")
    with PitchWriter(run_dir) as writer:
        for frame_record, image in iter_frames(
            ingest_run_dir, pitch_only=config.pitch_only
        ):
            t0 = time.perf_counter()
            homography = solver.solve(image)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            solver_wall_s += elapsed_ms / 1000.0

            writer.write_frame_pitch(
                FramePitch(
                    frame_idx=frame_record.frame_idx,
                    source_frame_idx=frame_record.source_frame_idx,
                    homography=homography,
                    solver_wall_ms=elapsed_ms,
                )
            )
            frames_processed += 1
            if homography is not None:
                frames_with_homography += 1
                reprojection_errors.append(homography.reprojection_error_px)

        finished_at = datetime.now(tz=UTC)
        mean_err: float | None = (
            float(sum(reprojection_errors) / len(reprojection_errors))
            if reprojection_errors
            else None
        )

        # cv_manifest.counters is guaranteed non-None by load_cv_manifest.
        assert cv_manifest.counters is not None
        counters = PitchCounters(
            frames_in=cv_manifest.counters.frames_processed,
            frames_processed=frames_processed,
            frames_with_homography=frames_with_homography,
            mean_reprojection_error_px=mean_err,
            solver_wall_s=solver_wall_s,
        )
        manifest = PitchManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            cv_run_id=cv_manifest.run_id,
            solver_name=config.solver,
            config=config,
            config_hash=config_hash_full,
            started_at=started_at,
            finished_at=finished_at,
            counters=counters,
        )
        writer.write_manifest(manifest)

    log.info(
        "run.done",
        frames_processed=frames_processed,
        frames_with_homography=frames_with_homography,
        mean_reprojection_error_px=(
            round(mean_err, 3) if mean_err is not None else None
        ),
        solver_s=round(solver_wall_s, 3),
        wall_s=round((finished_at - started_at).total_seconds(), 2),
    )
    return manifest


__all__ = ["run"]
