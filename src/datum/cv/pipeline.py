"""Top-level CV pipeline.

`run(ingest_run_dir, config, *, cv_runs_root)` is the entrypoint. It loads
the upstream ingest manifest, instantiates the configured detector, and
writes one FrameDetections record per processed frame.

Determinism. `cv_run_id` is `cv-<12 hex>` derived from
`sha256(ingest_run_id + ':' + cv_config_hash)`. The same ingest run + same
CV config map to the same cv_run_id, every time. A complete prior cv run
at that id is treated as a cache hit and returned untouched, which is what
makes re-running idempotent across both stages.

The pipeline does not import any specific detector. It looks the detector
up by name through `datum.cv.detect.get` and lets the registry resolve it.
That keeps adapter choice in config, not in code.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog

from datum.cv.detect import get as get_detector
from datum.cv.reader import iter_batches, load_ingest_manifest
from datum.cv.schemas import (
    SCHEMA_VERSION,
    CvConfig,
    CvCounters,
    CvManifest,
    DetectionBox,
    FrameDetections,
)
from datum.cv.writer import CvWriter

_log = structlog.get_logger(__name__)


def _config_hash(config: CvConfig) -> str:
    """Canonical-JSON sha256 of the config."""
    parsed = json.loads(config.model_dump_json())
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_run_id(ingest_run_id: str, config_hash: str) -> str:
    h = hashlib.sha256()
    h.update(ingest_run_id.encode("ascii"))
    h.update(b":")
    h.update(config_hash.encode("ascii"))
    return f"cv-{h.hexdigest()[:12]}"


def _is_complete_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("finished_at") is not None and data.get("counters") is not None


def _load_manifest(path: Path) -> CvManifest:
    return CvManifest.model_validate_json(path.read_text(encoding="utf-8"))


def run(
    ingest_run_dir: Path,
    config: CvConfig,
    *,
    cv_runs_root: Path,
    overwrite: bool = False,
) -> CvManifest:
    """Run CV detection on an ingest run. Returns the written manifest.

    `ingest_run_dir` is the path to a completed ingest run directory; it
    contains the manifest.json and frames.jsonl that this stage consumes.

    Idempotent. A complete prior CV run at the same `cv_run_id` is returned
    without touching disk unless `overwrite=True`.
    """
    ingest_run_dir = ingest_run_dir.expanduser().resolve()
    cv_runs_root = cv_runs_root.expanduser().resolve()
    log = _log.bind(ingest_run_dir=str(ingest_run_dir))

    ingest_manifest = load_ingest_manifest(ingest_run_dir)
    # load_ingest_manifest validated that counters and finished_at are set.
    assert ingest_manifest.counters is not None
    log = log.bind(ingest_run_id=ingest_manifest.run_id)
    log.info("upstream.loaded", source=ingest_manifest.source.path)

    config_hash_full = _config_hash(config)
    run_id = _derive_run_id(ingest_manifest.run_id, config_hash_full)
    run_dir = cv_runs_root / run_id
    log = log.bind(run_id=run_id)

    manifest_path = run_dir / "manifest.json"
    if not overwrite and _is_complete_manifest(manifest_path):
        log.info("run.cache-hit", path=str(run_dir))
        return _load_manifest(manifest_path)

    # Detector instantiation may download model weights on first use.
    detector_cls = get_detector(config.detector)
    detector = detector_cls(**config.detector_config)
    class_map = detector.class_map
    log.info(
        "detector.loaded",
        name=config.detector,
        classes=list(class_map.keys()),
    )

    started_at = datetime.now(tz=UTC)
    frames_processed = 0
    frames_with_detections = 0
    total_detections = 0
    detector_wall_s = 0.0

    log.info("run.start")
    with CvWriter(run_dir) as writer:
        for records, images in iter_batches(
            ingest_run_dir,
            batch_size=config.batch_size,
            pitch_only=config.pitch_only,
        ):
            t0 = time.perf_counter()
            batch_result = detector.detect(images)
            batch_wall = time.perf_counter() - t0
            detector_wall_s += batch_wall

            if len(batch_result.per_frame) != len(records):
                raise RuntimeError(
                    f"detector returned {len(batch_result.per_frame)} frame "
                    f"results for a batch of {len(records)} frames; "
                    "this violates the Detector contract"
                )

            # Per-frame inference time is the batch wall divided equally
            # across the frames in the batch. Not strictly accurate but a
            # reasonable attribution; per-frame variance inside a batch is
            # typically small for fixed-resolution input.
            per_frame_ms = (batch_wall * 1000.0) / len(records)

            for record, frame_detections in zip(
                records, batch_result.per_frame, strict=True
            ):
                persisted = [
                    DetectionBox(
                        x1=d.x1,
                        y1=d.y1,
                        x2=d.x2,
                        y2=d.y2,
                        confidence=d.confidence,
                        class_id=d.class_id,
                        class_name=class_map.get(d.class_id, f"class_{d.class_id}"),
                    )
                    for d in frame_detections
                ]
                if persisted:
                    frames_with_detections += 1
                    total_detections += len(persisted)
                frames_processed += 1

                writer.write_frame_detections(
                    FrameDetections(
                        frame_idx=record.frame_idx,
                        source_frame_idx=record.source_frame_idx,
                        detections=persisted,
                        inference_ms=per_frame_ms,
                    )
                )

        finished_at = datetime.now(tz=UTC)
        counters = CvCounters(
            frames_in=ingest_manifest.counters.sampled_frames,
            frames_processed=frames_processed,
            frames_with_detections=frames_with_detections,
            total_detections=total_detections,
            detector_wall_s=detector_wall_s,
        )
        manifest = CvManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            ingest_run_id=ingest_manifest.run_id,
            detector_name=config.detector,
            class_map=class_map,
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
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
        detector_s=round(detector_wall_s, 2),
        wall_s=round((finished_at - started_at).total_seconds(), 2),
    )
    return manifest


__all__ = ["run"]
