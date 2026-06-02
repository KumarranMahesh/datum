"""Top-level track pipeline.

`run(cv_run_dir, config, *, track_runs_root)` is the entrypoint. It loads
the upstream CV manifest, instantiates the configured tracker, walks the
CV run's detections.jsonl frame by frame, and writes one FrameTracks
record per processed frame.

Determinism. `track_run_id` is `track-<12 hex>` derived from
`sha256(cv_run_id + ':' + tracker_config_hash)`. Identical (cv_run_id,
tracker_config) inputs map to the same track_run_id; a complete prior
track run at that id is returned untouched, which is what makes the
pipeline idempotent across all three stages.

The pipeline does not import any specific tracker. It looks the tracker
up by name via `datum.cv.track.get` and lets the registry resolve it.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog

from datum.cv.detect import Detection
from datum.cv.track import get as get_tracker
from datum.cv.track.reader import iter_frame_detections, load_cv_manifest
from datum.cv.track.schemas import (
    SCHEMA_VERSION,
    FrameTracks,
    TrackCounters,
    TrackedDetection,
    TrackerConfig,
    TrackManifest,
)
from datum.cv.track.writer import TrackWriter
from datum.ingest.schemas import FrameRecord

_log = structlog.get_logger(__name__)


def _config_hash(config: TrackerConfig) -> str:
    parsed = json.loads(config.model_dump_json())
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_run_id(cv_run_id: str, config_hash: str) -> str:
    h = hashlib.sha256()
    h.update(cv_run_id.encode("ascii"))
    h.update(b":")
    h.update(config_hash.encode("ascii"))
    return f"track-{h.hexdigest()[:12]}"


def _is_complete_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("finished_at") is not None and data.get("counters") is not None


def _load_manifest(path: Path) -> TrackManifest:
    return TrackManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _load_scene_lookup(ingest_run_dir: Path) -> dict[int, int]:
    """Build a source_frame_idx -> scene_id lookup from an ingest run.

    Reads the ingest's frames.jsonl rather than scenes.jsonl because
    FrameRecord already carries scene_id per sampled frame and matches
    one-to-one with what the track pipeline iterates over.
    """
    frames_path = ingest_run_dir / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(
            f"reset_on_scene_cut requested but ingest frames.jsonl not found "
            f"at {frames_path}"
        )
    lookup: dict[int, int] = {}
    with frames_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            fr = FrameRecord.model_validate_json(line)
            lookup[fr.source_frame_idx] = fr.scene_id
    return lookup


def run(
    cv_run_dir: Path,
    config: TrackerConfig,
    *,
    track_runs_root: Path,
    ingest_runs_root: Path | None = None,
    overwrite: bool = False,
) -> TrackManifest:
    """Run tracking on a CV run. Returns the written manifest.

    Idempotent. A complete prior track run at the same `track_run_id` is
    returned without touching disk unless `overwrite=True`.

    `ingest_runs_root` is required when `config.reset_on_scene_cut` is True,
    because the track pipeline reads scene boundaries from the parent
    ingest run's frames.jsonl. Otherwise the parameter is unused.
    """
    cv_run_dir = cv_run_dir.expanduser().resolve()
    track_runs_root = track_runs_root.expanduser().resolve()
    log = _log.bind(cv_run_dir=str(cv_run_dir))

    cv_manifest = load_cv_manifest(cv_run_dir)
    assert cv_manifest.counters is not None
    log = log.bind(cv_run_id=cv_manifest.run_id)
    log.info("upstream.loaded", detector=cv_manifest.detector_name)

    config_hash_full = _config_hash(config)
    run_id = _derive_run_id(cv_manifest.run_id, config_hash_full)
    run_dir = track_runs_root / run_id
    log = log.bind(run_id=run_id)

    manifest_path = run_dir / "manifest.json"
    if not overwrite and _is_complete_manifest(manifest_path):
        log.info("run.cache-hit", path=str(run_dir))
        return _load_manifest(manifest_path)

    scene_lookup: dict[int, int] = {}
    if config.reset_on_scene_cut:
        if ingest_runs_root is None:
            raise ValueError(
                "reset_on_scene_cut=True requires ingest_runs_root so the "
                "track pipeline can read scene boundaries from the parent "
                "ingest run"
            )
        ingest_run_dir = (
            ingest_runs_root.expanduser().resolve() / cv_manifest.ingest_run_id
        )
        if not ingest_run_dir.exists():
            raise FileNotFoundError(
                f"reset_on_scene_cut requested but ingest run not found at "
                f"{ingest_run_dir}"
            )
        scene_lookup = _load_scene_lookup(ingest_run_dir)
        log.info(
            "scene-cut.loaded",
            ingest_run_dir=str(ingest_run_dir),
            frames_with_scene=len(scene_lookup),
            distinct_scenes=len(set(scene_lookup.values())),
        )

    tracker_cls = get_tracker(config.tracker)
    tracker = tracker_cls(**config.tracker_config)
    tracker.reset()
    class_map = cv_manifest.class_map
    log.info("tracker.loaded", name=config.tracker)

    started_at = datetime.now(tz=UTC)
    frames_processed = 0
    detections_in = 0
    tracked_observations = 0
    seen_track_ids: set[int] = set()
    tracker_wall_s = 0.0
    last_scene_id: int | None = None
    scene_resets_triggered: int = 0
    tracking_frame_idx: int = 0  # contiguous counter passed to the tracker

    log.info("run.start")
    with TrackWriter(run_dir) as writer:
        for frame_dets in iter_frame_detections(cv_run_dir):
            if scene_lookup:
                cur_scene_id = scene_lookup.get(frame_dets.source_frame_idx)
                if (
                    cur_scene_id is not None
                    and last_scene_id is not None
                    and cur_scene_id != last_scene_id
                ):
                    tracker.reset()
                    tracking_frame_idx = 0
                    scene_resets_triggered += 1
                if cur_scene_id is not None:
                    last_scene_id = cur_scene_id

            in_detections: list[Detection] = [
                Detection(
                    x1=d.x1,
                    y1=d.y1,
                    x2=d.x2,
                    y2=d.y2,
                    confidence=d.confidence,
                    class_id=d.class_id,
                )
                for d in frame_dets.detections
            ]
            detections_in += len(in_detections)

            t0 = time.perf_counter()
            observations = tracker.update(
                in_detections, frame_idx=tracking_frame_idx
            )
            tracker_wall_s += time.perf_counter() - t0
            tracking_frame_idx += 1

            persisted = [
                TrackedDetection(
                    track_id=o.track_id,
                    x1=o.x1,
                    y1=o.y1,
                    x2=o.x2,
                    y2=o.y2,
                    confidence=o.confidence,
                    class_id=o.class_id,
                    class_name=class_map.get(o.class_id, f"class_{o.class_id}"),
                )
                for o in observations
            ]

            for o in observations:
                seen_track_ids.add(o.track_id)
            tracked_observations += len(persisted)
            frames_processed += 1

            writer.write_frame_tracks(
                FrameTracks(
                    frame_idx=frame_dets.frame_idx,
                    source_frame_idx=frame_dets.source_frame_idx,
                    tracks=persisted,
                )
            )

        finished_at = datetime.now(tz=UTC)
        counters = TrackCounters(
            frames_in=cv_manifest.counters.frames_processed,
            frames_processed=frames_processed,
            detections_in=detections_in,
            tracked_observations=tracked_observations,
            unique_tracks=len(seen_track_ids),
            tracker_wall_s=tracker_wall_s,
        )
        manifest = TrackManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            cv_run_id=cv_manifest.run_id,
            tracker_name=config.tracker,
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
        detections_in=detections_in,
        tracked_observations=tracked_observations,
        unique_tracks=len(seen_track_ids),
        scene_resets=scene_resets_triggered,
        distinct_scenes=len(set(scene_lookup.values())) if scene_lookup else 0,
        tracker_s=round(tracker_wall_s, 3),
        wall_s=round((finished_at - started_at).total_seconds(), 2),
    )
    return manifest


__all__ = ["run"]
