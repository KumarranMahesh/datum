"""Top-level ingest pipeline.

`run(source_path, config, *, runs_root)` is the entrypoint. Wires reader,
sampler, segmenter, and writer in a single pass.

Determinism. `run_id` is `ingest-<12 hex>`, where the hex is the first
12 chars of `sha256(source.sha256 + ':' + config_hash)`. The same source
bytes plus the same config map to the same run_id, every time. A complete
prior run at that id is treated as a cache hit and returned untouched,
which is what makes re-running the pipeline a no-op by design.

This file uses `iter_frames`, `sample_stream`, `HistogramSceneSegmenter`,
and `IngestWriter`. None of those should leak to the public package
surface; the only thing the rest of the project imports from
`datum.ingest` is `run` and the schemas in `__init__.py`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import structlog

from datum.ingest.reader import iter_frames, probe_source
from datum.ingest.sampler import sample_stream
from datum.ingest.scenes import HistogramSceneSegmenter
from datum.ingest.schemas import (
    SCHEMA_VERSION,
    IngestConfig,
    IngestCounters,
    IngestManifest,
    SceneKind,
)
from datum.ingest.writer import IngestWriter

if TYPE_CHECKING:
    from datum.ingest.reader import DecodedFrame

_log = structlog.get_logger(__name__)

T = TypeVar("T")


class _CountingIterator(Iterator[T]):
    """Wrap an iterator and expose how many items have passed through it.

    Used to count source frames decoded without adding a second pass over
    the video file. The counter is updated *after* the item is yielded, so
    the value reflects items actually consumed downstream, not items the
    inner iterator has produced.
    """

    def __init__(self, inner: Iterator[T]) -> None:
        self._inner = inner
        self.count: int = 0

    def __iter__(self) -> "_CountingIterator[T]":
        return self

    def __next__(self) -> T:
        item = next(self._inner)
        self.count += 1
        return item


def _config_hash(config: IngestConfig) -> str:
    """Canonical JSON sha256 of the config.

    Pydantic's model_dump_json is not guaranteed to be byte-canonical, so
    the JSON is parsed and re-emitted with sorted keys before hashing.
    Cross-machine, cross-version determinism depends on this.
    """
    parsed = json.loads(config.model_dump_json())
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_run_id(source_sha256: str, config_hash: str) -> str:
    h = hashlib.sha256()
    h.update(source_sha256.encode("ascii"))
    h.update(b":")
    h.update(config_hash.encode("ascii"))
    return f"ingest-{h.hexdigest()[:12]}"


def _is_complete_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("finished_at") is not None and data.get("counters") is not None


def _load_manifest(path: Path) -> IngestManifest:
    return IngestManifest.model_validate_json(path.read_text(encoding="utf-8"))


def run(
    source_path: Path,
    config: IngestConfig,
    *,
    runs_root: Path,
    overwrite: bool = False,
) -> IngestManifest:
    """Ingest a single video. Returns the written manifest.

    Idempotent. A complete prior run at the same run_id is returned
    without touching disk unless `overwrite=True`.
    """
    source_path = source_path.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    log = _log.bind(source=str(source_path))

    log.info("probe.start")
    source = probe_source(source_path, min_resolution=config.min_resolution)
    log.info(
        "probe.done",
        duration_s=round(source.duration_s, 2),
        fps=round(source.fps, 3),
        width=source.width,
        height=source.height,
        sha256_prefix=source.sha256[:12],
    )

    config_hash_full = _config_hash(config)
    run_id = _derive_run_id(source.sha256, config_hash_full)
    run_dir = runs_root / run_id
    log = log.bind(run_id=run_id)

    manifest_path = run_dir / "manifest.json"
    if not overwrite and _is_complete_manifest(manifest_path):
        log.info("run.cache-hit", path=str(run_dir))
        return _load_manifest(manifest_path)

    started_at = datetime.now(tz=UTC)
    segmenter = HistogramSceneSegmenter(cut_threshold=config.scene_cut_threshold)

    pitch_count = 0
    sampled_count = 0

    decoded_counter: _CountingIterator[DecodedFrame] = _CountingIterator(
        iter_frames(source_path)
    )
    sampled = sample_stream(
        decoded_counter,
        source_fps=source.fps,
        sample_fps=config.sample_fps,
    )

    log.info("run.start")
    with IngestWriter(
        run_dir,
        image_format=config.image_format,
        image_quality=config.image_quality,
    ) as writer:
        for frame_idx, (frame, scene_id, kind) in enumerate(
            segmenter.process(sampled)
        ):
            writer.write_frame(
                frame,
                frame_idx=frame_idx,
                scene_id=scene_id,
                scene_kind=kind,
            )
            sampled_count += 1
            if kind == SceneKind.PITCH:
                pitch_count += 1

        segments = segmenter.finalize()
        writer.write_scenes(segments)

        finished_at = datetime.now(tz=UTC)
        counters = IngestCounters(
            source_frames=decoded_counter.count,
            sampled_frames=sampled_count,
            scenes=len(segments),
            pitch_frame_pct=(pitch_count / sampled_count) if sampled_count else 0.0,
        )
        manifest = IngestManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            source=source,
            config=config,
            config_hash=config_hash_full,
            started_at=started_at,
            finished_at=finished_at,
            counters=counters,
        )
        writer.write_manifest(manifest)

    log.info(
        "run.done",
        sampled=sampled_count,
        scenes=len(segments),
        pitch_pct=round(counters.pitch_frame_pct, 3),
        wall_s=round((finished_at - started_at).total_seconds(), 2),
    )
    return manifest


__all__ = ["run"]
