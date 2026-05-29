"""CV stage reader.

Loads an ingest run from disk and yields batches of (records, images) ready
for a `Detector.detect()` call. Non-pitch frames are dropped here when
`pitch_only` is set, so the detector never runs on crowd / bench / graphic
shots; that saves a meaningful fraction of GPU time on most broadcasts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from datum.ingest.schemas import FrameRecord, IngestManifest, SceneKind

_PITCH_KINDS: frozenset[SceneKind] = frozenset({SceneKind.PITCH, SceneKind.REPLAY})


class IngestRunNotFoundError(FileNotFoundError):
    """The expected ingest run directory or its manifest is missing."""


class IngestRunIncompleteError(RuntimeError):
    """The ingest manifest exists but does not show a finished run.

    Either the ingest crashed mid-flight, or the manifest was hand-edited
    in a way that broke its `finished_at` / `counters` fields. Either way,
    CV cannot run against an incomplete upstream artifact.
    """


def load_ingest_manifest(run_dir: Path) -> IngestManifest:
    """Load and validate the ingest manifest at `<run_dir>/manifest.json`."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise IngestRunNotFoundError(f"no manifest at {manifest_path}")
    manifest = IngestManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.finished_at is None or manifest.counters is None:
        raise IngestRunIncompleteError(
            f"ingest run at {run_dir} is not complete; "
            f"finished_at={manifest.finished_at}, counters={manifest.counters}"
        )
    return manifest


def iter_frame_records(run_dir: Path) -> Iterator[FrameRecord]:
    """Yield a FrameRecord for each line of `<run_dir>/frames.jsonl`."""
    frames_path = run_dir / "frames.jsonl"
    if not frames_path.exists():
        raise IngestRunNotFoundError(f"no frames.jsonl at {frames_path}")

    with frames_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            yield FrameRecord.model_validate_json(line)


def iter_batches(
    run_dir: Path,
    *,
    batch_size: int,
    pitch_only: bool = True,
) -> Iterator[tuple[list[FrameRecord], np.ndarray]]:
    """Yield `(records, images)` batches ready for `Detector.detect`.

    `images` is a 4D (N, H, W, 3) uint8 BGR ndarray. `records` is a list
    of length N aligned positionally with `images`. Frames that were
    skipped due to pitch_only do not appear in any batch; their absence is
    the signal to downstream that detection did not run there.

    All frames in a batch must share the same resolution; this is true for
    every ingest run produced from a single source file. If you somehow
    have a mixed-resolution run, np.stack will raise loudly.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")

    pending_records: list[FrameRecord] = []
    pending_images: list[np.ndarray] = []

    for record in iter_frame_records(run_dir):
        if pitch_only and record.scene_kind not in _PITCH_KINDS:
            continue

        image_path = run_dir / record.image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for {image_path}")

        pending_records.append(record)
        pending_images.append(image)

        if len(pending_records) >= batch_size:
            yield pending_records, np.stack(pending_images, axis=0)
            pending_records = []
            pending_images = []

    if pending_records:
        yield pending_records, np.stack(pending_images, axis=0)


__all__ = [
    "IngestRunIncompleteError",
    "IngestRunNotFoundError",
    "iter_batches",
    "iter_frame_records",
    "load_ingest_manifest",
]
