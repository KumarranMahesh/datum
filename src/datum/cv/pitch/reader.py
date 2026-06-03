"""Pitch stage reader.

The pitch stage operates on broadcast frames, not on detections, so it
needs the parent ingest run's images even though the CV run is what binds
the audit chain together. The reader loads the CV manifest, resolves the
ingest run from `cv_manifest.ingest_run_id`, then yields one
(FrameRecord, BGR image) pair per processed frame.

Non-pitch frames (as classified by the ingest's scene segmenter) are
dropped here when `pitch_only` is set so the solver never runs on crowd
or graphic shots, where homography is undefined.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from datum.cv.schemas import CvManifest
from datum.ingest.schemas import FrameRecord, IngestManifest, SceneKind

_PITCH_KINDS: frozenset[SceneKind] = frozenset({SceneKind.PITCH, SceneKind.REPLAY})


class CvRunNotFoundError(FileNotFoundError):
    """The expected CV run directory or its manifest is missing."""


class CvRunIncompleteError(RuntimeError):
    """The CV manifest exists but does not show a finished run."""


class IngestRunNotFoundError(FileNotFoundError):
    """The parent ingest run referenced by the CV manifest is missing
    from the configured ingest_runs_root."""


def load_cv_manifest(run_dir: Path) -> CvManifest:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise CvRunNotFoundError(f"no CV manifest at {manifest_path}")
    manifest = CvManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.finished_at is None or manifest.counters is None:
        raise CvRunIncompleteError(
            f"CV run at {run_dir} is not complete; "
            f"finished_at={manifest.finished_at}, counters={manifest.counters}"
        )
    return manifest


def resolve_ingest_run_dir(
    cv_manifest: CvManifest, ingest_runs_root: Path
) -> Path:
    candidate = ingest_runs_root / cv_manifest.ingest_run_id
    if not candidate.exists():
        raise IngestRunNotFoundError(
            f"ingest run '{cv_manifest.ingest_run_id}' referenced by the CV "
            f"manifest was not found at {candidate}. Pass --ingest-runs-root "
            "if the ingest runs live elsewhere."
        )
    return candidate


def load_ingest_manifest(ingest_run_dir: Path) -> IngestManifest:
    manifest_path = ingest_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise IngestRunNotFoundError(f"no ingest manifest at {manifest_path}")
    return IngestManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )


def iter_frames(
    ingest_run_dir: Path,
    *,
    pitch_only: bool = True,
) -> Iterator[tuple[FrameRecord, np.ndarray]]:
    """Yield (FrameRecord, BGR image) for each sampled frame.

    Records are emitted in the order they appear in frames.jsonl, which
    the ingest writer guarantees is chronological. When `pitch_only` is
    True, non-pitch frames are dropped before the disk read.
    """
    frames_path = ingest_run_dir / "frames.jsonl"
    if not frames_path.exists():
        raise IngestRunNotFoundError(f"no frames.jsonl at {frames_path}")

    with frames_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            record = FrameRecord.model_validate_json(line)
            if pitch_only and record.scene_kind not in _PITCH_KINDS:
                continue
            image_path = ingest_run_dir / record.image_path
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cv2.imread failed for {image_path}")
            yield record, image


__all__ = [
    "CvRunIncompleteError",
    "CvRunNotFoundError",
    "IngestRunNotFoundError",
    "iter_frames",
    "load_cv_manifest",
    "load_ingest_manifest",
    "resolve_ingest_run_dir",
]
