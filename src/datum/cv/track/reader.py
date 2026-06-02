"""Track stage reader.

Loads a CV run from disk and yields per-frame detections in chronological
order. The track pipeline feeds this directly into `Tracker.update()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from datum.cv.schemas import CvManifest, FrameDetections


class CvRunNotFoundError(FileNotFoundError):
    """The expected CV run directory or its manifest is missing."""


class CvRunIncompleteError(RuntimeError):
    """The CV manifest exists but does not show a finished run.

    Either CV crashed mid-flight, or the manifest was hand-edited in a way
    that broke its `finished_at` or `counters` fields. Either way, the
    track stage cannot run against an incomplete upstream artifact.
    """


def load_cv_manifest(run_dir: Path) -> CvManifest:
    """Load and validate the CV manifest at `<run_dir>/manifest.json`."""
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


def iter_frame_detections(run_dir: Path) -> Iterator[FrameDetections]:
    """Yield a `FrameDetections` per line of `<run_dir>/detections.jsonl`.

    The CV writer emits records in frame_idx order, so consumers can rely
    on chronological delivery without sorting.
    """
    path = run_dir / "detections.jsonl"
    if not path.exists():
        raise CvRunNotFoundError(f"no detections.jsonl at {path}")

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            yield FrameDetections.model_validate_json(line)


__all__ = [
    "CvRunIncompleteError",
    "CvRunNotFoundError",
    "iter_frame_detections",
    "load_cv_manifest",
]
