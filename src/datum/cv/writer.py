"""On-disk artifact writer for the CV stage.

Run directory layout:

    <cv_runs_root>/<cv_run_id>/
    ├── manifest.json           # CvManifest, atomic write
    └── detections.jsonl        # one FrameDetections per processed frame

No per-frame image files; the source images live in the upstream ingest
run and are referenced through `ingest_run_id` in the manifest. That keeps
CV runs cheap on disk (single-digit MB for a full broadcast).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO

from datum.cv.schemas import CvManifest, FrameDetections
from datum.utils.atomic import atomic_write_text

_DETECTIONS_FILE: str = "detections.jsonl"
_MANIFEST_FILE: str = "manifest.json"


class WriterStateError(RuntimeError):
    """Raised when the writer is used outside its context manager."""


class CvWriter(AbstractContextManager["CvWriter"]):
    """Persist CV-stage artifacts to a run directory.

    Used as a context manager so the underlying file handle is always
    closed. The manifest is only safe to read after `__exit__`.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._detections_fh: IO[str] | None = None

    def __enter__(self) -> "CvWriter":
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # 'w' mode truncates. Pipeline-level skip-if-complete handles the
        # idempotent re-run case before the writer is opened.
        self._detections_fh = (self._run_dir / _DETECTIONS_FILE).open(
            "w", encoding="utf-8"
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._detections_fh is not None:
            self._detections_fh.close()
        self._detections_fh = None

    def write_frame_detections(self, record: FrameDetections) -> None:
        if self._detections_fh is None:
            raise WriterStateError(
                "write_frame_detections called outside the writer's context"
            )
        self._detections_fh.write(record.model_dump_json() + "\n")

    def write_manifest(self, manifest: CvManifest) -> None:
        atomic_write_text(
            self._run_dir / _MANIFEST_FILE,
            manifest.model_dump_json(indent=2),
        )


__all__ = ["CvWriter", "WriterStateError"]
