"""On-disk artifact writer for the pitch stage.

Run directory layout:

    <pitch_runs_root>/<pitch_run_id>/
    ├── manifest.json       # PitchManifest, atomic write
    └── pitch.jsonl         # one FramePitch per processed frame

No per-frame images; source images live in the upstream ingest run and
detections live in the upstream CV run. Pitch runs are very small on disk
(homography matrices and metadata only, typically well under a megabyte
for a full broadcast clip).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO

from datum.cv.pitch.schemas import FramePitch, PitchManifest
from datum.utils.atomic import atomic_write_text

_PITCH_FILE: str = "pitch.jsonl"
_MANIFEST_FILE: str = "manifest.json"


class WriterStateError(RuntimeError):
    """Raised when the writer is used outside its context manager."""


class PitchWriter(AbstractContextManager["PitchWriter"]):
    """Persist pitch-stage artifacts to a run directory.

    Used as a context manager so the underlying file handle is always
    closed. The manifest is only safe to read after `__exit__`.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._pitch_fh: IO[str] | None = None

    def __enter__(self) -> "PitchWriter":
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._pitch_fh = (self._run_dir / _PITCH_FILE).open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._pitch_fh is not None:
            self._pitch_fh.close()
        self._pitch_fh = None

    def write_frame_pitch(self, record: FramePitch) -> None:
        if self._pitch_fh is None:
            raise WriterStateError(
                "write_frame_pitch called outside the writer's context"
            )
        self._pitch_fh.write(record.model_dump_json() + "\n")

    def write_manifest(self, manifest: PitchManifest) -> None:
        atomic_write_text(
            self._run_dir / _MANIFEST_FILE,
            manifest.model_dump_json(indent=2),
        )


__all__ = ["PitchWriter", "WriterStateError"]
