"""On-disk artifact writer for the track stage.

Run directory layout:

    <track_runs_root>/<track_run_id>/
    ├── manifest.json       # TrackManifest, atomic write
    └── tracks.jsonl        # one FrameTracks per processed frame

No per-frame images; source images live in the upstream ingest run, and
detections live in the upstream CV run. Track runs are tiny on disk
(typically under a megabyte for a full broadcast clip).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO

from datum.cv.track.schemas import FrameTracks, TrackManifest
from datum.utils.atomic import atomic_write_text

_TRACKS_FILE: str = "tracks.jsonl"
_MANIFEST_FILE: str = "manifest.json"


class WriterStateError(RuntimeError):
    """Raised when the writer is used outside its context manager."""


class TrackWriter(AbstractContextManager["TrackWriter"]):
    """Persist track-stage artifacts to a run directory.

    Used as a context manager so the underlying file handle is always
    closed. The manifest is only safe to read after `__exit__`.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._tracks_fh: IO[str] | None = None

    def __enter__(self) -> "TrackWriter":
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._tracks_fh = (self._run_dir / _TRACKS_FILE).open(
            "w", encoding="utf-8"
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._tracks_fh is not None:
            self._tracks_fh.close()
        self._tracks_fh = None

    def write_frame_tracks(self, record: FrameTracks) -> None:
        if self._tracks_fh is None:
            raise WriterStateError(
                "write_frame_tracks called outside the writer's context"
            )
        self._tracks_fh.write(record.model_dump_json() + "\n")

    def write_manifest(self, manifest: TrackManifest) -> None:
        atomic_write_text(
            self._run_dir / _MANIFEST_FILE,
            manifest.model_dump_json(indent=2),
        )


__all__ = ["TrackWriter", "WriterStateError"]
