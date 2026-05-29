"""On-disk artifact writer for the ingest stage.

Run directory layout:

    <runs_root>/<run_id>/
    ├── manifest.json          # IngestManifest, atomic write
    ├── frames.jsonl           # one FrameRecord per line
    ├── scenes.jsonl           # one SceneSegment per line
    └── images/
        └── 000000.jpg         # frame_idx-named, zero-padded

`manifest.json` is the durable record. It is written last, via tmp + rename,
so a crashed or killed ingest never leaves a half-corrupted manifest.
JSONL files are append-only and tolerate partial trailing lines on crash,
which is what makes them safe under SIGKILL.

The writer is a context manager. Use `with IngestWriter(...) as w:`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO

import cv2

from datum.ingest.reader import DecodedFrame
from datum.ingest.schemas import (
    FrameRecord,
    IngestManifest,
    SceneKind,
    SceneSegment,
)
from datum.utils.atomic import atomic_write_text

_IMAGE_SUBDIR: str = "images"
_FRAMES_FILE: str = "frames.jsonl"
_SCENES_FILE: str = "scenes.jsonl"
_MANIFEST_FILE: str = "manifest.json"


class WriterStateError(RuntimeError):
    """Raised when the writer is used outside its context manager."""


class IngestWriter(AbstractContextManager["IngestWriter"]):
    """Persist ingest artifacts to a run directory.

    Construction does not touch disk. Use as a context manager so the
    underlying file handles are always closed; the manifest is only safe
    to read after __exit__.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        image_format: str = "jpg",
        image_quality: int = 92,
    ) -> None:
        if image_format not in ("jpg", "png"):
            raise ValueError(f"image_format must be jpg or png; got {image_format}")
        self._run_dir = run_dir
        self._image_format = image_format
        self._image_quality = image_quality
        self._images_dir = run_dir / _IMAGE_SUBDIR
        self._frames_fh: IO[str] | None = None
        self._scenes_fh: IO[str] | None = None

    def __enter__(self) -> "IngestWriter":
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        # 'w' mode truncates. A run id collision must be handled at the
        # pipeline level (skip-if-complete), not silently here.
        self._frames_fh = (self._run_dir / _FRAMES_FILE).open("w", encoding="utf-8")
        self._scenes_fh = (self._run_dir / _SCENES_FILE).open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for fh in (self._frames_fh, self._scenes_fh):
            if fh is not None:
                fh.close()
        self._frames_fh = None
        self._scenes_fh = None

    def write_frame(
        self,
        frame: DecodedFrame,
        *,
        frame_idx: int,
        scene_id: int,
        scene_kind: SceneKind,
    ) -> FrameRecord:
        """Write the image to disk and append a FrameRecord to frames.jsonl.

        Returns the FrameRecord written. The caller does not need to do
        anything with the return value; it is mostly there to make tests
        readable.
        """
        if self._frames_fh is None:
            raise WriterStateError("write_frame called outside the writer's context")

        rel_image_path = f"{_IMAGE_SUBDIR}/{frame_idx:06d}.{self._image_format}"
        abs_image_path = self._run_dir / rel_image_path
        self._write_image(frame, abs_image_path)

        h, w = frame.image.shape[:2]
        record = FrameRecord(
            frame_idx=frame_idx,
            source_frame_idx=frame.source_frame_idx,
            pts_us=frame.pts_us,
            width=w,
            height=h,
            scene_id=scene_id,
            scene_kind=scene_kind,
            image_path=rel_image_path,
        )
        self._frames_fh.write(record.model_dump_json() + "\n")
        return record

    def write_scenes(self, segments: list[SceneSegment]) -> None:
        if self._scenes_fh is None:
            raise WriterStateError("write_scenes called outside the writer's context")
        for seg in segments:
            self._scenes_fh.write(seg.model_dump_json() + "\n")

    def write_manifest(self, manifest: IngestManifest) -> None:
        """Atomic write of the top-level manifest.

        Delegates to `datum.utils.atomic.atomic_write_text` so the
        tmp+fsync+rename contract lives in one place.
        """
        atomic_write_text(
            self._run_dir / _MANIFEST_FILE,
            manifest.model_dump_json(indent=2),
        )

    def _write_image(self, frame: DecodedFrame, path: Path) -> None:
        if self._image_format == "jpg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self._image_quality]
        else:
            # PNG compression 0..9; 3 is a reasonable speed/size tradeoff
            # for this use case. The reference is rough rather than tuned.
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        ok = cv2.imwrite(str(path), frame.image, params)
        if not ok:
            raise OSError(f"cv2.imwrite failed for {path}")


__all__ = ["IngestWriter", "WriterStateError"]
