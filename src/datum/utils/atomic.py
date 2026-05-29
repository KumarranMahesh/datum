"""Atomic file writes.

Used by every stage's manifest writer. Centralised here so the
crash-tolerance guarantee lives in one place and the pattern does not
drift between callers.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, contents: str) -> None:
    """Write `contents` to `path` via tmp + rename.

    The file either appears in full at `path` or not at all. A crash
    mid-write leaves no partial file at the target path; a stray `.tmp`
    sibling may be left behind and is safe to delete on next run.

    `os.replace` is atomic on POSIX. On Windows + NTFS it is atomic when
    both paths are on the same volume, which they are by construction
    here (the tmp file is a sibling of the target).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(contents)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


__all__ = ["atomic_write_text"]
