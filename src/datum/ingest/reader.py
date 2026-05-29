"""PyAV-based video reader.

This is the only module in the project that imports `av` directly.
Everything else goes through the iterator and dataclasses defined here.
Containing the PyAV surface in one file means a future swap to a GPU
decoder (NVDEC, VPF, torchvision.io) touches one place.

Performance note: decoding broadcast 1080p H.264 in pure software on an
i9 lands somewhere between 200 and 600 fps depending on the bitrate.
A 90-minute match at 30 fps takes 5 to 15 minutes to decode end to end.
GPU decode is on the roadmap, not in 0.1.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from datum.ingest.schemas import SourceInfo

_HASH_CHUNK_BYTES: int = 1 << 20  # 1 MiB


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One raw decoded frame from the source.

    `image` is BGR uint8 to match OpenCV's convention. PyAV's default is
    RGB; converting here avoids a transpose at every downstream consumer.
    """

    source_frame_idx: int
    pts_us: int
    image: np.ndarray  # (H, W, 3) BGR uint8


class UnsupportedSourceError(Exception):
    """Raised when a source cannot be processed.

    Low resolution, missing video stream, undecodable codec, zero fps.
    The pipeline catches this and exits non-zero rather than producing
    silent garbage downstream.
    """


def sha256_file(path: Path) -> str:
    """SHA-256 of file bytes, streamed.

    1 MiB chunks. Small enough not to spike memory on a 50 GB source,
    large enough that hashing isn't IO-bound on a typical SSD.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_source(path: Path, *, min_resolution: int) -> SourceInfo:
    """Open the container, read metadata, refuse anything too small.

    The file hash is computed eagerly so a wrong file fails fast, before
    the decode loop spends an hour producing artifacts that get thrown away.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise UnsupportedSourceError(f"no video stream in {path}")
        stream = container.streams.video[0]

        width = stream.codec_context.width
        height = stream.codec_context.height
        if min(width, height) < min_resolution:
            raise UnsupportedSourceError(
                f"{path} is {width}x{height}; minimum short axis is {min_resolution}. "
                f"Pitch homography is unreliable below {min_resolution}p, so this is "
                "refused rather than processed into bad downstream artifacts."
            )

        # average_rate is a Fraction. float is fine for metadata; PTS
        # arithmetic stays in the rational domain inside the decode loop.
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if fps <= 0:
            raise UnsupportedSourceError(f"could not determine fps for {path}")

        duration_s = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
        container_format = container.format.name
        codec_name = stream.codec_context.codec.name

    return SourceInfo(
        path=str(path),
        sha256=sha256_file(path),
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        container=container_format,
        video_codec=codec_name,
    )


def iter_frames(path: Path) -> Iterator[DecodedFrame]:
    """Yield every decoded frame from the source in order.

    PTS handling: when PyAV returns a frame with `pts is None` (some
    transcoded YouTube clips do this), the pts_us is synthesised from the
    frame index using the stream's average rate. Real broadcast streams
    almost never hit this branch.

    PTS is reported relative to the stream's own time base. It is *not*
    normalised to start at 0. Downstream stages that join across sources
    must handle their own offsets.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        avg_rate = stream.average_rate
        fps = float(avg_rate) if avg_rate else 25.0  # only used for synthetic PTS

        idx = 0
        for frame in container.decode(stream):
            if frame.pts is not None:
                pts_us = int(frame.pts * time_base * 1_000_000)
            else:
                pts_us = int(idx * 1_000_000 / fps)

            image = frame.to_ndarray(format="bgr24")

            yield DecodedFrame(
                source_frame_idx=idx,
                pts_us=pts_us,
                image=image,
            )
            idx += 1


__all__ = [
    "DecodedFrame",
    "UnsupportedSourceError",
    "iter_frames",
    "probe_source",
    "sha256_file",
]
