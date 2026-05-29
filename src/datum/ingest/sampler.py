"""Frame sampler.

Resamples a source frame stream down to a target fps. Handles non-integer
ratios (NTSC 24000/1001 to 5 fps is a real case) without drift.

Algorithm: track `next_threshold` as the source-index at which the next
sample is due. Each yielded frame advances the threshold by
`source_fps / sample_fps`. Over long durations the sample count converges
on `duration_s * sample_fps` regardless of whether the ratio is integer.

The sampler is intentionally state-light: a single float counter.
Deterministic given (source, source_fps, sample_fps), which is the
property the pipeline needs for reproducibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from datum.ingest.reader import DecodedFrame


def sample_stream(
    frames: Iterable[DecodedFrame],
    *,
    source_fps: float,
    sample_fps: float,
) -> Iterator[DecodedFrame]:
    """Yield frames at approximately `sample_fps`.

    If source_fps <= sample_fps the stream is passed through untouched.
    The function never invents frames; upsampling is out of scope.
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive; got {sample_fps}")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive; got {source_fps}")

    if source_fps <= sample_fps:
        yield from frames
        return

    interval = source_fps / sample_fps
    next_threshold = 0.0
    for src_idx, frame in enumerate(frames):
        if src_idx >= next_threshold:
            yield frame
            next_threshold += interval


__all__ = ["sample_stream"]
