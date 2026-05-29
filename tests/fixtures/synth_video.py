"""Synthetic video generator for ingest tests.

Produces a small mp4 with known scene cuts at known frame indices so the
scene cut detector can be verified deterministically. The fixture is
regenerated on demand and never committed to git; tests/fixtures/ is
gitignored for the same reason.

Layout of the default output:

    scene 0  (frames 0..59)    green dominant, pitch-like
    scene 1  (frames 60..119)  neutral grey, non-pitch
    scene 2  (frames 120..179) green dominant, pitch-like again

720p / 30 fps / 6 seconds. A small moving rectangle per scene prevents the
encoder from collapsing the bitrate to near-zero, which used to cause the
PyAV decoder to under-report frame counts on some platforms.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

# BGR tuples. Pitch-green sits squarely inside the classifier's hue band.
_PITCH_BGR: tuple[int, int, int] = (40, 180, 60)
_NON_PITCH_BGR: tuple[int, int, int] = (120, 120, 120)


def generate_synth_video(
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    scene_lengths: tuple[int, ...] = (60, 60, 60),
) -> Path:
    """Write a synthetic mp4 to out_path. Returns out_path.

    Output has `len(scene_lengths)` scenes, alternating pitch-green and
    grey. The histogram-diff cut detector should fire at every transition.
    """
    if any(length <= 0 for length in scene_lengths):
        raise ValueError(f"scene_lengths must all be positive; got {scene_lengths}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    palette = [_PITCH_BGR, _NON_PITCH_BGR]

    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # crf 28 is small-but-not-lossless. Tests do not need sharper.
        stream.options = {"crf": "28"}

        for scene_idx, length in enumerate(scene_lengths):
            color = palette[scene_idx % len(palette)]
            for local_idx in range(length):
                img = np.full((height, width, 3), color, dtype=np.uint8)
                x = (local_idx * 8) % (width - 80)
                y = (height // 2) - 40
                img[y : y + 80, x : x + 80] = (255, 255, 255)

                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)

        # Flush the encoder so the trailing frames make it into the container.
        for packet in stream.encode():
            container.mux(packet)

    return out_path


if __name__ == "__main__":
    import sys

    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("tests/fixtures/synth_3scenes.mp4")
    )
    generate_synth_video(out)
    print(f"wrote {out}")
