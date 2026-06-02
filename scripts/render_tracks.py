"""Render colored boxes per track_id from a track run onto its source frames.

Debug tool, not part of the library surface. Each `track_id` gets a
deterministic distinct color; identity persistence shows up as same-color
boxes across frames following the same player.

Usage:
    uv run python scripts/render_tracks.py <track_run_dir> [options]

Example:
    uv run python scripts/render_tracks.py data/track_runs/track-abcdef012345 --limit 30
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2

from datum.cv.schemas import CvManifest
from datum.cv.track.schemas import FrameTracks, TrackManifest


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Distinct BGR color per track_id, deterministic for the same id.

    Golden-angle hue spread keeps adjacent track_ids visually separated.
    """
    hue = (track_id * 137.508) % 360.0
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _draw_tracks(image: "cv2.typing.MatLike", record: FrameTracks) -> None:
    for t in record.tracks:
        color = _color_for_track(t.track_id)
        cv2.rectangle(
            image,
            (int(t.x1), int(t.y1)),
            (int(t.x2), int(t.y2)),
            color,
            2,
        )
        label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
        y_label = max(15, int(t.y1) - 4)
        cv2.putText(
            image,
            label,
            (int(t.x1), y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def render(
    track_run_dir: Path,
    *,
    out_dir: Path,
    cv_runs_root: Path,
    ingest_runs_root: Path,
    limit: int | None,
) -> None:
    track_run_dir = track_run_dir.resolve()
    cv_runs_root = cv_runs_root.resolve()
    ingest_runs_root = ingest_runs_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    track_manifest_path = track_run_dir / "manifest.json"
    if not track_manifest_path.exists():
        raise FileNotFoundError(f"no track manifest at {track_manifest_path}")
    track_manifest = TrackManifest.model_validate_json(
        track_manifest_path.read_text(encoding="utf-8")
    )

    cv_run_dir = cv_runs_root / track_manifest.cv_run_id
    cv_manifest_path = cv_run_dir / "manifest.json"
    if not cv_manifest_path.exists():
        raise FileNotFoundError(
            f"could not find CV run at {cv_run_dir}. "
            "Pass --cv-runs-root if CV runs live elsewhere."
        )
    cv_manifest = CvManifest.model_validate_json(
        cv_manifest_path.read_text(encoding="utf-8")
    )

    ingest_run_dir = ingest_runs_root / cv_manifest.ingest_run_id
    if not ingest_run_dir.exists():
        raise FileNotFoundError(
            f"could not find ingest run at {ingest_run_dir}. "
            "Pass --ingest-runs-root if ingest runs live elsewhere."
        )

    image_dir = ingest_run_dir / "images"
    tracks_path = track_run_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"no tracks.jsonl at {tracks_path}")

    rendered = 0
    total_processed = 0

    with tracks_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            record = FrameTracks.model_validate_json(line)
            total_processed += 1

            if not record.tracks:
                continue

            image_path: Path | None = None
            for ext in ("jpg", "png"):
                candidate = image_dir / f"{record.frame_idx:06d}.{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                print(f"  no source image for frame_idx={record.frame_idx}; skipping")
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  cv2.imread failed for {image_path}")
                continue

            _draw_tracks(img, record)

            out_path = out_dir / f"{record.frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), img)
            rendered += 1

            if limit is not None and rendered >= limit:
                break

    print(
        f"\nrendered {rendered} frames (of {total_processed} processed) "
        f"to {out_dir}"
    )
    if rendered == 0:
        print(
            "No frames had tracked observations. Either the tracker found "
            "nothing to track (synth video?), or the upstream CV run had no "
            "detections."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render colored boxes per track_id from a track run."
    )
    parser.add_argument(
        "track_run_dir",
        type=Path,
        help="Track run directory containing manifest.json and tracks.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/rendered/<track_run_id>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of frames rendered.",
    )
    parser.add_argument(
        "--cv-runs-root",
        type=Path,
        default=Path("data/cv_runs"),
        help="Where to look for the upstream CV run.",
    )
    parser.add_argument(
        "--ingest-runs-root",
        type=Path,
        default=Path("data/runs"),
        help="Where to look for the upstream ingest run.",
    )
    args = parser.parse_args()

    track_run_dir = args.track_run_dir
    out_dir = (
        args.out
        if args.out is not None
        else Path("data/rendered") / track_run_dir.name
    )

    render(
        track_run_dir,
        out_dir=out_dir,
        cv_runs_root=args.cv_runs_root,
        ingest_runs_root=args.ingest_runs_root,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
