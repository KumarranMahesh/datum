"""Render bounding boxes from a CV run onto its source frames.

This is a debug tool, not part of the library surface. It loads a CV run's
detections.jsonl, fetches the corresponding source images from the upstream
ingest run, draws every detection on top, and writes annotated jpgs to an
output directory.

Usage:
    uv run python scripts/render_detections.py <cv_run_dir> [options]

Example:
    uv run python scripts/render_detections.py data/cv_runs/cv-61d0783af5ef --limit 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from datum.cv.schemas import CvManifest, FrameDetections

# BGR per class name. Anything else gets the default amber.
_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 255, 0),
    "sports ball": (0, 200, 255),
}
_DEFAULT_COLOR: tuple[int, int, int] = (0, 165, 255)


def _draw_detections(image: "cv2.typing.MatLike", record: FrameDetections) -> None:
    for det in record.detections:
        color = _COLORS.get(det.class_name, _DEFAULT_COLOR)
        cv2.rectangle(
            image,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            color,
            2,
        )
        label = f"{det.class_name} {det.confidence:.2f}"
        y_label = max(15, int(det.y1) - 4)
        cv2.putText(
            image,
            label,
            (int(det.x1), y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def render(
    cv_run_dir: Path,
    *,
    out_dir: Path,
    ingest_runs_root: Path,
    limit: int | None,
) -> None:
    cv_run_dir = cv_run_dir.resolve()
    ingest_runs_root = ingest_runs_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cv_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no CV manifest at {manifest_path}")

    manifest = CvManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    ingest_run_dir = ingest_runs_root / manifest.ingest_run_id
    if not ingest_run_dir.exists():
        raise FileNotFoundError(
            f"could not find ingest run at {ingest_run_dir}. "
            "Pass --ingest-runs-root if your ingest runs live elsewhere."
        )

    detections_path = cv_run_dir / "detections.jsonl"
    if not detections_path.exists():
        raise FileNotFoundError(f"no detections.jsonl at {detections_path}")

    image_dir = ingest_run_dir / "images"

    rendered = 0
    total_processed = 0

    with detections_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            record = FrameDetections.model_validate_json(line)
            total_processed += 1

            if not record.detections:
                continue

            # The ingest writer chose jpg or png based on IngestConfig.
            # We do not have the config handy here, so try both.
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

            _draw_detections(img, record)

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
            "No frames had detections. Check that you ran a real detector "
            "(not noop) and that the source actually contains people / balls."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render bounding boxes from a CV run onto source frames."
    )
    parser.add_argument(
        "cv_run_dir",
        type=Path,
        help="CV run directory containing manifest.json and detections.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/rendered/<cv_run_id>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of frames rendered.",
    )
    parser.add_argument(
        "--ingest-runs-root",
        type=Path,
        default=Path("data/runs"),
        help="Where to look for the upstream ingest run. Default: data/runs.",
    )
    args = parser.parse_args()

    cv_run_dir = args.cv_run_dir
    out_dir = args.out if args.out is not None else Path("data/rendered") / cv_run_dir.name

    render(
        cv_run_dir,
        out_dir=out_dir,
        ingest_runs_root=args.ingest_runs_root,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
