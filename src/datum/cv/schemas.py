"""CV stage data contracts.

These are read by every downstream stage that consumes per-frame
detections (tracking, pose, spatial). Schema changes ripple, so additions
deserve a beat of thought before they land. Bump SCHEMA_VERSION when any
model in this file changes shape.

The chain to the upstream ingest run is recorded explicitly on
`CvManifest.ingest_run_id`. That makes a CV run self-describing: given a
cv run directory alone, you can trace back to the exact source and ingest
config that produced its inputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from datum.ingest.schemas import Sha256Hex

SCHEMA_VERSION: int = 1

CvRunId = Annotated[
    str, Field(pattern=r"^cv-[a-f0-9]{12}$", min_length=15, max_length=15)
]
"""CV run identifier of the form 'cv-<12 hex>'.

The hex is the first 12 chars of `sha256(ingest_run_id + ':' + cv_config_hash)`.
Identical (ingest_run_id, cv_config) inputs always map to the same cv_run_id,
which is what makes re-runs idempotent.
"""


class CvConfig(BaseModel):
    """Frozen config snapshot for one CV run.

    Anything that affects the produced detections belongs here. Things
    that only affect the *progress display* (logging, tqdm style) do not,
    because they should not change the run id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: str = Field(min_length=1)
    """Registered detector name. Looked up via `datum.cv.detect.get`."""

    detector_config: dict[str, Any] = Field(default_factory=dict)
    """Opaque dict passed to the detector constructor.

    Hashed into config_hash, so different values produce different run ids.
    Keep this JSON-compatible (no numpy arrays, no callables) for the hash
    to be stable across machines.
    """

    batch_size: int = Field(default=8, ge=1, le=128)
    """Frames per detector call. Higher saturates GPU; too high OOMs."""

    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    """Detections below this confidence are dropped before persistence."""

    pitch_only: bool = True
    """When True, skip frames whose ingest scene_kind is not PITCH or REPLAY.

    Saves significant compute on broadcasts dominated by crowd, bench, or
    graphic shots. Set False if you specifically want detections on those
    frames (e.g. to study sideline behaviour).
    """


class DetectionBox(BaseModel):
    """One persisted detection in pixel space.

    Mirrors the in-memory `datum.cv.detect.Detection` dataclass, with the
    class_name added so a CV run is interpretable without the detector
    model loaded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: float = Field(gt=0.0)
    y2: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)


class FrameDetections(BaseModel):
    """All detections for a single processed frame.

    An empty `detections` list means the detector ran and found nothing.
    Frames that were skipped (e.g. pitch_only filtered them out) produce
    no record at all; the absence is the signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_idx: int = Field(ge=0)
    source_frame_idx: int = Field(ge=0)
    detections: list[DetectionBox]
    inference_ms: float = Field(ge=0.0)
    """Detector wall-clock spent on the batch this frame belonged to,
    divided proportionally by frame count. Useful for spotting slow batches."""


class CvCounters(BaseModel):
    """Run-level aggregates.

    `frames_processed < frames_in` is normal when `pitch_only=True` and the
    broadcast contains crowd, bench, or graphic shots. Compare these two
    against `pitch_frame_pct` in the parent ingest manifest if a ratio
    looks off.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_in: int = Field(ge=0)
    """Frames present in the parent ingest run."""

    frames_processed: int = Field(ge=0)
    """Frames the detector actually ran on."""

    frames_with_detections: int = Field(ge=0)
    total_detections: int = Field(ge=0)
    detector_wall_s: float = Field(ge=0.0)


class CvManifest(BaseModel):
    """Top-level record for a completed CV run.

    Written at `<cv_run_dir>/manifest.json` once the run finishes.
    Manifest writing is atomic; the file either exists in full or not at all.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: CvRunId
    ingest_run_id: str
    """The ingest run consumed by this CV run. The chain audits through here."""

    detector_name: str
    """Echo of config.detector. Kept on the manifest for ergonomic reads
    (so you do not have to dig into nested config to know what produced
    the detections)."""

    class_map: dict[int, str]
    """The detector's class_id -> name map at run time.

    Persisted so downstream stages can interpret class_id values without
    instantiating the detector. JSON serialises int keys as strings; pydantic
    coerces them back on load.
    """

    config: CvConfig
    config_hash: Sha256Hex

    started_at: datetime
    finished_at: datetime | None = None
    counters: CvCounters | None = None


__all__ = [
    "SCHEMA_VERSION",
    "CvConfig",
    "CvCounters",
    "CvManifest",
    "CvRunId",
    "DetectionBox",
    "FrameDetections",
]
