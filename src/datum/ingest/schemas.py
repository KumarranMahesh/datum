"""Ingest stage data contracts.

Every downstream stage reads these. Breaking changes here cost the project
a migration, so resist the urge to put implementation details on these
models. If a field is useful only to the ingest internals, keep it out.

`schema_version` is the safety net. Bump it when any model in this file
changes shape. Downstream loaders refuse unknown versions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: int = 1

RunId = Annotated[str, Field(pattern=r"^ingest-[a-f0-9]{12}$", min_length=19, max_length=19)]
"""Run identifier of the form 'ingest-<12 hex chars>'.

The hex segment is the first 12 chars of sha256(source_sha256 + config_hash)
so a (source, config) pair always maps to the same run id. Re-runs are
no-ops by id, which is what makes the pipeline reproducible.
"""

Sha256Hex = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$", min_length=64, max_length=64)]


class SceneKind(str, Enum):
    """Coarse classification of a broadcast shot.

    Anything not PITCH or REPLAY is excluded from downstream stages by
    default. A PITCH shot that turns out to be unusable (extreme close-up,
    dropped feed) is still PITCH; the downstream stage flags it on its own.
    """

    PITCH = "pitch"
    REPLAY = "replay"
    CROWD = "crowd"
    BENCH = "bench"
    GRAPHIC = "graphic"
    UNKNOWN = "unknown"


class IngestConfig(BaseModel):
    """Frozen config snapshot. Hashed into the run id.

    Adding or removing a field here changes config_hash for every previous
    run, so treat additions as a breaking change to the artifact format and
    bump SCHEMA_VERSION at the same time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_fps: float = Field(default=5.0, gt=0, le=60)
    """Target frames per second after sampling.

    Source fps is preserved when source_fps <= sample_fps.
    """

    min_resolution: int = Field(default=720, ge=240)
    """Refuse input shorter than this on the short axis.

    480p broadcast homography is unreliable. Explicit failure beats silent
    garbage downstream.
    """

    scene_cut_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    """Histogram-distance threshold above which a frame is treated as a
    scene boundary. Higher means fewer cuts."""

    scene_classifier: str = Field(default="histogram-v1", min_length=1)
    """String identifier of the scene classifier.

    The pipeline rejects unknown values rather than silently falling back.
    """

    image_format: str = Field(default="jpg", pattern=r"^(jpg|png)$")
    image_quality: int = Field(default=92, ge=1, le=100)
    """Only meaningful when image_format == 'jpg'."""


class SourceInfo(BaseModel):
    """Identifying information about a source video file.

    `sha256` is the content hash of the file bytes, not of any decoded form.
    This is what makes the run id reproducible across machines.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: Sha256Hex
    duration_s: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    container: str
    video_codec: str


class FrameRecord(BaseModel):
    """One sampled frame.

    `source_frame_idx` is the position in the SOURCE stream;
    `frame_idx` is the position in the SAMPLED stream. Most downstream
    stages want `frame_idx`. Logs, debug tooling, and the spatial stage
    want `source_frame_idx`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_idx: int = Field(ge=0)
    source_frame_idx: int = Field(ge=0)
    pts_us: int = Field(ge=0)
    """Presentation timestamp in microseconds from the start of the source."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    scene_id: int = Field(ge=0)
    scene_kind: SceneKind

    image_path: str
    """Path relative to the run directory."""


class SceneSegment(BaseModel):
    """Contiguous run of source frames sharing a single broadcast shot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: int = Field(ge=0)
    start_source_frame_idx: int = Field(ge=0)
    end_source_frame_idx: int = Field(ge=0)
    """Inclusive."""

    kind: SceneKind
    confidence: float = Field(ge=0.0, le=1.0)


class IngestCounters(BaseModel):
    """Run-level aggregates.

    `pitch_frame_pct` is the most useful sanity signal in this struct.
    A healthy broadcast sits roughly between 0.55 and 0.75. Below 0.40 the
    feed is dominated by replays, crowd shots, or graphics, and downstream
    embedding quality will suffer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_frames: int = Field(ge=0)
    sampled_frames: int = Field(ge=0)
    scenes: int = Field(ge=0)
    pitch_frame_pct: float = Field(ge=0.0, le=1.0)


class IngestManifest(BaseModel):
    """Top-level record for a completed run.

    Written as JSON at `<run_dir>/manifest.json` once the run finishes.
    The writer uses tmp + rename so the file either exists in full or not
    at all. A manifest with `finished_at == None` is a run in progress; a
    manifest with `finished_at != None and counters is None` indicates a
    corrupted manifest.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: RunId
    source: SourceInfo
    config: IngestConfig
    config_hash: Sha256Hex

    started_at: datetime
    finished_at: datetime | None = None

    counters: IngestCounters | None = None


__all__ = [
    "SCHEMA_VERSION",
    "FrameRecord",
    "IngestConfig",
    "IngestCounters",
    "IngestManifest",
    "RunId",
    "SceneKind",
    "SceneSegment",
    "Sha256Hex",
    "SourceInfo",
]
