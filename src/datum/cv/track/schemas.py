"""Track stage data contracts.

Read by every downstream stage that consumes tracklets (spatial, features,
embeddings). Schema changes here ripple, so additions deserve a beat of
thought before they land. Bump SCHEMA_VERSION when any model in this file
changes shape.

The chain to the upstream CV run is recorded explicitly on
`TrackManifest.cv_run_id`. Given a track run directory alone, you can
trace back to the exact detections, CV config, ingest run, and source
that produced the input to this stage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from datum.cv.schemas import DetectionBox
from datum.ingest.schemas import Sha256Hex

SCHEMA_VERSION: int = 1

TrackRunId = Annotated[
    str, Field(pattern=r"^track-[a-f0-9]{12}$", min_length=18, max_length=18)
]
"""Track run identifier of the form 'track-<12 hex>'.

The hex is the first 12 chars of `sha256(cv_run_id + ':' + tracker_config_hash)`.
Identical (cv_run_id, tracker_config) inputs always map to the same
track_run_id, which is what makes re-runs idempotent.
"""


class TrackerConfig(BaseModel):
    """Frozen config snapshot for one track run.

    Anything that affects the emitted tracklets belongs here. Things that
    only affect progress display (logging verbosity, tqdm style) do not,
    because they should not change the run id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tracker: str = Field(min_length=1)
    """Registered tracker name. Looked up via `datum.cv.track.get`."""

    tracker_config: dict[str, Any] = Field(default_factory=dict)
    """Opaque dict passed to the tracker constructor.

    Hashed into config_hash, so different values produce different run ids.
    Keep this JSON-compatible (no numpy arrays, no callables) for the hash
    to be stable across machines.
    """

    reset_on_scene_cut: bool = False
    """When True, the pipeline calls `tracker.reset()` at every scene
    boundary recorded by upstream ingest. Useful for broadcasts: identities
    almost never carry across a hard cut, so resetting prevents the tracker
    from blindly stitching a player on one camera to an unrelated player
    on the next. Off by default for the simplest deterministic semantics.
    """


class TrackedDetection(DetectionBox):
    """A persisted detection with a track identity attached.

    Inherits every DetectionBox field and adds `track_id`. Stored in
    `tracks.jsonl` rather than `detections.jsonl`.
    """

    track_id: int = Field(ge=0)


class FrameTracks(BaseModel):
    """All tracked observations for a single processed frame.

    A frame with no detections (or no detections that the tracker chose to
    track) is allowed to have an empty `tracks` list. Frames the CV run
    never processed (filtered by `pitch_only`) produce no FrameTracks
    record at all; the absence is the signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_idx: int = Field(ge=0)
    source_frame_idx: int = Field(ge=0)
    tracks: list[TrackedDetection]


class TrackCounters(BaseModel):
    """Run-level aggregates.

    `unique_tracks` divided by run length is a useful first sanity signal:
    a 60-second broadcast clip should typically end with somewhere between
    20 and 60 unique tracks (10 to 14 players visible at most moments,
    with ID switches and reappearances after occlusions). Numbers far
    outside that band usually mean the tracker is fragmenting identities.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_in: int = Field(ge=0)
    """Frames present in the parent CV run."""

    frames_processed: int = Field(ge=0)
    """Frames the tracker actually saw."""

    detections_in: int = Field(ge=0)
    """Total detections seen across all processed frames."""

    tracked_observations: int = Field(ge=0)
    """Total (frame, track_id) pairs emitted. Less than or equal to
    detections_in, because the tracker may discard unmatched detections."""

    unique_tracks: int = Field(ge=0)
    """Number of distinct track_id values seen across the whole run."""

    tracker_wall_s: float = Field(ge=0.0)


class TrackManifest(BaseModel):
    """Top-level record for a completed track run.

    Written at `<track_run_dir>/manifest.json` once the run finishes.
    Manifest writing is atomic; the file either exists in full or not at all.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: TrackRunId
    cv_run_id: str
    """The CV run consumed by this track run. The chain audits through here."""

    tracker_name: str
    """Echo of config.tracker. Kept on the manifest for ergonomic reads."""

    config: TrackerConfig
    config_hash: Sha256Hex

    started_at: datetime
    finished_at: datetime | None = None
    counters: TrackCounters | None = None


__all__ = [
    "SCHEMA_VERSION",
    "FrameTracks",
    "TrackCounters",
    "TrackManifest",
    "TrackRunId",
    "TrackedDetection",
    "TrackerConfig",
]
