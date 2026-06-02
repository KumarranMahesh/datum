"""Pitch stage data contracts.

A `Homography` is a 3x3 image-to-pitch matrix plus diagnostic fields.
Multiplied with a homogeneous image point `(u, v, 1)` it yields a
homogeneous pitch point `(X*w, Y*w, w)` which divides out to a pitch
coordinate in metres.

Pitch coordinate convention: origin at the centre spot, x-axis along the
long (sideline-to-sideline) axis, y-axis along the short (goal-to-goal)
axis, in metres. Standard FIFA full-size pitch is 105 by 68 metres, so
coordinates land in roughly x in [-52.5, 52.5] and y in [-34, 34]. Other
pitch sizes are an open question; for now assume standard.

A FramePitch with `homography is None` records that the solver ran on
this frame and could not produce a reliable result. The absence of a
FramePitch record means the frame was never processed (e.g. filtered
upstream).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datum.ingest.schemas import Sha256Hex

SCHEMA_VERSION: int = 1

PitchRunId = Annotated[
    str, Field(pattern=r"^pitch-[a-f0-9]{12}$", min_length=18, max_length=18)
]
"""Pitch run id of the form 'pitch-<12 hex>'. Derived deterministically
from `sha256(cv_run_id + ':' + pitch_config_hash)`."""


class Homography(BaseModel):
    """One 3x3 image-to-pitch homography plus the diagnostics needed to
    decide how much to trust it.

    Matrix is stored row-major as a nested list of floats so the JSON
    serialisation is human-readable. Numerical ops should pull it into
    a numpy array (3, 3) at the boundary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    matrix: list[list[float]]
    """3x3 row-major. Element [i][j] is the (i, j) entry of the matrix."""

    method: str = Field(min_length=1)
    """Identifier of the solver that produced this matrix.

    Persisted so downstream can know how the homography was estimated
    (e.g. 'noop', 'lines-ransac-dlt-v1', 'tvcalib-pretrained') without
    re-instantiating the solver.
    """

    inlier_count: int = Field(ge=0)
    """How many keypoint correspondences were inliers of the final fit.

    Zero is meaningful: a solver that returns a Homography with zero
    inliers is reporting that the fit is structurally suspicious. The
    pipeline will record it but downstream consumers should filter on
    inlier_count and reprojection_error_px before using H.
    """

    reprojection_error_px: float = Field(ge=0.0)
    """Mean reprojection error of the inliers in pixels.

    Anything above 5 to 10 px on a 1080p broadcast is usually too noisy
    to use for tactical analysis at metre-scale resolution.
    """

    @field_validator("matrix")
    @classmethod
    def _matrix_must_be_3x3(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError(f"matrix must be 3x3; got shape {len(v)}x{len(v[0]) if v else 0}")
        return v


class PitchSolverConfig(BaseModel):
    """Frozen config snapshot for one pitch run.

    Anything that affects the emitted homographies belongs here. Things
    that only affect progress display (logging verbosity, image debug
    dumps) do not; they should not change the run id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    solver: str = Field(min_length=1)
    """Registered solver name. Looked up via `datum.cv.pitch.get`."""

    solver_config: dict[str, Any] = Field(default_factory=dict)
    """Opaque dict passed to the solver constructor. Hashed into
    config_hash, so different values produce different run ids."""

    pitch_only: bool = True
    """Skip frames whose ingest scene_kind is not PITCH or REPLAY. The
    homography solver is meaningless on crowd or graphic shots."""


class FramePitch(BaseModel):
    """The homography solution for one frame.

    `homography is None` means the solver ran and could not converge.
    Frames that were skipped by `pitch_only` produce no FramePitch record
    at all; their absence is the signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_idx: int = Field(ge=0)
    source_frame_idx: int = Field(ge=0)
    homography: Homography | None
    solver_wall_ms: float = Field(ge=0.0)
    """Per-frame solver time in milliseconds. Useful for spotting outliers
    where the solver got stuck on a hard frame."""


class PitchCounters(BaseModel):
    """Run-level aggregates.

    `frames_with_homography / frames_processed` is the first sanity
    signal. A solver that lands below 0.7 on broadcast football is
    probably misconfigured or being run on the wrong frames.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_in: int = Field(ge=0)
    """Frames available in the parent CV run."""

    frames_processed: int = Field(ge=0)
    """Frames the solver actually ran on (after pitch_only filtering)."""

    frames_with_homography: int = Field(ge=0)
    """Frames where the solver returned a non-None Homography."""

    mean_reprojection_error_px: float | None = Field(default=None, ge=0.0)
    """Mean over the frames with a homography. None when no homographies
    were produced."""

    solver_wall_s: float = Field(ge=0.0)


class PitchManifest(BaseModel):
    """Top-level record for a completed pitch run.

    Written at `<pitch_run_dir>/manifest.json` once the run finishes.
    Atomic write; the file exists in full or not at all.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: PitchRunId
    cv_run_id: str
    """The CV run this pitch run consumed. The audit chain runs through
    here back to the ingest run and the source video."""

    solver_name: str
    """Echo of config.solver, for ergonomic reads."""

    config: PitchSolverConfig
    config_hash: Sha256Hex

    started_at: datetime
    finished_at: datetime | None = None
    counters: PitchCounters | None = None


__all__ = [
    "SCHEMA_VERSION",
    "FramePitch",
    "Homography",
    "PitchCounters",
    "PitchManifest",
    "PitchRunId",
    "PitchSolverConfig",
]
