"""Ingest stage.

Public surface is the schemas and (once it lands) the `run` entrypoint.
Internals (reader, sampler, scenes, writer) are not re-exported on purpose;
nothing outside this package should depend on the PyAV-specific machinery.
"""

from datum.ingest.pipeline import run
from datum.ingest.schemas import (
    SCHEMA_VERSION,
    FrameRecord,
    IngestConfig,
    IngestCounters,
    IngestManifest,
    RunId,
    SceneKind,
    SceneSegment,
    Sha256Hex,
    SourceInfo,
)

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
    "run",
]
