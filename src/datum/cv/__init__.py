"""CV stage.

Public surface is the schemas, the `run` entrypoint, and the detector
and tracker registries.

Importing this package registers the bundled detector and tracker adapters
by name. Module-level execution in each adapter is intentionally cheap;
the heavy imports (torch, ultralytics) only run when an adapter is
*instantiated*. That keeps `datum --help` fast and lets contributors who
never plan to use YOLO skip installing ultralytics entirely.
"""

# Leaf schemas first; they have no inter-package dependencies.
from datum.cv.schemas import (
    SCHEMA_VERSION,
    CvConfig,
    CvCounters,
    CvManifest,
    CvRunId,
    DetectionBox,
    FrameDetections,
)

# Top-level pipeline entrypoint.
from datum.cv.pipeline import run

# Register bundled adapters by importing their modules. The side effect is
# the @register decorator call inside each module.
from datum.cv.detect import noop, ultralytics_yolo  # noqa: F401
from datum.cv.pitch import noop as _pitch_noop  # noqa: F401
from datum.cv.track import bytetrack as _track_bytetrack  # noqa: F401
from datum.cv.track import noop as _track_noop  # noqa: F401

__all__ = [
    "SCHEMA_VERSION",
    "CvConfig",
    "CvCounters",
    "CvManifest",
    "CvRunId",
    "DetectionBox",
    "FrameDetections",
    "run",
]
