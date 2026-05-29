"""CV stage.

Public surface is the schemas, the `run` entrypoint (added when pipeline
lands), and the detector registry.

Importing this package registers the bundled detector adapters by name.
Module-level execution in each adapter is intentionally cheap; the heavy
imports (torch, ultralytics) only run when an adapter is *instantiated*.
That keeps `datum --help` fast and lets contributors who never plan to use
YOLO skip installing ultralytics entirely.
"""

# Register bundled adapters by importing their modules. The side effect is
# the @register decorator call.
from datum.cv.detect import noop, ultralytics_yolo  # noqa: F401
from datum.cv.pipeline import run
from datum.cv.schemas import (
    SCHEMA_VERSION,
    CvConfig,
    CvCounters,
    CvManifest,
    CvRunId,
    DetectionBox,
    FrameDetections,
)

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
