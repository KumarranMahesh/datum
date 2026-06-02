"""Tracking adapter contract.

Trackers are stateful by design: identities are stitched across frames as
they arrive. Adapters implement `Tracker.update()` and `Tracker.reset()`,
and register under a stable string name. The pipeline owns chronological
order and calls `reset()` between independent sequences.

If you want to add a method here, stop and think first. The interface is
deliberately narrow. Adapter-specific knobs belong in the constructor and
the YAML config, not on the abstract base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from datum.cv.detect import Detection


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One detection enriched with a track identity.

    Mirrors `datum.cv.detect.Detection` plus `track_id`. Adapters emit
    this; the pipeline converts to the persisted TrackedDetection pydantic
    model before writing to disk.
    """

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class Tracker(ABC):
    """Implement this and register under a stable string name.

    The pipeline guarantees:

      * `update()` is called once per processed frame, in chronological
        order, with that frame's detections
      * `frame_idx` is monotonically non-decreasing across calls
      * `reset()` is called once before the first `update()` and may be
        called again between independent sequences
    """

    name: str = ""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        *,
        frame_idx: int,
    ) -> list[TrackObservation]:
        """Process one frame's detections and return tracked observations.

        Output length need not equal input length. A detection may be
        discarded (no track assignment) and an existing track may survive
        a frame without a matched detection (no output for that frame,
        but the track is kept alive internally).

        `track_id` values must be stable across calls for the same logical
        object, and unique per logical object within a contiguous sequence.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state.

        After `reset()`, the next `track_id` may restart from any value
        the adapter chooses (usually 0 or 1). The pipeline does not
        interpret `track_id` values, only their stability within a
        contiguous run.
        """
        ...


# --- registry ---------------------------------------------------------------
#
# Same pattern as datum.cv.detect. A dict and a decorator carry the project
# until there are many more adapters than the few it will ever ship with.

_REGISTRY: dict[str, type[Tracker]] = {}


def register(name: str) -> Callable[[type[Tracker]], type[Tracker]]:
    def _wrap(cls: type[Tracker]) -> type[Tracker]:
        if name in _REGISTRY:
            raise ValueError(
                f"tracker '{name}' already registered: {_REGISTRY[name]!r}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get(name: str) -> type[Tracker]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        available_names = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"unknown tracker '{name}'. available: {available_names}"
        ) from e


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "TrackObservation",
    "Tracker",
    "available",
    "get",
    "register",
    "run",
]


# Pipeline import lives at the bottom so the Tracker class and the registry
# helpers are already defined when pipeline.py back-imports `get` from here.
from datum.cv.track.pipeline import run  # noqa: E402
