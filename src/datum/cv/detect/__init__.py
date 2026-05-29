"""Detection adapter contract.

Every detector (YOLO, RT-DETR, your in-house thing) implements `Detector`
and registers itself via `@register`. Pipelines look detectors up by name
from the registry, so the only thing the rest of the codebase ever sees is
this interface.

If you find yourself wanting to add a method here, stop and think first.
The interface is deliberately narrow. Push detector-specific knobs into
the constructor and the YAML config instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Frames arrive as (N, H, W, 3) uint8 in BGR, the same convention OpenCV uses.
# Normalisation is the detector's responsibility, not the pipeline's.
FrameBatch = np.ndarray


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection in pixel space.

    `class_id` is detector-specific. The mapping from id to human name lives
    on `Detector.class_map`; the pipeline persists the name alongside the id
    so downstream stages do not need the detector loaded to interpret a run.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Detections for a batch of N frames.

    `per_frame[i]` is the list of detections for the i-th frame in the
    input batch. A frame with no detections is an empty list at that index.
    The pipeline owns the mapping from batch position to global frame_idx.
    """

    per_frame: list[list[Detection]]


class Detector(ABC):
    """Implement this and register under a stable string name.

    The pipeline guarantees:

      * frames arrive in chronological order
      * frames are BGR uint8
      * frames have already been resized to the model's expected input only
        if the config asked for it; otherwise raw broadcast resolution
    """

    name: str = ""

    @property
    @abstractmethod
    def class_map(self) -> dict[int, str]:
        """Mapping from class_id to human-readable name.

        Persisted alongside the run so downstream stages can read detections
        without the detector model loaded. Keep this stable across calls for
        a given detector instance; the pipeline reads it once per run.
        """
        ...

    @abstractmethod
    def detect(self, frames: FrameBatch) -> DetectionBatch:
        """Return per-frame detections for the input batch.

        Implementations should be deterministic with respect to
        (frames, config). Avoid stochastic NMS or dropout at inference; the
        pipeline relies on reproducible outputs for caching and debugging.

        `len(returned.per_frame)` must equal `len(frames)`.
        """
        ...


# --- registry ---------------------------------------------------------------
#
# Tiny on purpose. Entrypoint plugins are overkill here; a dict and a
# decorator carry the whole project until there are >50 adapters, and
# that day is not coming soon.

_REGISTRY: dict[str, type[Detector]] = {}


def register(name: str) -> Callable[[type[Detector]], type[Detector]]:
    def _wrap(cls: type[Detector]) -> type[Detector]:
        if name in _REGISTRY:
            # Loud failure on collision. Silent overrides have cost
            # roughly a person-year of debugging time across past projects.
            raise ValueError(f"detector '{name}' already registered: {_REGISTRY[name]!r}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get(name: str) -> type[Detector]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        available_names = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown detector '{name}'. available: {available_names}") from e


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Detection",
    "DetectionBatch",
    "Detector",
    "FrameBatch",
    "available",
    "get",
    "register",
]
