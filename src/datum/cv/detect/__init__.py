"""Detection adapter contract.

Every detector (YOLO, RT-DETR, your in-house thing) implements `Detector`
and registers itself via `@register`. Pipelines look detectors up by name
from the registry, so the only thing the rest of the codebase ever sees is
this interface.

If you find yourself wanting to add a method here, stop. The interface is
deliberately narrow. Push detector-specific knobs into the constructor and
the YAML config instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


# Frames come in as (N, H, W, 3) uint8 in BGR, the same convention OpenCV uses.
# No normalisation here; that's the detector's problem.
FrameBatch = np.ndarray


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection from one frame. Coordinates are pixel-space.

    `class_id` follows the COCO person-class convention by default (id=0).
    Subclassing detectors that emit ball/referee/keeper distinctions should
    document their own class map in their adapter, not redefine this struct.
    """
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    detections: list[Detection]
    # Carry through per-batch metadata the tracker downstream actually needs.
    frame_indices: list[int]


class Detector(ABC):
    """Implement this and register under a stable string name.

    The pipeline guarantees:
        - frames arrive in chronological order
        - frames are BGR uint8
        - frames have already been resized to the model's expected input
          *only if* the YAML config asked for it; otherwise raw broadcast
          resolution
    """

    name: str = ""

    @abstractmethod
    def detect(self, frames: FrameBatch) -> DetectionBatch:
        """Return detections for every frame in `frames`.

        Detectors must be deterministic with respect to (frames, config).
        Stochastic NMS, dropout, etc. are not acceptable at inference.
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
            # Loud failure on collision. Silent overrides have wasted
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
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown detector '{name}'. available: {available}") from e


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
