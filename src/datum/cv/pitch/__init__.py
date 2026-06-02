"""Pitch homography adapter contract.

Solvers map a broadcast image to an image-to-pitch homography matrix.
Implementations register under a stable string name; pipelines look them
up via `datum.cv.pitch.get` and feed them BGR uint8 frames.

The interface is narrow on purpose: one method, one return type. Solver-
specific knobs (Hough thresholds, RANSAC iteration counts, model paths)
belong in the adapter constructor and the YAML config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from datum.cv.pitch.schemas import Homography


class PitchSolver(ABC):
    """Implement this and register under a stable string name.

    The pipeline guarantees:

      * frames arrive in chronological order
      * frames are BGR uint8 at full source resolution
      * the solver is called once per processed frame; statefulness is
        permitted (some solvers benefit from temporal smoothing) but
        adapters that hold state must implement `reset()` if they want
        the pipeline's scene-cut hooks to behave sensibly
    """

    name: str = ""

    @abstractmethod
    def solve(self, image: np.ndarray) -> Homography | None:
        """Return the image-to-pitch homography for one frame.

        Return None when no reliable solution can be found (extreme close-
        up, low light, off-pitch shot that slipped past pitch-only
        filtering, etc). The pipeline records the absence and moves on;
        downstream stages already handle a None homography per frame.
        """
        ...

    def reset(self) -> None:
        """Clear any temporal state. Default is a no-op.

        Stateless solvers should leave this alone. Adapters that track
        previous-frame information across calls (Kalman over H, temporal
        keypoint smoothing) should override.
        """
        return None


# --- registry ---------------------------------------------------------------
#
# Same tiny dict + decorator pattern as datum.cv.detect and
# datum.cv.track. Entrypoint plugins are still overkill at this scale.

_REGISTRY: dict[str, type[PitchSolver]] = {}


def register(name: str) -> Callable[[type[PitchSolver]], type[PitchSolver]]:
    def _wrap(cls: type[PitchSolver]) -> type[PitchSolver]:
        if name in _REGISTRY:
            raise ValueError(
                f"pitch solver '{name}' already registered: {_REGISTRY[name]!r}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get(name: str) -> type[PitchSolver]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        available_names = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"unknown pitch solver '{name}'. available: {available_names}"
        ) from e


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "PitchSolver",
    "available",
    "get",
    "register",
]
