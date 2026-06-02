"""Noop pitch solver.

Always returns None. Used for:

  * unit-testing the pitch pipeline without committing to a real solver
  * confirming downstream consumers handle a None homography correctly
    (since the same path will trigger on hard real frames)
  * benchmarking pipeline overhead exclusive of solver wall-clock
"""

from __future__ import annotations

import numpy as np

from datum.cv.pitch import PitchSolver, register
from datum.cv.pitch.schemas import Homography


@register("pitch-noop")
class NoopPitchSolver(PitchSolver):
    """Returns None for every frame."""

    def solve(self, image: np.ndarray) -> Homography | None:  # noqa: ARG002
        return None


__all__ = ["NoopPitchSolver"]
