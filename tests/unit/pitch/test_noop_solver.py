"""Noop pitch solver tests.

Confirms the registry resolves it and the contract holds: every frame
returns None.
"""

from __future__ import annotations

import numpy as np
import pytest

from datum.cv.pitch import PitchSolver, available, get, register


def test_noop_is_registered() -> None:
    cls = get("pitch-noop")
    assert cls.name == "pitch-noop"


def test_noop_returns_none() -> None:
    solver = get("pitch-noop")()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert solver.solve(frame) is None


def test_noop_reset_is_a_noop() -> None:
    solver = get("pitch-noop")()
    # Should not raise; default Tracker.reset is a pass-through, and the
    # noop solver does not override.
    solver.reset()


def test_available_lists_noop() -> None:
    assert "pitch-noop" in available()


def test_registry_rejects_duplicate_registration() -> None:
    # Define a throwaway subclass and try to register under an existing name.
    class _Bogus(PitchSolver):
        def solve(self, image: np.ndarray):  # type: ignore[override]
            return None

    with pytest.raises(ValueError, match="already registered"):
        register("pitch-noop")(_Bogus)


def test_registry_get_unknown_raises_with_available_names() -> None:
    with pytest.raises(KeyError, match="available:"):
        get("not-a-real-solver")
