"""Top-level pytest configuration.

Session-scoped fixtures live here. The synthetic video is generated once
per session and reused across every test that asks for it (encoding takes
roughly half a second, but tests run thousands of times in CI).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.synth_video import generate_synth_video


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Path to the canonical 3-scene synthetic mp4.

    Session-scoped, so every test reuses the same file. If a test mutates
    the file it will leak across tests; tests must treat the file as
    read-only.
    """
    out = tmp_path_factory.mktemp("synth_video") / "synth_3scenes.mp4"
    generate_synth_video(out)
    yield out
