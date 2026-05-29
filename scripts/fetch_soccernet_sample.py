"""One-shot SoccerNet download for the smoke test.

Pulls a single 720p game so the CV pipeline has real broadcast footage to
chew on. Not part of the library; this is a debug-time fetch.

Requires the NDA password emailed by the SoccerNet maintainers. Pass it
via the SOCCERNET_PASSWORD environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence the SoccerNet analytics call that crashes when the DNS lookup for
# ssl.google-analytics.com fails (common on restricted / offline networks).
# The actual download finishes fine; only the post-download telemetry breaks.
# ---------------------------------------------------------------------------
import importlib

_gmp_report = importlib.import_module("google_measurement_protocol.report")
_original_make_request = _gmp_report._make_request


def _silent_make_request(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Drop-in replacement that swallows network errors from GA telemetry."""
    try:
        return _original_make_request(*args, **kwargs)
    except Exception:
        # Analytics failure is never worth aborting a data download.
        return None


_gmp_report._make_request = _silent_make_request
# ---------------------------------------------------------------------------

from SoccerNet.Downloader import SoccerNetDownloader


def main() -> None:
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise SystemExit("Set SOCCERNET_PASSWORD before running.")

    out_dir = Path("data/soccernet").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(out_dir))
    downloader.password = password

    # One game at 720p. The downloader pulls the full match; you trim it
    # afterwards with ffmpeg into a short clip for the smoke test.
    downloader.downloadGames(
        files=["1_720p.mkv"],
        split=["test"],
    )

    print(f"\ndownloaded under {out_dir}")
    print("look for a .mkv a few directories deep (league/season/match/).")


if __name__ == "__main__":
    main()