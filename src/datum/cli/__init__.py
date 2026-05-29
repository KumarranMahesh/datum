"""`datum` command-line entrypoint.

Thin Typer wrapper. Real logic lives in the library; the CLI is just one
of several possible clients. If you find yourself putting business logic
here, you're doing it wrong. Move it into the appropriate `datum.<stage>`
module and have the CLI call it.

Imports are deliberately lazy inside each command. Loading PyAV, torch,
and OpenCV at `datum --help` time pushes startup past one second and
makes the CLI feel like a Java tool. Keep the top of this file cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml
from rich.console import Console

if TYPE_CHECKING:
    from datum.cv import CvConfig
    from datum.ingest import IngestConfig

app = typer.Typer(
    name="datum",
    help="Single-camera scouting pipeline.",
    no_args_is_help=True,
    add_completion=False,
)

_console = Console()


@app.callback()
def _root() -> None:
    pass


@app.command()
def version() -> None:
    """Print the package version and exit."""
    from datum import __version__

    typer.echo(__version__)


@app.command()
def ingest(
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to the broadcast video file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="YAML config file. Defaults to IngestConfig() if omitted.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    runs_root: Annotated[
        Path,
        typer.Option(
            "--runs-root",
            help="Directory where run artifacts are written.",
        ),
    ] = Path("data/runs"),
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Re-run even if a complete manifest already exists.",
        ),
    ] = False,
) -> None:
    """Ingest a single broadcast video.

    Decodes the file, samples frames at the configured rate, classifies
    each shot, and writes a self-describing run directory under
    `--runs-root`.

    Re-running the same source with the same config is a no-op; the prior
    manifest is returned untouched. Pass `--overwrite` to force.
    """
    # Heavyweight imports stay inside the command. See module docstring.
    from datum.ingest import IngestConfig
    from datum.ingest import run as ingest_run

    cfg = _load_ingest_config(config)

    _console.print(f"[dim]source[/dim] {source}")
    _console.print(f"[dim]config[/dim]\n{cfg.model_dump_json(indent=2)}")

    try:
        manifest = ingest_run(source, cfg, runs_root=runs_root, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001
        # Re-raise as a Typer Exit so the exit code is non-zero but the
        # traceback is suppressed unless DATUM_DEBUG is set. Trading a bit
        # of Pythonic purity for a CLI that does not vomit a stack trace
        # at a user who just typo'd a path.
        import os

        if os.environ.get("DATUM_DEBUG"):
            raise
        _console.print(f"\n[bold red]ingest failed[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    counters = manifest.counters
    assert counters is not None, "pipeline must set counters on a successful run"

    _console.print()
    _console.print(f"[bold green]run_id[/bold green]          {manifest.run_id}")
    _console.print(f"  sampled frames     {counters.sampled_frames}")
    _console.print(f"  scenes             {counters.scenes}")
    _console.print(f"  pitch_frame_pct    {counters.pitch_frame_pct:.3f}")
    _console.print(f"  artifacts at       {(runs_root / manifest.run_id).resolve()}")


def _load_ingest_config(path: Path | None) -> IngestConfig:
    """Load an IngestConfig from YAML, or return defaults if path is None."""
    from datum.ingest import IngestConfig

    if path is None:
        return IngestConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return IngestConfig(**raw)


def _load_cv_config(path: Path | None, *, fallback_detector: str) -> CvConfig:
    """Load a CvConfig from YAML, or build a minimal one from fallback_detector."""
    from datum.cv import CvConfig

    if path is None:
        return CvConfig(detector=fallback_detector)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return CvConfig(**raw)


@app.command()
def cv(
    ingest_run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a completed ingest run directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    detector: Annotated[
        str,
        typer.Option(
            "--detector",
            "-d",
            help="Registered detector name. Ignored when --config is provided.",
        ),
    ] = "noop",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="YAML config file. When provided, --detector is ignored.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    cv_runs_root: Annotated[
        Path,
        typer.Option(
            "--cv-runs-root",
            help="Directory where CV run artifacts are written.",
        ),
    ] = Path("data/cv_runs"),
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Re-run even if a complete CV manifest already exists.",
        ),
    ] = False,
) -> None:
    """Run a detector on an ingest run.

    Reads the upstream ingest manifest, batches its frames, runs the
    configured detector, and writes one FrameDetections record per
    processed frame to a new run directory under `--cv-runs-root`.
    """
    from datum.cv import run as cv_run

    cfg = _load_cv_config(config, fallback_detector=detector)

    _console.print(f"[dim]ingest_run_dir[/dim] {ingest_run_dir}")
    _console.print(f"[dim]config[/dim]\n{cfg.model_dump_json(indent=2)}")

    try:
        manifest = cv_run(
            ingest_run_dir, cfg, cv_runs_root=cv_runs_root, overwrite=overwrite
        )
    except Exception as exc:  # noqa: BLE001
        import os

        if os.environ.get("DATUM_DEBUG"):
            raise
        _console.print(f"\n[bold red]cv run failed[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    counters = manifest.counters
    assert counters is not None, "pipeline must set counters on a successful run"

    _console.print()
    _console.print(f"[bold green]cv_run_id[/bold green]        {manifest.run_id}")
    _console.print(f"  detector             {manifest.detector_name}")
    _console.print(f"  ingest_run_id        {manifest.ingest_run_id}")
    _console.print(f"  frames_in            {counters.frames_in}")
    _console.print(f"  frames_processed     {counters.frames_processed}")
    _console.print(f"  frames_with_detect   {counters.frames_with_detections}")
    _console.print(f"  total_detections     {counters.total_detections}")
    _console.print(f"  detector_wall_s      {counters.detector_wall_s:.2f}")
    _console.print(f"  artifacts at         {(cv_runs_root / manifest.run_id).resolve()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
