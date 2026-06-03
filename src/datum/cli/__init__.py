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
    from datum.cv.pitch.schemas import PitchSolverConfig
    from datum.cv.track.schemas import TrackerConfig
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


def _load_track_config(path: Path | None, *, fallback_tracker: str) -> TrackerConfig:
    """Load a TrackerConfig from YAML, or build a minimal one from fallback_tracker."""
    from datum.cv.track.schemas import TrackerConfig

    if path is None:
        return TrackerConfig(tracker=fallback_tracker)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return TrackerConfig(**raw)


def _load_pitch_config(
    path: Path | None, *, fallback_solver: str
) -> PitchSolverConfig:
    """Load a PitchSolverConfig from YAML, or build a minimal one from fallback_solver."""
    from datum.cv.pitch.schemas import PitchSolverConfig

    if path is None:
        return PitchSolverConfig(solver=fallback_solver)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return PitchSolverConfig(**raw)


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


@app.command()
def track(
    cv_run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a completed CV run directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    tracker: Annotated[
        str,
        typer.Option(
            "--tracker",
            "-t",
            help="Registered tracker name. Ignored when --config is provided.",
        ),
    ] = "bytetrack",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="YAML config file. When provided, --tracker is ignored.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    track_runs_root: Annotated[
        Path,
        typer.Option(
            "--track-runs-root",
            help="Directory where track run artifacts are written.",
        ),
    ] = Path("data/track_runs"),
    ingest_runs_root: Annotated[
        Path,
        typer.Option(
            "--ingest-runs-root",
            help=(
                "Where to look for the parent ingest run. "
                "Only consulted when the config sets reset_on_scene_cut=true."
            ),
        ),
    ] = Path("data/runs"),
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Re-run even if a complete track manifest already exists.",
        ),
    ] = False,
) -> None:
    """Run a tracker on a CV run.

    Reads the upstream CV manifest, walks its per-frame detections in
    chronological order, runs the configured tracker, and writes one
    FrameTracks record per processed frame to a new run directory under
    `--track-runs-root`.

    When the config sets `reset_on_scene_cut: true`, the pipeline also
    reads scene boundaries from the parent ingest run at
    `--ingest-runs-root/<ingest_run_id>/frames.jsonl` and resets the
    tracker at every boundary.
    """
    from datum.cv.track import run as track_run

    cfg = _load_track_config(config, fallback_tracker=tracker)

    _console.print(f"[dim]cv_run_dir[/dim] {cv_run_dir}")
    _console.print(f"[dim]config[/dim]\n{cfg.model_dump_json(indent=2)}")

    try:
        manifest = track_run(
            cv_run_dir,
            cfg,
            track_runs_root=track_runs_root,
            ingest_runs_root=ingest_runs_root,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        import os

        if os.environ.get("DATUM_DEBUG"):
            raise
        _console.print(f"\n[bold red]track run failed[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    counters = manifest.counters
    assert counters is not None, "pipeline must set counters on a successful run"

    _console.print()
    _console.print(f"[bold green]track_run_id[/bold green]     {manifest.run_id}")
    _console.print(f"  tracker              {manifest.tracker_name}")
    _console.print(f"  cv_run_id            {manifest.cv_run_id}")
    _console.print(f"  frames_in            {counters.frames_in}")
    _console.print(f"  frames_processed     {counters.frames_processed}")
    _console.print(f"  detections_in        {counters.detections_in}")
    _console.print(f"  tracked_observations {counters.tracked_observations}")
    _console.print(f"  unique_tracks        {counters.unique_tracks}")
    _console.print(f"  tracker_wall_s       {counters.tracker_wall_s:.2f}")
    _console.print(
        f"  artifacts at         {(track_runs_root / manifest.run_id).resolve()}"
    )


@app.command()
def pitch(
    cv_run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a completed CV run directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    solver: Annotated[
        str,
        typer.Option(
            "--solver",
            "-s",
            help="Registered pitch solver name. Ignored when --config is provided.",
        ),
    ] = "pitch-noop",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="YAML config file. When provided, --solver is ignored.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    pitch_runs_root: Annotated[
        Path,
        typer.Option(
            "--pitch-runs-root",
            help="Directory where pitch run artifacts are written.",
        ),
    ] = Path("data/pitch_runs"),
    ingest_runs_root: Annotated[
        Path,
        typer.Option(
            "--ingest-runs-root",
            help=(
                "Where to look for the parent ingest run. The pitch solver "
                "reads images from here, not from the CV run."
            ),
        ),
    ] = Path("data/runs"),
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Re-run even if a complete pitch manifest already exists.",
        ),
    ] = False,
) -> None:
    """Solve image-to-pitch homography on the images of a CV run's parent ingest.

    Reads the CV manifest for the chain audit, resolves the parent ingest
    run, walks its frames, calls the configured solver, and writes one
    FramePitch record per processed frame under `--pitch-runs-root`.
    """
    from datum.cv.pitch import run as pitch_run

    cfg = _load_pitch_config(config, fallback_solver=solver)

    _console.print(f"[dim]cv_run_dir[/dim] {cv_run_dir}")
    _console.print(f"[dim]config[/dim]\n{cfg.model_dump_json(indent=2)}")

    try:
        manifest = pitch_run(
            cv_run_dir,
            cfg,
            pitch_runs_root=pitch_runs_root,
            ingest_runs_root=ingest_runs_root,
            overwrite=overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        import os

        if os.environ.get("DATUM_DEBUG"):
            raise
        _console.print(f"\n[bold red]pitch run failed[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    counters = manifest.counters
    assert counters is not None, "pipeline must set counters on a successful run"

    mean_err = counters.mean_reprojection_error_px
    mean_err_str = f"{mean_err:.2f}" if mean_err is not None else "n/a"

    _console.print()
    _console.print(f"[bold green]pitch_run_id[/bold green]     {manifest.run_id}")
    _console.print(f"  solver               {manifest.solver_name}")
    _console.print(f"  cv_run_id            {manifest.cv_run_id}")
    _console.print(f"  frames_in            {counters.frames_in}")
    _console.print(f"  frames_processed     {counters.frames_processed}")
    _console.print(f"  frames_with_H        {counters.frames_with_homography}")
    _console.print(f"  mean_reproj_err_px   {mean_err_str}")
    _console.print(f"  solver_wall_s        {counters.solver_wall_s:.2f}")
    _console.print(
        f"  artifacts at         {(pitch_runs_root / manifest.run_id).resolve()}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
