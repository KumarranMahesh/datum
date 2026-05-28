"""`datum` command-line entrypoint.

Thin Typer wrapper. Real logic lives in the library; the CLI is just one
of several possible clients. If you find yourself putting business logic
here, you're doing it wrong — move it into the appropriate `datum.<stage>`
module and have the CLI call it.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="datum",
    help="Single-camera scouting pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


# Subcommands are wired in lazily so `--help` doesn't pay the cost of
# importing torch on a cold CLI invocation.
@app.callback()
def _root() -> None:
    pass


@app.command()
def version() -> None:
    """Print the package version and exit."""
    from datum import __version__
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
