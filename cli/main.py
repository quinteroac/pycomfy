"""Main Typer application for the ``parallax`` CLI.

Entry point: ``python -m cli`` or via the ``parallax`` script defined in
``pyproject.toml``.
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from cli._version import __version__
from cli.commands.create import app as create_app
from cli.commands.edit import app as edit_app
from cli.commands.frontend import app as frontend_app
from cli.commands.install import install
from cli.commands.jobs import app as jobs_app
from cli.commands.mcp import app as mcp_app
from cli.commands.ms import app as ms_app
from cli.commands.upscale import app as upscale_app

app = typer.Typer(
    name="parallax",
    help="Parallax CLI — run ComfyUI-backed inference pipelines from the command line.",
)

app.add_typer(create_app, name="create")
app.add_typer(edit_app,   name="edit")
app.add_typer(frontend_app, name="frontend")
app.add_typer(jobs_app,   name="jobs")
app.add_typer(mcp_app,    name="mcp")
app.add_typer(ms_app,     name="ms")
app.add_typer(upscale_app, name="upscale")
app.command("install")(install)


@app.command(
    "async",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Queue any parallax subcommand as an async job and return a job ID immediately.",
)
def async_command(ctx: typer.Context) -> None:
    """Usage: parallax async <subcommand> [args...]

    Example::

        parallax async create image --model anima --prompt '1girl, bangs'
    """
    from cli._async import _uv_path, enqueue_cmd

    subcmd = ctx.args
    if not subcmd:
        typer.echo("Error: provide a subcommand after 'async'.", err=True)
        raise typer.Exit(code=1)

    cmd = [_uv_path(), "run", "parallax"] + subcmd
    enqueue_cmd(cmd)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"parallax {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = None,
) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
