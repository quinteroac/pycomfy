"""Main Typer application for the ``parallax`` CLI.

Entry point: ``python -m cli`` or via the ``parallax`` script defined in
``pyproject.toml``.
"""

from __future__ import annotations

import typer

from cli.commands.create import app as create_app
from cli.commands.edit import app as edit_app
from cli.commands.install import install
from cli.commands.jobs import app as jobs_app
from cli.commands.upscale import app as upscale_app

app = typer.Typer(
    name="parallax",
    help="Parallax CLI — run ComfyUI-backed inference pipelines from the command line.",
    no_args_is_help=True,
)

app.add_typer(create_app, name="create")
app.add_typer(edit_app,   name="edit")
app.add_typer(jobs_app,   name="jobs")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
