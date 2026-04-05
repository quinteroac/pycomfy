"""Main Typer application for the ``parallax`` CLI.

Entry point: ``python -m cli`` or via the ``parallax`` script defined in
``pyproject.toml``.
"""

from __future__ import annotations

import typer

from cli.commands.create import app as create_app
from cli.commands.edit import app as edit_app
from cli.commands.upscale import app as upscale_app

app = typer.Typer(
    name="parallax",
    help="Parallax CLI — run ComfyUI-backed inference pipelines from the command line.",
    no_args_is_help=True,
)

app.add_typer(create_app, name="create")
app.add_typer(edit_app,   name="edit")
app.add_typer(upscale_app, name="upscale")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
