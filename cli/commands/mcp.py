"""``parallax mcp install`` — register the Parallax MCP server in Claude Desktop.

Acceptance criteria implemented:
  AC01 — check ~/.parallax/env + parallax-mcp script exist; exit 1 if not
  AC02 — locate Claude Desktop config at platform-appropriate path
  AC03 — add/update mcpServers.parallax-mcp with absolute script path
  AC04 — create config file (with only mcpServers) if it does not exist
  AC05 — never overwrite keys other than mcpServers.parallax-mcp
  AC06 — print "MCP server registered. Restart Claude Desktop to apply."
  AC07 — already registered → print "Already registered." and exit 0
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Annotated, Any

import typer

from cli.commands._common import ENV_DIR as _ENV_DIR

app = typer.Typer(name="mcp", help="Manage MCP server integration.", no_args_is_help=True)
_MCP_SCRIPT_NAME = "parallax-mcp"


# ---------------------------------------------------------------------------
# Internal helpers (extracted for testability)
# ---------------------------------------------------------------------------


def _mcp_script_path() -> Path:
    """Return the expected absolute path of the parallax-mcp script."""
    return _ENV_DIR / "bin" / _MCP_SCRIPT_NAME


def _claude_config_path() -> Path:
    """Return the platform-appropriate Claude Desktop config file path (AC02)."""
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "Claude"
        return base / "claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    # Linux and everything else
    return Path.home() / ".config" / "claude" / "claude_desktop_config.json"


def _read_config(config_path: Path) -> dict[str, Any]:
    """Read and parse the Claude Desktop config JSON, or return {} if absent (AC04)."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return {}


def _write_config(config_path: Path, config: dict[str, Any]) -> None:
    """Write the config dict as JSON, creating parent directories as needed (AC04)."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


@app.command("install")
def install(
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show verbose output.")
    ] = False,
) -> None:
    """Register the Parallax MCP server in Claude Desktop."""
    # AC01 — ensure parallax install has been run
    script = _mcp_script_path()
    if not _ENV_DIR.exists() or not script.exists():
        typer.echo("Run `parallax install` first.")
        raise typer.Exit(1)

    # AC02 — locate Claude Desktop config
    config_path = _claude_config_path()
    if verbose:
        typer.echo(f"Claude Desktop config: {config_path}")

    # AC04 — read existing config (or start fresh)
    config = _read_config(config_path)

    # AC05 — only touch mcpServers.parallax-mcp
    mcp_servers = config.setdefault("mcpServers", {})
    entry_command = str(script)

    # AC07 — already registered with the same path
    existing = mcp_servers.get(_MCP_SCRIPT_NAME, {})
    if existing.get("command") == entry_command:
        typer.echo("Already registered.")
        return

    # AC03 — add or update the entry
    mcp_servers[_MCP_SCRIPT_NAME] = {"command": entry_command}

    # AC04 / AC05 — write back (only parallax-mcp key changed)
    _write_config(config_path, config)
    if verbose:
        typer.echo(f"Wrote config to {config_path}")

    # AC06
    typer.echo("MCP server registered. Restart Claude Desktop to apply.")
