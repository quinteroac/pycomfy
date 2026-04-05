"""FastMCP server entry point for Parallax."""

from __future__ import annotations

import importlib.metadata

from fastmcp import FastMCP

_VERSION = importlib.metadata.version("comfy-diffusion")

mcp = FastMCP("parallax-mcp", version=_VERSION)


def main() -> None:
    """Start the MCP server in stdio mode."""
    mcp.run(transport="stdio")
