"""FastMCP server entry point for Parallax."""

from __future__ import annotations

import importlib.metadata

from fastmcp import FastMCP

_VERSION = importlib.metadata.version("comfy-diffusion")

mcp = FastMCP("parallax-mcp", version=_VERSION)

# Register inference tools (US-001)
from mcp.tools.inference import (  # noqa: E402
    create_audio,
    create_image,
    create_video,
    edit_image,
    upscale_image,
)

# Register job status tools (US-002, US-003)
from mcp.tools.jobs import get_job_status, wait_for_job  # noqa: E402

for _fn in (
    create_image,
    create_video,
    create_audio,
    edit_image,
    upscale_image,
    get_job_status,
    wait_for_job,
):
    mcp.tool()(_fn)


def main() -> None:
    """Start the MCP server in stdio mode."""
    mcp.run(transport="stdio")
