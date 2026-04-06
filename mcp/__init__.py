"""MCP package — extends the installed mcp SDK namespace with Parallax tools.

This file shadows the installed ``mcp`` package so that ``mcp/main.py`` and
``mcp/__main__.py`` are importable as ``mcp.main``.  It first adds the SDK's
``mcp/`` directory to ``__path__`` and then re-exports everything the SDK's
own ``__init__.py`` exports, preserving full backwards compatibility with any
code that does ``from mcp import McpError`` etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Locate the installed mcp SDK (in .venv/site-packages/mcp/) and add it to
# __path__ so sub-packages (mcp.client, mcp.server, mcp.shared, mcp.types…)
# are still found there by relative-import machinery.
_this = Path(__file__).resolve()
for _entry in sys.path:
    if not _entry:
        continue  # skip '' (the current directory, which points back to us)
    _candidate = Path(_entry) / "mcp" / "__init__.py"
    if _candidate.resolve() != _this and _candidate.exists():
        _sdk_dir = str(_candidate.parent)
        if _sdk_dir not in __path__:
            __path__.append(_sdk_dir)  # type: ignore[name-defined]
        break

# Re-export everything the SDK's mcp/__init__.py exports so that
# ``from mcp import McpError`` (used by fastmcp) and all other SDK consumers
# continue to work exactly as before.
from .client.session import ClientSession  # noqa: F401
from .client.session_group import ClientSessionGroup  # noqa: F401
from .client.stdio import StdioServerParameters, stdio_client  # noqa: F401
from .server.session import ServerSession  # noqa: F401
from .server.stdio import stdio_server  # noqa: F401
from .shared.exceptions import McpError, UrlElicitationRequiredError  # noqa: F401
from .types import (  # noqa: F401
    CallToolRequest,
    ClientCapabilities,
    ClientNotification,
    ClientRequest,
    ClientResult,
    CompleteRequest,
    CreateMessageRequest,
    CreateMessageResult,
    CreateMessageResultWithTools,
    ErrorData,
    GetPromptRequest,
    GetPromptResult,
    Implementation,
    IncludeContext,
    InitializedNotification,
    InitializeRequest,
    InitializeResult,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsResult,
    LoggingLevel,
    LoggingMessageNotification,
    Notification,
    PingRequest,
    ProgressNotification,
    PromptsCapability,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
    ResourcesCapability,
    ResourceUpdatedNotification,
    RootsCapability,
    SamplingCapability,
    SamplingContent,
    SamplingContextCapability,
    SamplingMessage,
    SamplingMessageContentBlock,
    SamplingToolsCapability,
    ServerCapabilities,
    ServerNotification,
    ServerRequest,
    ServerResult,
    SetLevelRequest,
    StopReason,
    SubscribeRequest,
    Tool,
    ToolChoice,
    ToolResultContent,
    ToolsCapability,
    ToolUseContent,
    UnsubscribeRequest,
)
from .types import Role as SamplingRole  # noqa: F401

__all__ = [
    "CallToolRequest",
    "ClientCapabilities",
    "ClientNotification",
    "ClientRequest",
    "ClientResult",
    "ClientSession",
    "ClientSessionGroup",
    "CompleteRequest",
    "CreateMessageRequest",
    "CreateMessageResult",
    "CreateMessageResultWithTools",
    "ErrorData",
    "GetPromptRequest",
    "GetPromptResult",
    "Implementation",
    "IncludeContext",
    "InitializeRequest",
    "InitializeResult",
    "InitializedNotification",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "ListPromptsRequest",
    "ListPromptsResult",
    "ListResourcesRequest",
    "ListResourcesResult",
    "ListToolsResult",
    "LoggingLevel",
    "LoggingMessageNotification",
    "McpError",
    "Notification",
    "PingRequest",
    "ProgressNotification",
    "PromptsCapability",
    "ReadResourceRequest",
    "ReadResourceResult",
    "Resource",
    "ResourcesCapability",
    "ResourceUpdatedNotification",
    "RootsCapability",
    "SamplingCapability",
    "SamplingContent",
    "SamplingContextCapability",
    "SamplingMessage",
    "SamplingMessageContentBlock",
    "SamplingRole",
    "SamplingToolsCapability",
    "ServerCapabilities",
    "ServerNotification",
    "ServerRequest",
    "ServerResult",
    "ServerSession",
    "SetLevelRequest",
    "StdioServerParameters",
    "StopReason",
    "SubscribeRequest",
    "Tool",
    "ToolChoice",
    "ToolResultContent",
    "ToolsCapability",
    "ToolUseContent",
    "UnsubscribeRequest",
    "UrlElicitationRequiredError",
    "stdio_client",
    "stdio_server",
]
