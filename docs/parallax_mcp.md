# @parallax/mcp

MCP (Model Context Protocol) server that exposes Parallax inference capabilities as tools for Claude and other MCP-compatible AI clients.

## Role in the monorepo

Allows Claude Desktop, GitHub Copilot, or any MCP client to call image/video/audio generation directly from a conversation. Delegates all work to `@parallax/cli` — it is a thin tool-registration layer, not an inference engine.

## Location

`packages/parallax_mcp/`

## Stack

- Bun runtime
- `@modelcontextprotocol/sdk` ^1.0.0
- `@parallax/sdk` for shared types
- Transport: stdio (standard for MCP clients)

## Starting the server

```bash
# from packages/parallax_mcp/
bun run start

# from repo root
cd packages/parallax_mcp && bun run start
```

The server uses stdio transport and waits for MCP protocol messages. It has no HTTP port — register it in your AI client config (see below) and the client manages the process lifecycle.

## Available Tools

| Tool | Description |
|------|-------------|
| `create_image` | Generate an image from a text prompt (`parallax create image`) |
| `create_video` | Generate a video from text or an input image (`parallax create video`) |
| `create_audio` | Generate audio from a text prompt (`parallax create audio`) |
| `edit_image` | Edit an existing image with a prompt (`parallax edit image`) |
| `upscale_image` | Upscale an image using ESRGAN or latent upscale (`parallax upscale image`) |

## Registering in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "parallax": {
      "command": "bun",
      "args": ["run", "/absolute/path/to/packages/parallax_mcp/src/index.ts"]
    }
  }
}
```

Replace `/absolute/path/to/` with the actual path to your local clone of this repository.
After saving, restart Claude Desktop — the 5 Parallax tools will appear in the tool list.

## Registering in GitHub Copilot (VS Code / CLI)

Create or edit `.vscode/mcp.json` in your workspace (VS Code with GitHub Copilot extension):

```json
{
  "servers": {
    "parallax": {
      "type": "stdio",
      "command": "bun",
      "args": ["run", "/absolute/path/to/packages/parallax_mcp/src/index.ts"]
    }
  }
}
```

For the GitHub Copilot CLI agent, create `.github/copilot/mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "parallax": {
      "command": "bun",
      "args": ["run", "${workspaceFolder}/packages/parallax_mcp/src/index.ts"]
    }
  }
}
```

## Registering in Claude Code (CLI)

Add to `.claude/settings.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "parallax": {
      "command": "bun",
      "args": ["run", "/absolute/path/to/packages/parallax_mcp/src/index.ts"]
    }
  }
}
```

## Dependencies

```json
{
  "@modelcontextprotocol/sdk": "^1.0.0",
  "@parallax/sdk": "workspace:*",
  "zod": "3"
}
```
