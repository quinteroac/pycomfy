# @parallax/mcp

MCP (Model Context Protocol) server that exposes Parallax inference capabilities as tools for Claude and other MCP-compatible AI clients.

## Role in the monorepo

Allows Claude Code (or any MCP client) to call image/video/audio generation directly from a conversation. Delegates all work to `@parallax/ms` — it is a thin tool-registration layer, not an inference engine.

## Location

`packages/parallax_mcp/`

## Stack

- Bun runtime
- `@modelcontextprotocol/sdk` ^1.0.0
- `@parallax/sdk` for shared types
- Transport: stdio (default for Claude Code integration)

## Running

```bash
# from repo root
bun run mcp

# direct
cd packages/parallax_mcp && bun run src/index.ts
```

## Registering in Claude Code

Add to `.claude/settings.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "parallax": {
      "command": "bun",
      "args": ["run", "/path/to/packages/parallax_mcp/src/index.ts"]
    }
  }
}
```

## Planned Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate an image from a text prompt |
| `edit_image` | Edit an existing image with a prompt |
| `generate_video` | Generate a video from text or image |
| `list_models` | List available inference models |
| `get_job_status` | Poll job status by ID |

## Dependencies

```json
{
  "@modelcontextprotocol/sdk": "^1.0.0",
  "@parallax/sdk": "workspace:*"
}
```
