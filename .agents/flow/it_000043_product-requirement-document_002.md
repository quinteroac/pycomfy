# Requirement: MCP Server — Parallax CLI Tool Bindings

## Context

`@parallax/mcp` already has a skeleton (`src/index.ts`) with `McpServer` and `StdioServerTransport` wired up but no registered tools. This feature implements the full set of MCP tools that map to the existing `parallax` CLI commands (`create image`, `create video`, `create audio`, `edit image`, `upscale image`). Each tool invokes the CLI as a child process via `Bun.spawn`, captures stdout/stderr, and returns the output file path to the AI client. The goal is to allow AI clients (Claude Desktop, GitHub Copilot, etc.) to trigger media generation through the MCP protocol without any HTTP server or manual CLI usage.

## Goals

- Expose all existing `parallax` CLI commands as MCP tools inside `@parallax/mcp`.
- Each tool invokes the `parallax` CLI binary as a subprocess via `Bun.spawn`, forwarding arguments built from the tool's input schema.
- Return the output file path (and any relevant metadata) to the calling AI client upon success.
- The MCP server is startable via `bun run start` in `packages/parallax_mcp/` and registerable in any MCP-compatible AI client config.

## User Stories

### US-001: Create Image via MCP Tool

**As a** developer using an AI client (e.g. Claude Desktop), **I want** to call a `create_image` MCP tool **so that** the AI client can generate an image by invoking `parallax create image` without me running CLI commands manually.

**Acceptance Criteria:**
- [ ] An MCP tool named `create_image` is registered on the server with an input schema matching `parallax create image` options: `model` (required), `prompt` (required), `negativePrompt`, `width`, `height`, `steps`, `cfg`, `seed`, `output`, `modelsDir`.
- [ ] Invoking the tool spawns `bun run src/index.ts create image --model <model> --prompt <prompt> [...]` via `Bun.spawn` inside `@parallax/cli`.
- [ ] On success, the tool result includes the resolved output file path.
- [ ] On failure (non-zero exit code), the tool result includes the stderr output as an error message.
- [ ] Typecheck / lint passes.
- [ ] Visually verified: Claude Desktop (or equivalent) can call `create_image` and receive the output path.

### US-002: Create Video via MCP Tool

**As a** developer using an AI client, **I want** to call a `create_video` MCP tool **so that** the AI client can trigger `parallax create video` and receive the output path.

**Acceptance Criteria:**
- [ ] An MCP tool named `create_video` is registered with input schema matching `parallax create video` options: `model` (required), `prompt` (required), `input`, `width`, `height`, `length`, `steps`, `cfg`, `seed`, `output`, `modelsDir`.
- [ ] Spawns the CLI subprocess correctly and returns output file path on success or stderr on failure.
- [ ] Typecheck / lint passes.

### US-003: Create Audio via MCP Tool

**As a** developer using an AI client, **I want** to call a `create_audio` MCP tool **so that** the AI client can trigger `parallax create audio`.

**Acceptance Criteria:**
- [ ] An MCP tool named `create_audio` is registered with input schema matching `parallax create audio` options: `model` (required), `prompt` (required), `length`, `steps`, `cfg`, `bpm`, `lyrics`, `seed`, `output`, `modelsDir`.
- [ ] Spawns the CLI subprocess and returns output file path on success or stderr on failure.
- [ ] Typecheck / lint passes.

### US-004: Edit Image via MCP Tool

**As a** developer using an AI client, **I want** to call an `edit_image` MCP tool **so that** the AI client can trigger `parallax edit image`.

**Acceptance Criteria:**
- [ ] An MCP tool named `edit_image` is registered with input schema matching `parallax edit image` options: `model` (required), `prompt` (required), `input` (required), `subjectImage`, `width`, `height`, `steps`, `cfg`, `seed`, `output`, `image2`, `image3`, `noLora`, `modelsDir`.
- [ ] Spawns the CLI subprocess and returns output file path on success or stderr on failure.
- [ ] Typecheck / lint passes.

### US-005: Upscale Image via MCP Tool

**As a** developer using an AI client, **I want** to call an `upscale_image` MCP tool **so that** the AI client can trigger `parallax upscale image`.

**Acceptance Criteria:**
- [ ] An MCP tool named `upscale_image` is registered with input schema matching `parallax upscale image` options: `model` (required), `prompt` (required), `input` (required), `checkpoint`, `esrganCheckpoint`, `latentUpscaleCheckpoint`, `negativePrompt`, `width`, `height`, `steps`, `cfg`, `seed`, `output`, `outputBase`, `modelsDir`.
- [ ] Spawns the CLI subprocess and returns output file path on success or stderr on failure.
- [ ] Typecheck / lint passes.

### US-006: MCP Server Startup and Client Registration

**As a** developer, **I want** to start the MCP server with `bun run start` and register it in my AI client config **so that** the tools become available to the AI client.

**Acceptance Criteria:**
- [ ] `bun run start` in `packages/parallax_mcp/` starts the server without errors.
- [ ] The server responds to MCP `tools/list` with all 5 tools (`create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`).
- [ ] The README or inline comment documents the config snippet needed to register the server in Claude Desktop / GitHub Copilot.
- [ ] Typecheck / lint passes.
- [ ] Visually verified: AI client config registers the server and tools appear in the client.

## Functional Requirements

- FR-1: All MCP tools must be registered in `packages/parallax_mcp/src/index.ts` using `server.tool()` from `@modelcontextprotocol/sdk`.
- FR-2: Subprocess invocation must use `Bun.spawn` pointing at the `@parallax/cli` entry point (`packages/parallax_cli/src/index.ts`), not a compiled binary, to avoid build prerequisites.
- FR-3: Tool input schemas must use `zod` (or the SDK's built-in schema helpers) and must match the CLI flag names (camelCase mapping to kebab-case flags).
- FR-4: Boolean flags (e.g. `noLora`) must be conditionally appended — only when `true` — since the CLI uses presence-based flags.
- FR-5: The tool result on success must be a text content block containing at minimum the resolved output file path.
- FR-6: The tool result on failure must be a text content block containing the process stderr, and `isError: true`.
- FR-7: No new top-level dependencies beyond what is already in `packages/parallax_mcp/package.json` unless strictly necessary (prefer `zod` which is likely already available via the SDK).

## Non-Goals (Out of Scope)

- Implementing `edit video` (the CLI command itself is not yet implemented — `notImplemented()` stub).
- Streaming output back to the AI client in real-time (return final result only).
- Automatic discovery of available models (hard-code the same lists as the CLI's `models/registry.ts`).
- HTTP transport — stdio only for this iteration.
- Input file upload via MCP (file paths must be local paths the server can access).

## Open Questions

- None.
