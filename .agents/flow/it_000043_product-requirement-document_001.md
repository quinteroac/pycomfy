# Requirement: MCP Server Installation CLI Command

## Context
The user wants to add an installation command `parallax mcp install` to the `@parallax/cli`. This command will deploy the parallax MCP server configuration to the user's local AI clients, making it easier for them to consume the MCP capabilities without manual configuration.

## Goals
- Provide an interactive `parallax mcp install` command in the CLI.
- Allow users to select which AI clients (Claude, Gemini, GitHub Copilot, Codex) they want to install the MCP server to.
- Automatically update the respective configuration files for the chosen clients.

## User Stories

### US-001: Run Interactive MCP Install Command
**As an** end user, **I want** to execute `parallax mcp install` **so that** I can interactively configure my AI clients to use the MCP server.

**Acceptance Criteria:**
- [ ] Executing `parallax mcp install` starts an interactive prompt using `@clack/prompts`.
- [ ] The prompt displays a multi-select list of supported clients: Claude, Gemini, GitHub Copilot, Codex.
- [ ] The user can select one or more clients from the list.
- [ ] Typecheck / lint passes.
- [ ] **[UI stories only]** Visually verified in terminal.

### US-002: Apply MCP Configuration to Selected Clients
**As an** end user, **I want** the CLI to automatically update the configuration files of the clients I selected **so that** I don't have to manually edit their config files.

**Acceptance Criteria:**
- [ ] The CLI identifies the correct configuration file path for the selected clients based on the host OS (Linux, macOS, Windows).
- [ ] For Claude Desktop, the `claude_desktop_config.json` is updated with the parallax MCP server details.
- [ ] For Gemini, the appropriate configuration file is updated with the parallax MCP server details.
- [ ] For GitHub Copilot, the appropriate configuration file is updated with the parallax MCP server details.
- [ ] For Codex, the appropriate configuration file is updated with the parallax MCP server details.
- [ ] A success message is displayed indicating which clients were successfully configured.
- [ ] Typecheck / lint passes.

## Functional Requirements
- FR-1: The CLI must use the existing `@clack/prompts` library for interactive elements.
- FR-2: The tool must append or update the MCP server configuration into the client's configuration files (valid JSON or required format) without overwriting existing settings.
- FR-3: The CLI command must be implemented in the `@parallax/cli` package.

## Non-Goals (Out of Scope)
- Automatic detection of installed AI clients (users must select manually from the list).
- Uninstalling the MCP server from clients (this can be a separate command later).
- Supporting clients other than the 4 specified in the MVP (Claude, Gemini, GitHub Copilot, Codex).

## Resolved Questions
- **Gemini Config:** The Gemini CLI uses a JSON configuration file. Path: `~/.gemini/settings.json` (global). Format: `{"mcpServers": { "name": { "command": "...", "args": [...] } }}`.
- **GitHub Copilot Config:** GitHub Copilot CLI uses `~/.copilot/mcp-config.json`.
- **Codex Config:** If referring to Cursor, the path is `~/.cursor/mcp.json` (or `%USERPROFILE%\.cursor\mcp.json` on Windows). Format is identical to standard MCP `{"mcpServers": {...}}`. If referring to OpenAI Codex CLI, it's `~/.codex/config.toml`. Assume Cursor for standard MCP JSON format unless otherwise specified.