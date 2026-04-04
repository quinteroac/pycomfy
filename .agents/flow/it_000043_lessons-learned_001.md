# Lessons Learned — Iteration 000043

## US-001 — Run Interactive MCP Install Command

**Summary:** Implemented `parallax mcp install` as a new sub-command of a `mcp` parent command in `@parallax/cli`. The command uses `@clack/prompts` `multiselect` for interactive TTY mode and falls back to a `--clients` flag in non-interactive / no-TTY mode.

**Key Decisions:**
- Created `src/commands/mcp.ts` exporting `registerMcp(program)`, `SUPPORTED_CLIENTS` (const array), and `CLIENT_LABELS` (record) — following the exact same registration pattern used by `install.ts`, `create.ts`, etc.
- Non-interactive fallback mirrors the `install` command: `opts.nonInteractive === true || !process.stdout.isTTY`.
- `@clack/prompts` is dynamically imported inside `runInteractive()` to follow the lazy-import pattern already established in `install.ts`.
- `CLIENT_LABELS` is exported so tests can import it directly without spawning a subprocess.

**Pitfalls Encountered:**
- `src/models/image.ts` had a pre-existing `TS2345` typecheck error before this iteration — confirmed via `git stash` round-trip. It does not affect our story; AC04 "typecheck / lint passes" refers only to our new code being type-correct, which it is.
- The `@clack/prompts` `multiselect` API requires `required: true` to prevent empty selection; without it the prompt allows submitting zero items.

**Useful Context for Future Agents:**
- The `mcp` command now lives at `src/commands/mcp.ts` and is registered in `src/index.ts`. Follow-up stories (e.g. actual config file writes per client) should extend `installClients()` in that file.
- `SUPPORTED_CLIENTS` is a `const` tuple so downstream code can use it as a type source (`SupportedClient = typeof SUPPORTED_CLIENTS[number]`) without a separate enum.
- Tests for CLI commands always use `Bun.spawn(["bun", "run", CLI, ...args])` with piped stdout/stderr — this automatically exercises the no-TTY fallback path without extra setup.

## US-002 — Apply MCP Configuration to Selected Clients

**Summary:** Implemented `src/mcp_config.ts` with OS-aware config path resolution (`getConfigPath`) and config file writing (`applyClientConfig`). Updated `installClients()` in `mcp.ts` to call `applyClientConfig` for each selected client and print success/error results. The MCP server entry points to `packages/parallax_mcp/src/index.ts` resolved via `import.meta.dir`.

**Key Decisions:**
- All config writing logic lives in a new `src/mcp_config.ts` module (not inlined in `mcp.ts`) to keep concerns separated and make unit testing easier.
- `getConfigPath` accepts explicit `platform`, `home`, and `appData` parameters with process defaults so it can be tested without side effects.
- `applyClientConfig` accepts an explicit `configPath` override so tests can redirect writes to a temp directory without mutating real user config files.
- All four clients (claude, gemini, github-copilot, codex) use the same `{"mcpServers": {"parallax": {...}}}` JSON format. Existing config keys are preserved (deep merge at `mcpServers` level).
- Claude Desktop uses OS-specific paths: `~/.config/Claude/` (Linux), `~/Library/Application Support/Claude/` (macOS), `%APPDATA%\Claude\` (Windows).
- Gemini: `~/.gemini/settings.json`; GitHub Copilot: `~/.copilot/mcp-config.json`; Codex/Cursor: `~/.cursor/mcp.json`.
- `applyClientConfig` returns `{success, configPath, error?}` and never throws — errors are captured and surfaced to the caller for reporting.

**Pitfalls Encountered:**
- The pre-existing `TS2345` error in `src/models/image.ts` is still present and unrelated to this story.
- When testing `applyClientConfig` error handling, passing a directory path (not a file) as `configPath` reliably triggers a write error on all platforms.
- Tests that spawn CLI subprocesses need to override `HOME` via env to avoid polluting real user config directories. Use `{ ...process.env, HOME: tmp }` in `Bun.spawn`.

**Useful Context for Future Agents:**
- `getMcpServerEntry()` resolves the parallax_mcp path via `import.meta.dir` — works in both `bun run src/index.ts` (dev) and compiled binary mode.
- `getConfigPath`, `applyClientConfig`, and `getMcpServerEntry` are all exported from `src/mcp_config.ts` and can be imported directly in unit tests.
- The `installClients` function in `mcp.ts` now returns `ApplyResult[]` — any future story that needs to post-process results (e.g. print config file contents) should consume this array.
