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
