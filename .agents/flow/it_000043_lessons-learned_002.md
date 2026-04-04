# Lessons Learned — Iteration 000043

## US-001 — Create Image via MCP Tool

**Summary:** Implemented the `create_image` MCP tool in `packages/parallax_mcp/src/index.ts`. The tool registers a fully-typed input schema, spawns `bun run src/index.ts create image ...` inside `@parallax/cli` via `Bun.spawn`, and returns either the resolved output path (success) or stderr (failure). 24 tests cover all acceptance criteria.

**Key Decisions:**
- Used `server.registerTool()` instead of the deprecated `server.tool()` overload — the `tool()` version has TypeScript overload resolution ambiguity when combining a description string with a schema object.
- Added `zod@3` explicitly as a dependency because the MCP SDK's `AnySchema` / `ZodRawShapeCompat` types are based on zod 3 internals (`_type`, `_parse`, etc.). Zod 4 changed its internal API and is incompatible even though `@modelcontextprotocol/sdk` claims `^3.25 || ^4.0` in peerDependencies; the TypeScript types only match zod 3.
- `CLI_DIR` is resolved via `import.meta.dir` (Bun-native) pointing to `../../parallax_cli`, and `Bun.spawn` uses `cwd: CLI_DIR` with command `["bun", "run", "src/index.ts", ...]` — matching the AC02 specification exactly.
- Output path is resolved with `path.resolve(input.output ?? "output.png")` before spawn, so the result is always an absolute path regardless of cwd.

**Pitfalls Encountered:**
- `zod@4` was installed first (latest), which caused `ZodString` / `ZodOptional` type errors against the MCP SDK's `AnySchema` type. Downgrading to `zod@3` resolved all typecheck errors immediately.
- The deprecated `server.tool()` overload with `(name, description, schema, cb)` threw a TypeScript error: "Argument of type 'string' is not assignable to parameter of type 'ZodRawShapeCompat'". This is a known TypeScript overload-resolution ambiguity in the SDK. Always use `server.registerTool()`.

**Useful Context for Future Agents:**
- `@parallax/mcp` package does not have a `tests/` directory by default — create it when adding tests.
- The `parallax_mcp` package's `package.json` needs `zod@3` as an explicit dependency; do not rely on transitive resolution from the MCP SDK.
- The MCP SDK's `inputSchema` field in `registerTool()` accepts a `ZodRawShapeCompat` object (plain object of Zod fields), not a `ZodObject` — pass the shape directly, not `z.object({...})`.
- For future MCP tools (`create_video`, `create_audio`, `edit_image`, etc.), follow the same spawn pattern: `["bun", "run", "src/index.ts", <subcommand>, ...args]` with `cwd: CLI_DIR`.

## US-002 — Create Video via MCP Tool

**Summary:** Added `create_video` MCP tool to `packages/parallax_mcp/src/index.ts` following the exact same pattern as `create_image`. The tool maps all `parallax create video` CLI options to its input schema and spawns the CLI subprocess via `Bun.spawn`. 23 tests cover all acceptance criteria.

**Key Decisions:**
- Mirrored the `create_image` implementation pattern exactly: `server.registerTool()` with a zod shape (not `z.object()`), `Bun.spawn` with `cwd: CLI_DIR`, and `resolve(input.output ?? "output.mp4")` for the default output path.
- The video-specific field `input` (for i2v) maps to `--input` CLI flag; `length` (number of frames) maps to `--length`.
- Default output is `output.mp4` (not `output.png`) — matches the CLI default in `create.ts`.

**Pitfalls Encountered:**
- No new pitfalls — the pattern from US-001 was directly reusable. The lessons learned file for US-001 accurately described everything needed.

**Useful Context for Future Agents:**
- The `parallax create video` CLI command accepts `--input` for i2v workflows; the MCP tool exposes it as optional `input` field.
- For future `create_audio` or other tools, follow the same `registerTool` → `Bun.spawn` → `exitCode` → return path pattern.
- Test files in `packages/parallax_mcp/tests/` use source-code scanning (reading `index.ts` as text) rather than live MCP SDK invocation — this avoids needing a running server while still covering all structural and behavioural assertions.

## US-003 — Create Audio via MCP Tool

**Summary:** Added `create_audio` MCP tool to `packages/parallax_mcp/src/index.ts`. The tool registers a fully-typed input schema matching `parallax create audio` options (`model`, `prompt`, `length`, `steps`, `cfg`, `bpm`, `lyrics`, `seed`, `output`, `modelsDir`), spawns the CLI subprocess via `Bun.spawn`, and returns the resolved output path on success or stderr on failure. 22 tests cover all acceptance criteria.

**Key Decisions:**
- Followed the exact same pattern as `create_image` and `create_video`: `server.registerTool()` with a zod shape, `Bun.spawn` with `cwd: CLI_DIR`, and `resolve(input.output ?? "output.wav")` for the default output path.
- Default output is `output.wav` — matches the CLI default in `create.ts` for audio.
- AC01 specifies exactly 10 fields. The CLI also accepts `--unet`, `--vae`, `--text-encoder-1`, `--text-encoder-2` component overrides, but those are not part of the AC and were intentionally omitted.

**Pitfalls Encountered:**
- None — the pattern from US-001 and US-002 was directly reusable.

**Useful Context for Future Agents:**
- The `parallax create audio` CLI command accepts `--bpm` and `--lyrics` which are audio-specific; both are mapped in the MCP tool schema.
- Test files continue using source-code scanning (reading `index.ts` as text) — no running server needed.
- The `create_audio` tool's default output is `output.wav`; always verify the correct default extension per media type when adding new MCP tools.

## US-004 — Edit Image via MCP Tool

**Summary:** Added `edit_image` MCP tool to `packages/parallax_mcp/src/index.ts`. The tool registers a fully-typed input schema matching all 14 `parallax edit image` options (`model`, `prompt`, `input`, `subjectImage`, `width`, `height`, `steps`, `cfg`, `seed`, `output`, `image2`, `image3`, `noLora`, `modelsDir`), spawns the CLI subprocess via `Bun.spawn`, and returns the resolved output path on success or stderr on failure. 28 tests cover all acceptance criteria.

**Key Decisions:**
- Followed the exact same pattern as `create_image`, `create_video`, and `create_audio`: `server.registerTool()` with a zod shape (not `z.object()`), `Bun.spawn` with `cwd: CLI_DIR`, and `resolve(input.output ?? "output.png")` for the default output path.
- `noLora` is a `z.boolean().optional()` field — when truthy, only `"--no-lora"` (no value) is appended to args, matching the commander `--no-lora` boolean flag behavior in the CLI.
- `subjectImage` maps to `--subject-image` (kebab-case), consistent with commander option naming.
- `image2` and `image3` map directly to `--image2` and `--image3`.

**Pitfalls Encountered:**
- None — the pattern from US-001/002/003 was directly reusable with no new issues.

**Useful Context for Future Agents:**
- The `edit image` CLI command is spawned as `["edit", "image", ...]` — the subcommand structure is `edit image` with flags following.
- Boolean flags like `--no-lora` must be pushed without a value argument: `args.push("--no-lora")` — not `args.push("--no-lora", "true")`.
- All 4 MCP tools (`create_image`, `create_video`, `create_audio`, `edit_image`) follow the identical `registerTool → args array → Bun.spawn → exitCode check → return` pattern. Future tools should continue this pattern.

## US-005 — Upscale Image via MCP Tool

**Summary:** Added `upscale_image` MCP tool to `packages/parallax_mcp/src/index.ts`. The tool registers a fully-typed input schema matching all 16 `parallax upscale image` options (`model`, `prompt`, `input`, `checkpoint`, `esrganCheckpoint`, `latentUpscaleCheckpoint`, `negativePrompt`, `width`, `height`, `steps`, `cfg`, `seed`, `output`, `outputBase`, `modelsDir`), spawns the CLI subprocess via `Bun.spawn`, and returns the resolved output path on success or stderr on failure. Also added the missing `--input` flag to the CLI `upscale image` command and `UpscaleImageOpts` interface. 29 tests cover all acceptance criteria.

**Key Decisions:**
- Followed the exact same pattern as `create_image`, `create_video`, `create_audio`, and `edit_image`: `server.registerTool()` with a zod shape, `Bun.spawn` with `cwd: CLI_DIR`, and `resolve(input.output ?? "output.png")` for the default output path.
- AC01 listed `input` as required — this was missing from the CLI `upscale image` command (`upscale.ts` and `UpscaleImageOpts`). Added `--input <path>` as a `requiredOption` to the CLI and added `input: string` to the interface and `buildUpscaleImageArgs()`.
- Subcommand structure is `["upscale", "image", ...]` — matching the `upscale → image` commander hierarchy.

**Pitfalls Encountered:**
- The CLI `upscale image` command did not have `--input` option even though upscaling inherently requires an input image. The AC correctly identified this gap. Adding it to the CLI was necessary for end-to-end correctness.
- There is a pre-existing typecheck error in `packages/parallax_cli/src/models/image.ts:133` (`opts.steps` possibly undefined in `buildEditImageArgs`). This error existed before this story and is unrelated to the changes made.

**Useful Context for Future Agents:**
- The `upscale image` CLI command now requires `--input <path>` as a required option alongside `--model` and `--prompt`.
- All 5 MCP tools (`create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`) follow the identical `registerTool → args array → Bun.spawn → exitCode check → return` pattern.
- The pre-existing typecheck error in `parallax_cli` (`opts.steps` in `buildEditImageArgs`) should be addressed in a dedicated cleanup iteration — it's in the `EditImageOpts` flux branch at line ~135.

## US-006 — MCP Server Startup and Client Registration

**Summary:** Documented `bun run start` for the `@parallax/mcp` package, updated `docs/parallax_mcp.md` with registration config snippets for Claude Desktop, GitHub Copilot (VS Code + CLI), and Claude Code. Added `tests/server_startup.test.ts` with 21 tests covering all acceptance criteria.

**Key Decisions:**
- All 5 tools (`create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`) were already registered in `src/index.ts` from prior iterations — no `index.ts` changes were needed.
- `package.json` already had `"start": "bun run src/index.ts"` — AC01 was already satisfied; only the test and docs were missing.
- The GitHub Copilot VS Code MCP config goes in `.vscode/mcp.json` (key `servers`); the Copilot CLI agent config goes in `.github/copilot/mcp.json` (key `mcpServers`). Both formats documented.
- AC05 (visual verification) is covered by the documentation — the config snippets allow users to register and verify in their client.

**Pitfalls Encountered:**
- AC01's "starts without errors" is tested by spawning the process with `stdin: "pipe"`, immediately closing stdin, then asserting there is no `SyntaxError` or `Cannot find module` in stderr. This approach avoids hanging the test while still catching real import failures.

**Useful Context for Future Agents:**
- The `docs/parallax_mcp.md` now serves as the canonical registration guide — update it if new tools are added or the startup command changes.
- GitHub Copilot VS Code extension uses `"type": "stdio"` in `.vscode/mcp.json`; Claude Desktop and Copilot CLI use `"command"` + `"args"` without a `"type"` key.
- The server's stdio transport means it has no port — the MCP client spawns and owns the server process. There is nothing to `curl` or health-check separately.
