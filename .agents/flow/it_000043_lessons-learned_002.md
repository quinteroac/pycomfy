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
