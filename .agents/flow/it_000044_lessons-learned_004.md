# Lessons Learned — Iteration 000044

## US-001 — Non-blocking inference tools

**Summary:** The MCP server's five inference tools (`create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`) were already implemented non-blocking via `submitJob` from `@parallax/sdk/submit`. The work consisted of fixing 11 failing tests and adding an AC04 timing test with a mock queue.

**Key Decisions:**
- The "script registry" tests in each tool test file were checking `index.ts` for script path strings, but those paths live in `packages/parallax_cli/src/models/registry.ts`. Fixed by adding `const REGISTRY = readFileSync(...)` pointing to `registry.ts` and updating only the path assertions to use `REGISTRY` instead of `SRC`.
- For AC04 (timing test with mock queue), created `non_blocking.test.ts` using `mock.module("@parallax/sdk/submit", ...)` to stub `submitJob` as an instant resolver, then calling the same pattern as each tool handler and asserting `job_id:` presence and sub-500ms completion.
- All other AC01–AC03 assertions (checking `SRC = index.ts` for `submitJob`, `job_id:`, `status: queued`, and description strings) were already passing.

**Pitfalls Encountered:**
- The `create_audio.test.ts` has a describe block named `"US-001 script registry: AUDIO_CREATE_SCRIPTS"` (not `"US-001 script registry and arg mapping"`); the edit was initially applied to the wrong describe block title. Always verify the exact describe name before editing.
- The MCP `index.ts` is a self-executing server script, not an importable module, so the tool handlers (closures inside `server.registerTool()`) cannot be imported and tested directly. The timing test uses the same handler pattern inlined, not the actual handler.

**Useful Context for Future Agents:**
- Script paths for all models live exclusively in `packages/parallax_cli/src/models/registry.ts` — `index.ts` never contains hardcoded paths.
- Tests that check model/script metadata should read `registry.ts`; tests that check tool structure/arg-building should read `index.ts`.
- `mock.module()` in Bun requires the mock to be registered before the module is dynamically imported. Use `beforeAll` + `await import(...)` inside the test body rather than top-level imports.
