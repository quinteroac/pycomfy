# Lessons Learned — Iteration 000044

## US-001 — --async flag on generation commands

**Summary:** Added `--async` flag to `parallax create image/video/audio`, `parallax edit image`, and `parallax upscale image`. When `--async` is provided the CLI builds a `ParallaxJobData` payload, calls `submitJob()` from `@parallax/sdk/submit`, prints `Job <jobId> queued\n  → parallax jobs watch <jobId>`, and exits 0. Without `--async`, behavior is unchanged.

**Key Decisions:**
- Added two pure helper functions to `runner.ts` — `buildJobData()` and `formatAsyncMessage()` — to keep the async path testable without spawning processes or calling `process.exit`. This mirrors the existing pattern in the CLI where all testable logic is extracted into pure functions (model builders, registry lookups).
- Added subpath `exports` to `packages/parallax_sdk/package.json` so that `import { submitJob } from "@parallax/sdk/submit"` resolves correctly. The SDK previously had no `exports` field, only `"main"`, which made subpath imports fail TypeScript resolution.
- `runAsync()` in `runner.ts` shares the same config-resolution logic (`runtimeDir ?? repoRoot`, `uvPath`) as `spawnPipeline()`, keeping the error message consistent.

**Pitfalls Encountered:**
- `@parallax/sdk/submit` failed TypeScript resolution because the SDK's `package.json` had no `exports` field — only `"main"`. Had to add full `exports` map covering all subpaths to fix `bun typecheck`.
- `opts.async` in Commander action handlers: `async` is a reserved keyword in statement context but valid as a property key in TypeScript. Commander stores the flag under exactly that key via camelCase conversion of `--async`, so `opts.async` works correctly.

**Useful Context for Future Agents:**
- The existing test suite has several pre-existing failures in `tests/index.test.ts` where tests assert `"Error: PARALLAX_REPO_ROOT is required"` but the actual error from `runner.ts` is the longer message `"Error: no script directory configured — run \`parallax install\` to set runtimeDir, or set PARALLAX_REPO_ROOT"`. Do not fix these unless explicitly tasked — they are integration tests for a different US.
- The CLI testing pattern is: test pure helper functions directly (no mocking, no spawning). Action handlers that call `process.exit` are validated via `bun typecheck` and integration tests in `tests/index.test.ts`. Follow this pattern for any future story touching command handlers.
- `@parallax/sdk` now has subpath exports in `package.json`. Adding a new file to the SDK requires adding a corresponding entry in the `exports` map.
