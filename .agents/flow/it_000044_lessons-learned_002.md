# Lessons Learned — Iteration 000044

## US-001 — Submit inference jobs via REST

**Summary:** Implemented 5 Elysia REST endpoints in `packages/parallax_ms/src/index.ts` that accept job submission requests, call `submitJob()` from `@parallax/sdk`, and return `{ job_id, status: "queued" }`. Added tests covering all 5 ACs with both structural (source-reading) and functional (`app.handle()`) test strategies.

**Key Decisions:**
- **Script registry inlined in `parallax_ms`**: Rather than adding `@parallax/cli` as a dependency (which would couple the gateway to the CLI), the script maps were duplicated inline. This keeps `parallax_ms` depending only on `@parallax/sdk` and `elysia`.
- **`app` exported, `.listen()` guarded by `import.meta.main`**: This is the standard pattern for Elysia apps to support testing via `app.handle()` without binding a port.
- **Elysia `t.Object` for body validation + `onError` hook for 400**: Elysia's built-in validation returns 422 by default. An `onError` hook converts `VALIDATION` errors to 400 to satisfy AC07.
- **`mock.module("@parallax/sdk", ...)` in Bun tests**: Bun hoists `mock.module()` calls before static imports, so the mock is active when the app module resolves its SDK dependency. No dynamic imports needed.
- **`getScriptBase()` and `getUvPath()` read env vars**: `PARALLAX_RUNTIME_DIR`, `PARALLAX_REPO_ROOT`, and `PARALLAX_UV_PATH` follow the same convention as `parallax_cli`'s `readConfig()` for portability.

**Pitfalls Encountered:**
- Elysia 1.4.x has pre-existing TypeScript type errors in its own `.d.ts` files (visible in `tsc --noEmit`), unrelated to our code. These are not caused by our changes.
- The `submitJob()` function in `@parallax/sdk` creates a SQLite database for the queue (`~/.config/parallax/jobs.db`) — tests must mock `submitJob` or they'll attempt real file I/O and database operations.

**Useful Context for Future Agents:**
- `packages/parallax_ms/src/index.ts` now exports `app` (for testing) and only calls `.listen(3000)` when run as `import.meta.main`.
- When adding new routes to `parallax_ms`, also update the inline script maps (`IMAGE_CREATE_SCRIPTS`, etc.) to include new models.
- The `parallax_ms` test pattern: structural tests check source text; functional tests use `app.handle(new Request(...))` with `mock.module("@parallax/sdk", ...)` at test file top-level.
- Elysia's `t` (TypeBox) is imported directly from `"elysia"` in version 1.x — no separate `@elysia/typebox` package needed.
