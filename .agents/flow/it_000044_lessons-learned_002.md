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

## US-002 — Get job status by ID

**Summary:** Implemented `GET /jobs/:id` in `packages/parallax_ms/src/index.ts` that returns a normalized job status object. Added `getJobStatus()` to `packages/parallax_sdk/src/status.ts` and re-exported it from the SDK index. Added 17 tests in `packages/parallax_ms/tests/job_status.test.ts`.

**Key Decisions:**
- **`getJobStatus()` lives in the SDK**: The function uses `getQueue()` (already in SDK), calls `queue.getJob(id)`, then `job.getState()` for current state, and maps bunqueue's `JobStateType` to the 4-value enum (`waiting|active|completed|failed`). States `prioritized`, `delayed`, `waiting-children`, and `unknown` all map to `waiting`.
- **`queue.close()` called after each operation**: Consistent with the `submitJob()` pattern in `submit.ts` — the queue connection is closed after every single operation to avoid holding open file handles.
- **Route placement**: The `GET /jobs/:id` route is placed before all POST routes in the Elysia chain to avoid any accidental wildcard conflicts.
- **Test mock shape**: The test mock for `getJobStatus` returns typed objects matching `ParallaxJobStatus` exactly. `null` is returned for unknown IDs, which triggers the 404 path.

**Pitfalls Encountered:**
- `bun typecheck` in `packages/parallax_ms` (`tsc --noEmit`) reports pre-existing errors from Elysia 1.4.x's own `.d.ts` files — these are not caused by this implementation. The SDK's `typecheck` passes cleanly.
- `job.id` in bunqueue's `Job<T>` interface is typed as `string`, not `string | undefined`, so no null-check is needed.

**Useful Context for Future Agents:**
- `ParallaxJobStatus` type is now exported from `@parallax/sdk` — use it in the CLI and MCP packages when displaying job status.
- `getJobStatus()` returns `null` (not throws) when a job is not found — consumers should handle null with a 404.
- bunqueue's `getJob()` is on the `Queue` instance; `getJobState()` is also available if you only need the state string without other job fields.
- The test file `job_status.test.ts` follows the same mock pattern as `jobs.test.ts`: `mock.module("@parallax/sdk", ...)` at top-level (Bun hoists it before static imports).
