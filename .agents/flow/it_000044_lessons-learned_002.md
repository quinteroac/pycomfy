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

## US-003 — Stream job progress via SSE

**Summary:** Implemented `GET /jobs/:id/stream` in `packages/parallax_ms/src/index.ts` using `@elysiajs/stream`'s `Stream` class. The endpoint returns a `text/event-stream` response that polls `getJobStatus` every 500ms, emitting named SSE events (`progress`, `completed`, `failed`) until the job terminates. Added 6 tests in `packages/parallax_ms/tests/stream.test.ts`.

**Key Decisions:**
- **`Stream` class from `@elysiajs/stream` v1.1.0**: Used the callback form `new Stream(async (s) => {...})`. The `event` setter (`s.event = "progress"`) prepends the SSE `event:` field before each `send()` call, producing correctly-formatted named SSE events without raw byte manipulation.
- **`return new Response(stream.value, { headers: {...} })`**: The `Stream` instance holds its `ReadableStream` in `.value`. Wrapping it in a native `Response` with explicit `Content-Type: text/event-stream` is the most reliable Elysia pattern — it avoids any ambiguity in how Elysia's `mapResponse` handles custom class instances.
- **Route ordering**: `GET /jobs/:id/stream` is defined BEFORE `GET /jobs/:id` to avoid the `/stream` suffix being interpreted as a job ID parameter.
- **`s.close()` must be called explicitly**: In the callback form, `Stream` does NOT auto-close when the callback promise resolves — `close()` must be called manually at the end of each terminal branch.
- **Existence check before stream opens (AC05)**: `getJobStatus` is called once synchronously before creating the `Stream`. If `null`, returns 404 JSON immediately without opening the SSE connection.

**Pitfalls Encountered:**
- **`send()` wraps data in `data:` — named events require `stream.event` setter**: To emit `event: progress\ndata: {...}\n\n`, set `stream.event = "progress"` before calling `stream.send(data)`. The README is minimal; this requires reading the source.
- **`Stream` is marked `@deprecated` in v1.1.0**: The callback form is the only approach that supports named events via the `event` setter without accessing private internals.
- **Test timing**: 500ms polling interval means streaming tests take ≥500ms per terminal event. Set timeouts to 2000–3000ms accordingly.

**Useful Context for Future Agents:**
- `@elysiajs/stream@1.1.0` is now in `packages/parallax_ms/package.json` dependencies.
- `stream.event = "<name>"` BEFORE `stream.send(data)` → named SSE event. Call `stream.close()` explicitly to end the stream.
- `stream.wait(ms)` is a utility on the `Stream` instance — use inside the callback instead of `new Promise(r => setTimeout(r, ms))`.
- Test pattern for streaming: mock `getJobStatus` with call counter, read full body via `resp.text()` (waits for stream closure), assert on accumulated SSE text.
- The `Stream` callback is NOT awaited by the constructor — the async function runs in the background; `close()` signals the controller to end.

## US-004 — List jobs

**Summary:** Implemented `GET /jobs` in `packages/parallax_ms/src/index.ts` with optional `?status=` query param. Added `listJobs()` to `packages/parallax_sdk/src/list.ts`, exported from the SDK index, and added 10 tests in `packages/parallax_ms/tests/list_jobs.test.ts`.

**Key Decisions:**
- **`listJobs()` in SDK**: Follows the same pattern as `getJobStatus()` — uses `getQueue()`, calls async bunqueue methods, closes queue after operations. Uses `queue.getJobsAsync()` with a state array for all-states or single-state queries, plus `queue.getJobCountsAsync()` for counts. Both are called in parallel with `Promise.all`.
- **State resolution via `job.getState()`**: `getJobsAsync()` returns Job objects; their current state is resolved via `await job.getState()` (same approach as `getJobStatus`). The `mapState()` function is duplicated from `status.ts` — they could be extracted to a shared helper in a future refactor.
- **Elysia `t.Union([t.Literal(...)])` for enum query params**: Used `t.Optional(t.Union([t.Literal("waiting"), ...]))` for the `status` query param validation. This pattern works correctly in Elysia 1.4.x.
- **`JobStatusValue` type exported from SDK**: Created a shared `JobStatusValue` type alias (`"waiting" | "active" | "completed" | "failed"`) exported from `list.ts` so `parallax_ms` can import it and avoid re-declaring the union.

**Pitfalls Encountered:**
- **`mock()` return type inference for `jobs: []`**: When declaring `mock(async () => ({ jobs: [], ... }))`, TypeScript infers `jobs` as `never[]`, causing `TS2322` when `mockResolvedValueOnce` tries to assign objects to it. Fix: add explicit return type annotation `Promise<JobListResult>` to the mock factory function.

**Useful Context for Future Agents:**
- `listJobs()` is exported from `@parallax/sdk` and accepts `{ status?: JobStatusValue; limit?: number }`. The default limit of 50 is applied inside the function — callers don't need to pass it.
- `queue.getJobsAsync({ state: ['waiting', 'active', 'completed', 'failed'] })` fetches all jobs across states in a single call; `queue.getJobCountsAsync()` returns `{ waiting, active, completed, failed, prioritized, delayed, paused }` — only the 4 main states are exposed in `JobListResult.counts`.
- `JobSummary.model/action/media` come from `job.data` (typed as `ParallaxJobData`) — jobs submitted through `submitJob()` always have these fields populated.

## US-005 — Cancel a job

**Summary:** Implemented `DELETE /jobs/:id` in `packages/parallax_ms/src/index.ts` that cancels a running or waiting job. Added `cancelJob()` to `packages/parallax_sdk/src/cancel.ts`, exported it from the SDK index, and added 7 tests in `packages/parallax_ms/tests/cancel_job.test.ts`.

**Key Decisions:**
- **`cancelJob()` returns a discriminated value** (`true | null | "terminal"`) rather than throwing: `null` = not found (→ 404), `"terminal"` = already in a completed/failed state (→ 409), `true` = success (→ `{ cancelled: true }`). This keeps the SDK consistent with the null-for-not-found pattern from `getJobStatus`.
- **`queue.removeAsync(id)` for cancellation**: bunqueue's `Queue` class does not expose a `cancel()` method. The user story's "calls `getQueue().cancel(jobId)`" is a conceptual description — the implementation uses `removeAsync(id)` which removes the job regardless of its current non-terminal state.
- **State checked before removal**: `job.getState()` is called before `removeAsync()` to detect terminal states and return `"terminal"` instead of blindly removing.
- **`DELETE` route placed after `GET /jobs/:id`** in the Elysia chain — Elysia matches by method first, so ordering by path specificity is less critical here than for GET routes.

**Pitfalls Encountered:**
- `bun typecheck` in `packages/parallax_ms` still reports pre-existing Elysia 1.4.x type errors unrelated to this implementation. The SDK typechecks cleanly.
- bunqueue has no `cancel()` on the `Queue` class — confirmed by inspecting `packages/parallax_sdk/node_modules/bunqueue/dist/client/queue/queue.d.ts`.

**Useful Context for Future Agents:**
- `cancelJob(id)` is now exported from `@parallax/sdk` — use it in the CLI (`parallax job cancel <id>`) and MCP packages.
- The `CancelJobOutcome` type (`true | null | "terminal"`) is exported from `cancel.ts` and re-exported via the SDK index.
- Functional test pattern: mock `cancelJob` with a function that returns `true` for active/waiting IDs, `"terminal"` for done/failed IDs, and `null` for unknown IDs.
- The `queue.close()` call is required after every queue operation — always include it in all branches (early returns and happy path).
