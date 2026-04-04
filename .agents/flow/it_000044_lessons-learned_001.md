# Lessons Learned — Iteration 000044

## US-001 — Define shared job types

**Summary:** Created `packages/parallax_sdk/src/jobs.ts` exporting `ParallaxJobData`, `ParallaxJobResult`, and `PythonProgress` interfaces, and re-exported them from `src/index.ts`.

**Key Decisions:** Added a dedicated `jobs.ts` module rather than appending to `types.ts`, keeping request/response HTTP types separate from job queue types. All fields from the ACs were typed with the appropriate primitives; optional fields (`frame`, `total`, `output`, `error` on `PythonProgress`) use the `?` modifier.

**Pitfalls Encountered:** None. The SDK package is straightforward — `bun run typecheck` delegates to `tsc --noEmit` and passes cleanly with zero config changes.

**Useful Context for Future Agents:** `@parallax/sdk` uses `"type": "module"` and points `"main"` directly at the TypeScript source (`./src/index.ts`). There is no build step — Bun resolves `.ts` directly. The typecheck script is `tsc --noEmit` defined in `package.json`. Re-export all new modules from `src/index.ts` to keep the public API surface in one place.

## US-002 — Bunqueue singleton in parallax_sdk

**Summary:** Created `packages/parallax_sdk/src/queue.ts` exporting `getQueue(): Queue` (with a `Bunqueue` type alias) — a lazy singleton backed by `~/.config/parallax/jobs.db` in embedded mode. Added `bunqueue` and `bun-types` as dependencies, and updated `tsconfig.json` to include `"types": ["bun-types"]`.

**Key Decisions:**
- Used `Queue` from `bunqueue/client` (not the all-in-one `Bunqueue` class). The `Bunqueue` class mandates a `processor`, `routes`, or `batch` option at construction time — incompatible with AC04 ("no processor, consumers attach their own"). `Queue` supports `embedded: true` with no processor requirement.
- Exported `Queue as Bunqueue` as a type alias so callers can import `Bunqueue` as a type name from the SDK even though the underlying class is `Queue`.
- Added `bun-types` devDependency and `"types": ["bun-types"]` to `tsconfig.json` because the SDK previously had no Node.js/Bun built-ins typed. Without this, `import os from "os"` and `import path from "path"` produce TS2307 errors even in Bun.

**Pitfalls Encountered:**
- `Bunqueue` class from `bunqueue/client` throws at runtime (`Bunqueue requires "processor", "routes", or "batch"`) when constructed with no handler — the AC assumption was incorrect for that class. Switch to `Queue` which is the producer-only interface.
- `@parallax/sdk` `tsconfig.json` had no `types` array, causing `os`/`path` module not-found errors. Other packages (e.g. `parallax_ms`) already use `"types": ["bun-types"]` — apply the same pattern.

**Useful Context for Future Agents:** `bunqueue/client` exports both `Queue` (producer + job management) and `Worker` (consumer/processor) separately. Prefer `Queue` when building a shared queue handle with no processing logic attached. The `Bunqueue` convenience class is unsuitable when you want to defer processor assignment. `QueueOptions` accepts `embedded: boolean` and `dataPath: string` for SQLite path configuration.


## US-003 — submitJob() helper

**Summary:** Created `packages/parallax_sdk/src/submit.ts` exporting `submitJob(data: ParallaxJobData): Promise<string>`. It enqueues a `"pipeline"` job via `getQueue().add(...)` with `attempts: 1` and `timeout: 30 * 60 * 1000`, spawns a detached Bun worker process, closes the queue, then returns the job ID string.

**Key Decisions:**
- Used `Bun.spawn(...)` (built-in Bun API) for the detached child process — no `child_process` import needed. `detached: true` with all stdio set to `"ignore"` satisfies AC03.
- The job ID is coerced to `String(job.id)` because `bunqueue` returns an ID typed as `number | string`; callers expect `Promise<string>`.
- `queue.close()` is called after spawning (not before) to ensure the job is fully committed to the SQLite DB before the connection is torn down (AC04).
- Re-exported from `src/index.ts` to keep the single-entry-point convention.

**Pitfalls Encountered:** None. `bun run typecheck` passes cleanly with the existing `tsconfig.json` setup (already has `"types": ["bun-types"]` from US-002).

**Useful Context for Future Agents:** `Bun.spawn` is the correct API for detached subprocesses in Bun — it mirrors Node's `child_process.spawn` with `detached: true`. The `detached` option combined with all stdio set to `"ignore"` produces a fully independent process that survives the parent exiting.

## US-004 — Detached worker process (_run.ts)

**Summary:** Created `packages/parallax_cli/src/_run.ts` — a detached Bun worker process that picks up a specific job by ID, runs the Python pipeline via `Bun.spawn` with `stdout: "pipe"` / `stderr: "pipe"`, streams NDJSON progress from stdout into `job.updateProgress(pct)`, and marks the job completed or failed via the `Bunqueue` routes API.

**Key Decisions:**
- Used `Bunqueue` (not `Queue` + `Worker`) because the AC requires a `routes` processor — `Bunqueue` is the only class with a `routes: Record<string, Processor>` option that handles completed/failed lifecycle events via `queue.once(...)`.
- Added `bunqueue` as a direct dependency of `parallax_cli` (not just `parallax_sdk`) — each workspace package must declare its own dependencies in Bun workspaces; transitive dependencies are not automatically resolvable by TypeScript.
- Imported `Job` type from `bunqueue/client` explicitly to avoid the `Parameter 'job' implicitly has an 'any' type` TS error in the route processor function.
- stderr is consumed in a parallel async IIFE so it doesn't block stdout reading; collected text is appended to the thrown `Error` message on non-zero exit.

**Pitfalls Encountered:**
- Pre-existing `TS2345` error in `src/models/image.ts:135`: `opts.steps` (`string | undefined`) passed directly to `args.push()` expecting `string`. Fixed by adding an `!== undefined` guard (consistent with the `qwen` branch above it and AC09 requirement for clean typecheck).
- `bun typecheck` delegates to `tsc --noEmit` which is strict — explicit type annotations on route processors are required.

**Useful Context for Future Agents:** `Bunqueue` closes are async — call `await queue.close()` inside the `queue.once("completed", ...)` and `queue.once("failed", ...)` callbacks. The `Bunqueue` class in `bunqueue/client` exposes `on` / `once` for lifecycle events (`completed`, `failed`, `closed`, `drained`, etc.) directly on the `Bunqueue` instance (not on an internal `.worker` property). The worker auto-starts polling on construction — no explicit `.start()` call needed.
