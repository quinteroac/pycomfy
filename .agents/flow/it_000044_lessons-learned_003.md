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

## US-002 — parallax jobs list

**Summary:** Added `parallax jobs list` command that fetches up to 20 most-recent jobs from the bunqueue queue via `listJobs({ limit: 20 })` from `@parallax/sdk/list`, then renders a formatted table with columns ID, Status, Action, Model, Progress, Started, Duration. Status is color-coded with ANSI escape codes. When no jobs exist, a guidance message is printed.

**Key Decisions:**
- Used raw ANSI escape codes (`\x1b[2m`, `\x1b[36m`, `\x1b[32m`, `\x1b[31m`) directly instead of pulling in `picocolors` as an explicit dep. `picocolors` is only a transitive dep of `@clack/prompts`; importing it directly would add an implicit dependency not in `package.json`.
- Extracted all pure helpers (`colorStatus`, `formatStarted`, `buildRows`, `formatJobsTable`, `EMPTY_MESSAGE`) so they can be unit-tested without spawning processes or touching the queue.
- `visibleLength()` strips ANSI codes before measuring string width for column alignment — critical for correct padding when status cells contain escape sequences.
- `formatJobsTable` is the main tested function: it takes `JobSummary[]` and returns a string, making it trivially testable.
- The `Duration` column shows `—` for all jobs because `JobSummary` has no `endedAt`/`completedAt` field. This is intentional and correct given the current SDK data model.

**Pitfalls Encountered:**
- None significant. The SDK already exports `listJobs` with `limit` support from `@parallax/sdk/list` (already in the `exports` map), so no SDK changes were needed.

**Useful Context for Future Agents:**
- `JobSummary` (from `@parallax/sdk/list`) has: `id`, `status`, `progress`, `model`, `action`, `media`, `createdAt`. No end-time field exists yet.
- `listJobs({ limit: 20 })` already sorts newest-first and enforces the limit — the CLI does not need to re-sort or re-slice.
- All ANSI color codes used: dim=`\x1b[2m`, cyan=`\x1b[36m`, green=`\x1b[32m`, red=`\x1b[31m`, reset=`\x1b[0m`.
- The `tsconfig.json` for the CLI only includes `src/`, not `tests/` — test files are type-checked by bun test itself, not `tsc --noEmit`. This means typecheck in tests only catches obvious errors; full type safety is in `src/`.

## US-003 — parallax jobs watch \<id\>

**Summary:** Added `parallax jobs watch <id>` subcommand to the `jobs` group in `src/commands/jobs.ts`. It polls `getJob(id)` every 500ms, renders a Braille spinner with the job's action as the step label and numeric progress, then prints `✔ Done: <outputPath>` or `✖ Failed: <reason>` on terminal state. Non-existent job IDs print `Job <id> not found` and exit 1.

**Key Decisions:**
- Created a fresh `Queue` instance directly via `import("bunqueue/client")` inside `watchJobAction` (lazy import) rather than using the `getQueue()` singleton from `@parallax/sdk/queue`. This avoids the close-after-use singleton problem: other SDK functions (`listJobs`, `getJobStatus`, `cancelJob`) all call `queue.close()` after every operation, which leaves the singleton in a closed state for subsequent calls. For polling, a dedicated open-during-lifetime queue instance is essential.
- `job.returnvalue` is typed as `ParallaxJobResult | null` (cast) — bunqueue stores it as the deserialized worker return value `{ outputPath }`. The existing `status.ts` uses `String(job.returnvalue)` which yields `[object Object]` — a pre-existing bug. The watch command casts correctly to `ParallaxJobResult`.
- The step label shown in the spinner comes from `job.data.action` (e.g. `"create"`, `"edit"`), since bunqueue does not store the `PythonProgress.step` string separately. Changing `_run.ts` to call `updateProgress({ step, pct })` would require updating `status.ts` and was out of scope.
- Pure helpers (`formatSpinnerLine`, `formatDoneLine`, `formatFailLine`, `formatNotFoundMessage`, `SPINNER_FRAMES`) are exported for testability, following the established CLI testing pattern.

**Pitfalls Encountered:**
- The `getQueue()` singleton close-after-use pattern is incompatible with polling. Using `getJobStatus()` in a loop would silently fail on the second poll because the queue is already closed and the singleton is not reset. Always create a dedicated Queue instance for watch-style long-lived consumers.
- `job.failedReason` is not on the bunqueue `Job` type definition; required `(job as any).failedReason` cast to access it.

**Useful Context for Future Agents:**
- `@parallax/cli` already depends on `bunqueue` directly (in `package.json` dependencies), so importing `Queue` from `bunqueue/client` inside CLI code is safe and does not add a new dependency.
- The `getQueue()` singleton in `@parallax/sdk/queue.ts` is NOT reset after `close()`. If you call `close()` on it and then call `getQueue()` again, you get the same closed instance. For any polling or long-lived queue consumer, bypass the singleton and construct `new Queue(...)` directly with the same `dbPath` configuration.
- `ParallaxJobResult` from `@parallax/sdk/jobs` is `{ outputPath: string }`. Access `job.returnvalue as ParallaxJobResult` after a completed state check.

## US-004 — parallax jobs status \<id\>

**Summary:** Added `parallax jobs status <id>` command to the `jobs` group in `src/commands/jobs.ts`. Prints a structured block with `id`, `status`, `progress`, `model`, `action`, `output` (or `error`), `startedAt`, `finishedAt`. Supports `--json` flag for machine-readable output. Exits 1 for failed jobs or missing job IDs.

**Key Decisions:**
- Extended `ParallaxJobStatus` type in `@parallax/sdk/status.ts` to add `model` and `action` fields (sourced from `job.data as ParallaxJobData`). Also fixed `output` to use `job.returnvalue?.outputPath` (via cast) instead of `String(job.returnvalue)` which produced `[object Object]` — pre-existing bug.
- Removed `createdAt` from `ParallaxJobStatus` (was unused in the one-shot status display; `startedAt` and `finishedAt` are the relevant timestamps per the AC).
- `statusJobAction` uses `getJobStatus()` from `@parallax/sdk/status` directly (one-shot, no polling), so the singleton close-after-use pattern is fine here — no need to create a dedicated Queue instance.
- Pure helpers `formatJobStatus` and `formatJobStatusJson` exported for testability following established CLI convention.
- `formatJobStatusJson` omits the `output` key entirely when `error` is non-null (uses `undefined`), ensuring clean JSON with no `null` noise.

**Pitfalls Encountered:**
- Removing `createdAt` from `ParallaxJobStatus` is a breaking change to the type. Any consumers that relied on `status.createdAt` will get a TypeScript error. Check `parallax_ms` if it uses `getJobStatus` directly.

**Useful Context for Future Agents:**
- The `getJobStatus()` singleton close-after-use pattern IS safe for one-shot calls (the queue is fully usable until `.close()` is called). Only polling/long-lived consumers need a dedicated Queue instance (as established in US-003).
- `ParallaxJobStatus` no longer has `createdAt` — use `startedAt` (maps to `job.processedOn`) or `finishedAt` (maps to `job.finishedOn`).
- `formatJobStatusJson` uses `undefined` for absent/mutually-exclusive fields — these are omitted from `JSON.stringify` output automatically.

## US-005 — parallax jobs cancel \<id\>

**Summary:** Added `parallax jobs cancel <id>` command to the `jobs` group in `src/commands/jobs.ts`. Calls `cancelJob()` from `@parallax/sdk/cancel`, handles not-found (exit 1), already-terminal (exit 0 with status in message), and success cases.

**Key Decisions:**
- Updated `CancelJobOutcome` in `@parallax/sdk/cancel.ts` to return `"completed" | "failed"` instead of the opaque `"terminal"` — needed so the CLI can print `Job <id> is already <status>` with the actual status value.
- Updated `parallax_ms/src/index.ts` DELETE handler from `result === "terminal"` to `result !== true` — this is backward-compatible with both old `"terminal"` mock values in ms tests and the new `"completed"/"failed"` return values.
- `cancelJob` import added as a direct top-level import in `jobs.ts` (consistent with `listJobs` and `getJobStatus` pattern; not lazy-imported).
- Pure helpers `formatCancelledMessage` and `formatAlreadyTerminalMessage` are exported for testability, following the established CLI testing pattern.

**Pitfalls Encountered:**
- `parallax_ms/tests/cancel_job.test.ts` was already failing before this story due to a missing `getQueueStats` export in `@parallax/sdk/index.ts` — pre-existing issue, not introduced by this story.

**Useful Context for Future Agents:**
- `CancelJobOutcome` is now `true | null | "completed" | "failed"` (no `"terminal"` anymore). Any consumer that checks `=== "terminal"` needs updating; the ms handler was updated to use `!== true`.
- The CLI cancel command correctly exits with code 1 for not-found and code 0 for already-terminal (per AC02/AC03). The `process.exit(1)` is only called in the not-found path.

## US-006 — parallax jobs open <id>

**Summary:** Added `parallax jobs open <id>` command to the `jobs` group. It fetches the job via `getJobStatus()`, guards for not-found (exit 1) and non-completed status (exit 1 with status in message), then calls `Bun.spawn(["xdg-open", outputPath])` on Linux or `Bun.spawn(["open", outputPath])` on macOS via `process.platform === "darwin"` check.

**Key Decisions:**
- Reused `getJobStatus()` (one-shot singleton pattern) — safe because no polling is needed.
- Reused `formatNotFoundMessage()` from US-003/005 — consistent message format across all job subcommands.
- Added `formatNotCompletedMessage(id, status)` as a pure exported helper for testability.
- Platform detection uses `process.platform === "darwin"` to pick `"open"` vs `"xdg-open"`.

**Pitfalls Encountered:**
- None. The pattern was well-established by US-003/004/005; this story was a straightforward extension.

**Useful Context for Future Agents:**
- `job.output` from `ParallaxJobStatus` holds the `outputPath` string (or null). Always guard for null before passing to `Bun.spawn`.
- `Bun.spawn` is a global in the Bun runtime — no import needed.
- The `openJobAction` uses `getJobStatus()` (one-shot, singleton close-after-use) which is correct for non-polling use cases.
