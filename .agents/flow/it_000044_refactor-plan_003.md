# Refactor Completion Report — Iteration 000044, Pass 003

## Summary of changes

Three changes were applied across two packages based on the audit findings:

### 1. FR-2 — `watchJobAction` now uses `getQueue()` from `@parallax/sdk/queue`

**File:** `packages/parallax_cli/src/commands/jobs.ts`

The direct `bunqueue/client` `Queue` instantiation (with a hard-coded `~/.config/parallax/jobs.db` path) was replaced with a lazy import of `getQueue()` from `@parallax/sdk/queue`. This restores the SDK abstraction contract: if the db path ever changes in the SDK, `watch` inherits the change automatically with no risk of silent divergence.

### 2. FR-4 — `jobs list` and `jobs watch` now use `@clack/prompts`

**File:** `packages/parallax_cli/src/commands/jobs.ts`

- **`jobs list`**: replaced `console.log(formatJobsTable(...))` with a lazy import of `log` from `@clack/prompts` and `log.message(formatJobsTable(...))`. The table is now rendered inside a structured clack message block, consistent with the `install` and `mcp` commands.
- **`jobs watch`**: replaced manual `process.stdout.write` spinner animation with a `@clack/prompts` `spinner()`. The spinner is started with `.start("Watching job…")`, updated every poll cycle with `.message("${step}… ${progress}%")`, then stopped with `.stop("")` before a final `log.message(formatDoneLine(...))` on success or `log.message(formatFailLine(...))` on failure. Not-found is reported via `log.warn()`.

### 3. Duration column now shows real elapsed time for completed jobs

**File:** `packages/parallax_sdk/src/list.ts`

Added `duration: number | null` to the `JobSummary` interface, computed from `job.finishedOn - job.processedOn` when both timestamps are available. When unavailable (e.g. job still active or waiting), `duration` is `null`.

**File:** `packages/parallax_cli/src/commands/jobs.ts`

Added `formatDuration(durationMs: number | null): string` helper (exported) that renders milliseconds as `Xs`, `Xm`, or `Xm Ys`. `buildRows` now calls `formatDuration(job.duration)` instead of hard-coding `"—"`.

**File:** `packages/parallax_cli/tests/commands/jobs.test.ts`

Updated `makeJob()` factory to include `duration: null` to satisfy the updated `JobSummary` type.

---

## Quality checks

| Check | Scope | Outcome |
|---|---|---|
| `bun run typecheck` (tsc --noEmit) | `packages/parallax_cli` | ✅ Pass |
| `bun run typecheck` (tsc --noEmit) | `packages/parallax_sdk` | ✅ Pass |
| `bun test tests/commands/jobs.test.ts` | `packages/parallax_cli` | ✅ 62/62 pass |

The pre-existing test failures in the full test suite (`bun test` across all files) are unrelated to this iteration — they concern subprocess spawning and `PARALLAX_REPO_ROOT` environment checks in older tests and were failing before these changes.

---

## Deviations from refactor plan

None. All three recommended changes from the audit (`conclusionsAndRecommendations`) were applied:
- `watchJobAction` migrated to `getQueue()` (FR-2 fix).
- `jobs list` and `jobs watch` switched to `@clack/prompts` APIs (FR-4 fix).
- Duration column now computed from real timing data instead of always showing `—`.
