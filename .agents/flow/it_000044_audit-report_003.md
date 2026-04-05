# Audit Report — Iteration 000044, PRD 003

## Executive Summary

PRD 003 introduces the `--async` flag on all generation commands and a `parallax jobs` subcommand with `list`, `watch`, `status`, `cancel`, and `open` sub-commands. The implementation is broadly compliant: all six user stories are functionally satisfied, the `--async` flag is present on all five target commands, and `bun typecheck` passes.

Two compliance gaps exist:
- **FR-4**: `jobs list` and `jobs watch` use `console.log`/`process.stdout.write` directly instead of the `@clack/prompts` spinner/log API.
- **FR-2**: `watchJobAction` bypasses `getQueue()` from `@parallax/sdk` and directly instantiates `Queue` from `bunqueue/client`.

Both gaps are architectural — the functional behaviour is correct but deviates from the conventions set in the PRD.

---

## Verification by FR

| FR   | Assessment        | Notes |
|------|-------------------|-------|
| FR-1 | ✅ comply         | All sub-commands in `jobs.ts`; `registerJobs` wired in `index.ts` under `jobs` parent. |
| FR-2 | ⚠️ partially_comply | SDK sub-commands call `getQueue()` + `close()` correctly. `watchJobAction` bypasses `getQueue()` and directly instantiates `Queue` from `bunqueue/client`. Queue is still closed; no handle leak, but SDK abstraction is violated. |
| FR-3 | ✅ comply         | `--async` registered as `.option(...)` on all five target commands — visible in `--help`. |
| FR-4 | ❌ does_not_comply | `jobs list` uses `console.log`; `jobs watch` uses `process.stdout.write` + `console.log`. Neither uses `@clack/prompts`. |

---

## Verification by US

| US     | Assessment | Notes |
|--------|------------|-------|
| US-001 | ✅ comply  | `--async` on all 5 commands; routes to `runAsync()` → `submitJob()`; output format matches spec; fallback to `spawnPipeline()` intact; typecheck passes. |
| US-002 | ✅ comply  | All 7 table columns present; ANSI color-coding per status; limit 20; correct empty message; typecheck passes. |
| US-003 | ✅ comply  | 500 ms poll; spinner with step+%; correct done/failed/not-found messages and exit codes; typecheck passes. |
| US-004 | ✅ comply  | Full status block; `--json` outputs valid JSON; correct exit codes; typecheck passes. |
| US-005 | ✅ comply  | Cancel prints `✔ Job <id> cancelled`; not-found exits 1; already-terminal message exits 0; typecheck passes. |
| US-006 | ✅ comply  | `xdg-open`/`open` per platform; not-completed/not-found messages + exit codes correct; typecheck passes. |

---

## Minor Observations

1. `watchJobAction` hard-codes the db path `~/.config/parallax/jobs.db` from `bunqueue/client` directly. If `@parallax/sdk/queue` changes the path, `watch` will silently diverge.
2. `jobs list` and `jobs watch` use raw terminal APIs; inconsistent with `install`/`mcp` commands that use `@clack/prompts`.
3. `cancelJob` uses `queue.removeAsync()` — the PRD references `getQueue().cancel()`, but `bunqueue` does not expose `.cancel()`. Naming discrepancy only; behaviour is correct.
4. Duration column in `jobs list` always shows `—`. Could be computed from `finishedOn - processedOn` for completed jobs.
5. Tests cover pure helper functions only; action-handler integration paths are not covered.

---

## Conclusions and Recommendations

The prototype is functionally complete and all user stories comply. Two targeted refactor items are recommended:

1. **Fix FR-4** — Replace `console.log`/`process.stdout.write` in `jobs list` and `jobs watch` with `@clack/prompts` APIs (`spinner`, `log.success`, `log.error`) to maintain UX consistency with the rest of the CLI.
2. **Fix FR-2** — Refactor `watchJobAction` to use `getQueue()` from `@parallax/sdk/queue` instead of directly instantiating `Queue` from `bunqueue/client`, to avoid db-path divergence risk.
3. **Nice-to-have** — Compute `duration` from `job.finishedOn - job.processedOn` in `buildRows()`.

---

## Refactor Plan

### Task 1 — Fix FR-4: use `@clack/prompts` in `jobs list` and `jobs watch`

**File:** `packages/parallax_cli/src/commands/jobs.ts`

- In the `list` action handler: replace `console.log(formatJobsTable(...))` with `@clack/prompts` `log.message()` or `note()`.
- In `watchJobAction`: replace the manual spinner loop (`process.stdout.write("\r" + ...)`) with `@clack/prompts` `spinner()` API — call `spin.start()`, `spin.message()`, `spin.stop()` / `spin.cancel()`.

### Task 2 — Fix FR-2: use `getQueue()` in `watchJobAction`

**File:** `packages/parallax_cli/src/commands/jobs.ts`

- Remove the inline `Queue` import from `bunqueue/client` and the manual db-path construction.
- Import `{ getQueue }` from `@parallax/sdk/queue` (or rely on existing `@parallax/sdk` re-exports).
- Replace `new Queue(...)` with `getQueue()` in `watchJobAction`.

### Task 3 (nice-to-have) — Compute Duration in `buildRows`

**File:** `packages/parallax_cli/src/commands/jobs.ts`

- Extend `JobSummary` (or use a local computed field) to carry `finishedAt` / `processedOn`.
- Compute duration string (e.g. `"12s"`, `"3m 4s"`) and display it in the Duration column for completed/failed jobs.
