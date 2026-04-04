# Requirement: parallax_cli Async Mode and Jobs Subcommand

## Context
The CLI currently blocks for the entire duration of inference (up to 5+ minutes for video). This PRD adds an `--async` flag to all generation commands so they return immediately with a job ID, and introduces the `parallax jobs` subcommand for listing, watching, cancelling, and opening job results. The synchronous path (`spawnPipeline`) is preserved unchanged for backward compatibility.

## Goals
- Add `--async` flag to `create image`, `create video`, `create audio`, `edit image`, and `upscale image` commands.
- Add `parallax jobs` subcommand with `list`, `watch`, `status`, `cancel`, and `open` sub-commands.
- Show a real-time progress bar in `parallax jobs watch` using the existing `@clack/prompts` dependency.

## User Stories

### US-001: --async flag on generation commands
**As a** developer, **I want** to append `--async` to any generation command **so that** the CLI returns a job ID immediately and I can continue working without waiting for inference to finish.

**Acceptance Criteria:**
- [ ] The following commands accept an `--async` flag: `parallax create image`, `parallax create video`, `parallax create audio`, `parallax edit image`, `parallax upscale image`.
- [ ] When `--async` is provided, the command calls `submitJob()` from `@parallax/sdk/submit` instead of `spawnPipeline()`.
- [ ] The command prints exactly: `Job <jobId> queued\n  → parallax jobs watch <jobId>` and exits with code 0.
- [ ] When `--async` is NOT provided, behavior is identical to current (calls `spawnPipeline()`).
- [ ] `bun typecheck` passes.
- [ ] Manually verified: `parallax create video --model wan22 --prompt "a sunset" --async` returns within 1 second.

### US-002: parallax jobs list
**As a** developer, **I want** to run `parallax jobs list` and see a table of recent jobs **so that** I can see what is running, queued, and done.

**Acceptance Criteria:**
- [ ] `parallax jobs list` prints a table with columns: `ID`, `Status`, `Action`, `Model`, `Progress`, `Started`, `Duration`.
- [ ] Status is color-coded: `waiting` = dim, `active` = cyan, `completed` = green, `failed` = red (using `@clack/prompts` or `kleur`).
- [ ] Shows at most 20 most-recent jobs, newest first.
- [ ] When no jobs exist, prints: `No jobs found. Run a command with --async to submit one.`
- [ ] `bun typecheck` passes.
- [ ] Visually verified in terminal.

### US-003: parallax jobs watch \<id\>
**As a** developer, **I want** to run `parallax jobs watch <id>` and see a live progress bar until the job finishes **so that** I can monitor a previously submitted job.

**Acceptance Criteria:**
- [ ] `parallax jobs watch <id>` polls `getQueue().getJob(id)` every 500ms.
- [ ] Displays a spinner with the current step and percentage (e.g. `sampling… 45%`).
- [ ] When the job completes, stops the spinner and prints: `✔ Done: <outputPath>`.
- [ ] When the job fails, stops the spinner and prints: `✖ Failed: <failedReason>` and exits with code 1.
- [ ] When the job ID does not exist, prints: `Job <id> not found` and exits with code 1.
- [ ] `bun typecheck` passes.
- [ ] Visually verified: spinner updates, success/failure message renders correctly.

### US-004: parallax jobs status \<id\>
**As a** developer or script, **I want** to run `parallax jobs status <id>` and get a one-shot status print **so that** I can check a job without keeping a watch loop open.

**Acceptance Criteria:**
- [ ] `parallax jobs status <id>` prints a single structured block: `id`, `status`, `progress`, `model`, `action`, `output` (or `error`), `startedAt`, `finishedAt`.
- [ ] When `--json` flag is provided, output is valid JSON (one object, no extra text).
- [ ] Returns exit code 0 if completed, 1 if failed, 0 otherwise.
- [ ] `bun typecheck` passes.

### US-005: parallax jobs cancel \<id\>
**As a** developer, **I want** to run `parallax jobs cancel <id>` to stop a running job **so that** I can free GPU resources if inference is no longer needed.

**Acceptance Criteria:**
- [ ] `parallax jobs cancel <id>` calls `getQueue().cancel(jobId)` and prints: `✔ Job <id> cancelled`.
- [ ] When the job does not exist, prints: `Job <id> not found` and exits with code 1.
- [ ] When the job is already completed or failed, prints: `Job <id> is already <status> — nothing to cancel` and exits with code 0.
- [ ] `bun typecheck` passes.

### US-006: parallax jobs open \<id\>
**As a** developer, **I want** to run `parallax jobs open <id>` to open the output file in the system default application **so that** I can preview the result without navigating to the output directory.

**Acceptance Criteria:**
- [ ] `parallax jobs open <id>` reads the job result's `outputPath` and calls `Bun.spawn(["xdg-open", outputPath])` on Linux or `["open", outputPath]` on macOS.
- [ ] When the job is not yet completed, prints: `Job <id> is not completed yet (status: <status>)` and exits with code 1.
- [ ] When the job does not exist, prints: `Job <id> not found` and exits with code 1.
- [ ] `bun typecheck` passes.

## Functional Requirements
- FR-1: All `parallax jobs` sub-commands must be implemented in `packages/parallax_cli/src/commands/jobs.ts` and registered in `src/index.ts` under a `jobs` parent command.
- FR-2: `getQueue()` from `@parallax/sdk` must be called and `.close()` must be called before process exit to avoid hanging SQLite handles.
- FR-3: The `--async` flag must be a Commander `Option` so it appears in `--help` output.
- FR-4: `parallax jobs list` and `parallax jobs watch` must use the spinner/log API from `@clack/prompts` — not `console.log`.

## Non-Goals (Out of Scope)
- Making `--async` the default behavior — opt-in only.
- Pagination in `parallax jobs list`.
- `parallax jobs clean` (purge old jobs) — deferred.
- Integration with `parallax_ms` REST API — the CLI reads directly from SQLite.

## Open Questions
- None.
