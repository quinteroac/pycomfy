# Requirement: Shared Job Infrastructure (parallax_sdk)

## Context
All three consumers (parallax_cli, parallax_ms, parallax_mcp) need to submit and observe long-running inference jobs without blocking. This PRD defines the shared foundation: job types, a bunqueue-backed queue singleton, a submit helper, and the detached worker process that actually runs the Python pipeline. Everything else (REST API, CLI commands, MCP tools) depends on this layer.

## Goals
- Introduce a persistent, file-based job queue (bunqueue + SQLite) in `parallax_sdk` accessible by all packages.
- Provide a `submitJob()` function that enqueues a job and spawns a detached worker process, returning a job ID in < 100ms.
- Implement the worker process (`packages/parallax_cli/src/_run.ts`) that picks up one job, runs the Python pipeline, reads NDJSON progress from stdout, and updates the job record.
- Add a Python `ProgressReporter` helper so existing pipelines can emit structured progress without rewriting their core logic.

## User Stories

### US-001: Define shared job types
**As a** TypeScript developer on any parallax package, **I want** well-typed `JobData` and `JobResult` interfaces in `@parallax/sdk` **so that** I can submit and read jobs without casting or guessing shapes.

**Acceptance Criteria:**
- [ ] `packages/parallax_sdk/src/jobs.ts` exports `ParallaxJobData`, `ParallaxJobResult`, and `PythonProgress` interfaces matching the shapes defined in the design doc.
- [ ] `ParallaxJobData` includes: `action`, `media`, `model`, `script`, `args`, `scriptBase`, `uvPath`.
- [ ] `ParallaxJobResult` includes: `outputPath: string`.
- [ ] `PythonProgress` includes: `step: string`, `pct: number`, and optional `frame`, `total`, `output`, `error`.
- [ ] Types are re-exported from `packages/parallax_sdk/src/index.ts`.
- [ ] `bun typecheck` passes with no errors in `parallax_sdk`.

### US-002: Bunqueue singleton in parallax_sdk
**As a** parallax package, **I want** to call `getQueue()` and get a shared bunqueue instance backed by `~/.config/parallax/jobs.db` **so that** all packages read and write to the same SQLite job store without owning the queue lifecycle themselves.

**Acceptance Criteria:**
- [ ] `packages/parallax_sdk/src/queue.ts` exports `getQueue(): Bunqueue`.
- [ ] `getQueue()` is a lazy singleton — creates the instance on first call, returns the same instance on subsequent calls within a process.
- [ ] The SQLite database path resolves to `~/.config/parallax/jobs.db` (uses `os.homedir()`).
- [ ] Queue is created in embedded mode (`embedded: true`) with no processor — consumers attach their own processors.
- [ ] `bunqueue` is added as a dependency to `packages/parallax_sdk/package.json`.
- [ ] `bun typecheck` passes.

### US-003: submitJob() helper
**As a** CLI command or API handler, **I want** to call `submitJob(data)` and receive a job ID immediately **so that** I can return the ID to the user without waiting for inference to finish.

**Acceptance Criteria:**
- [ ] `packages/parallax_sdk/src/submit.ts` exports `submitJob(data: ParallaxJobData): Promise<string>` returning the job ID.
- [ ] `submitJob` enqueues a job named `"pipeline"` via `getQueue().add(...)` with `attempts: 1` and `timeout: 30 * 60 * 1000` (30 min).
- [ ] After enqueuing, `submitJob` spawns a detached Bun process: `bun packages/parallax_cli/src/_run.ts <jobId>` with `stdin: "ignore"`, `stdout: "ignore"`, `stderr: "ignore"`, `detached: true`.
- [ ] `submitJob` closes the queue connection before returning.
- [ ] The full call (enqueue + spawn) completes in under 200ms on a cold start.
- [ ] `bun typecheck` passes.

### US-004: Detached worker process (_run.ts)
**As a** submitted job, **I want** a dedicated worker process to pick me up, run the Python pipeline, stream NDJSON progress into the job record, and mark me completed or failed **so that** the submitting process can exit without losing my execution.

**Acceptance Criteria:**
- [ ] `packages/parallax_cli/src/_run.ts` reads `process.argv[2]` as the job ID.
- [ ] Creates a bunqueue instance with a `"pipeline"` route processor.
- [ ] Spawns the Python pipeline with `Bun.spawn([uvPath, "run", "python", join(scriptBase, script), ...args], { stdout: "pipe", stderr: "pipe" })`.
- [ ] Reads stdout line by line; for each valid JSON line matching `PythonProgress`, calls `job.updateProgress(event.pct)`.
- [ ] On successful exit, returns `{ outputPath }` as the job result.
- [ ] On non-zero exit code, throws an `Error` with the message `Pipeline failed (exit <code>)`.
- [ ] Closes the queue once the job is `completed` or `failed`.
- [ ] Subprocess stderr is captured and included in the failure message when the pipeline exits non-zero.
- [ ] `bun typecheck` passes.

### US-005: Python ProgressReporter helper
**As a** Python pipeline author, **I want** to call `progress(step, pct, **kwargs)` **so that** the worker process receives structured progress without me rewriting the pipeline's inference logic.

**Acceptance Criteria:**
- [ ] `packages/parallax_cli/runtime/progress.py` exports a `progress(step: str, pct: float, **kwargs) -> None` function.
- [ ] The function prints `json.dumps({"step": step, "pct": pct, **kwargs})` followed by a newline to stdout, with `flush=True`.
- [ ] At minimum, the following pipeline files are updated to call `progress()` at logical milestones (download, model load, sampling start, sampling per-frame or per-step, encode, done): `runtime/video/ltx/ltx2/t2v.py` and `runtime/video/wan/wan22/t2v.py`.
- [ ] Updated pipelines still print the final output path as the last line (unchanged behavior for the sync path).
- [ ] No changes to pipeline core logic (model loading, sampling, saving).

## Functional Requirements
- FR-1: The SQLite database file is created automatically on first use if it does not exist.
- FR-2: The detached worker process must not inherit the parent's stdio file descriptors.
- FR-3: Non-JSON lines emitted by the Python pipeline to stdout must be silently ignored by the worker (no crash).
- FR-4: All new TypeScript modules in `parallax_sdk` must be re-exported from `src/index.ts`.
- FR-5: The `bunqueue` package version must be pinned to a specific minor version in `package.json` (no `^` or `~` floating).

## Non-Goals (Out of Scope)
- Retry logic, backoff, or DLQ configuration — `attempts: 1`, fail fast.
- Job priority or delay scheduling.
- Daemon process or persistent worker pool.
- Any UI or REST endpoint — that is PRD 002.
- Changes to the synchronous `spawnPipeline()` in `runner.ts` — backward compatibility must be preserved.

## Open Questions
- None.
