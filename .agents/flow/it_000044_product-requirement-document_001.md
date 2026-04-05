# Python Job Infrastructure

## Context

The current async job layer is implemented in TypeScript across `@parallax/sdk` (types + bunqueue/SQLite queue) and `packages/parallax_cli/src/_worker.ts` (the subprocess worker). Migrating the full stack to Python requires a Python-native equivalent: Pydantic job types, an `aiosqlite`-backed job queue singleton, a `submit_job()` helper that spawns a detached worker, and a Python worker process that executes pipelines and emits NDJSON progress. All other Python packages (gateway, CLI, MCP) will depend on this layer.

## Goals

- Define Pydantic models for all job contracts (`JobData`, `JobResult`, `PythonProgress`).
- Implement a persistent SQLite-backed job queue in Python using `aiosqlite`.
- Provide a `submit_job()` function that enqueues a job and spawns a detached worker, returning a job ID in < 100ms.
- Implement `server/worker.py` — picks up one job, runs the pipeline subprocess, reads NDJSON progress from stdout, and updates the job record.
- Provide a `ProgressReporter` helper that pipelines can use to emit structured NDJSON progress to stdout.

## User Stories

### US-001: Pydantic job types
**As a** Python developer on any parallax package, **I want** well-typed `JobData`, `JobResult`, and `PythonProgress` Pydantic models in `server/jobs.py` **so that** I can submit and read jobs without dict casting or guessing field names.

**Acceptance Criteria:**
- [ ] `server/jobs.py` exports `JobData`, `JobResult`, and `PythonProgress` as Pydantic `BaseModel` subclasses.
- [ ] `JobData` includes: `action: str`, `media: str`, `model: str`, `script: str`, `args: dict`, `script_base: str`, `uv_path: str`.
- [ ] `JobResult` includes: `output_path: str`.
- [ ] `PythonProgress` includes: `step: str`, `pct: float`, and optional `frame: int | None`, `total: int | None`, `output: str | None`, `error: str | None`.

### US-002: SQLite job queue singleton
**As a** parallax Python package, **I want** to call `get_queue()` and get a shared async job queue backed by `~/.config/parallax/jobs.db` **so that** all packages read and write to the same SQLite job store without owning the queue lifecycle themselves.

**Acceptance Criteria:**
- [ ] `server/queue.py` exports `get_queue() -> JobQueue` where `JobQueue` wraps `aiosqlite`.
- [ ] `get_queue()` is a lazy async singleton — creates the DB and table on first call, returns the same instance on subsequent calls within a process.
- [ ] The jobs table schema includes: `id TEXT PRIMARY KEY`, `status TEXT`, `data TEXT` (JSON), `result TEXT` (JSON, nullable), `created_at TEXT`, `updated_at TEXT`.
- [ ] `JobQueue` exposes: `enqueue(data: JobData) -> str` (returns job id), `get(job_id: str) -> dict | None`, `update_status(job_id, status, result=None)`, `list_jobs(limit=50) -> list[dict]`, `cancel(job_id) -> bool`.

### US-003: submit_job() helper
**As a** gateway or CLI caller, **I want** to call `submit_job(data: JobData) -> str` **so that** I get a job ID back in < 100ms without waiting for inference to finish.

**Acceptance Criteria:**
- [ ] `server/submit.py` exports `submit_job(data: JobData) -> str`.
- [ ] `submit_job()` enqueues the job via `get_queue()` and spawns `server/worker.py` as a detached subprocess (using `subprocess.Popen` with `start_new_session=True`).
- [ ] Returns the job ID string. Total wall-clock time from call to return is < 100ms in a unit test with a mock queue.
- [ ] The spawned worker process receives the job ID as its only CLI argument.

### US-004: Python worker process
**As a** job queue consumer, **I want** `server/worker.py` to pick up a job by ID, execute the pipeline, and stream NDJSON progress back to the queue **so that** any observer can follow progress in real time.

**Acceptance Criteria:**
- [ ] `uv run python server/worker.py <job_id>` runs without error when the job exists and `status == "queued"`.
- [ ] The worker reads `JobData` from the queue, spawns the pipeline subprocess (`uv run python <script> ...`), and streams stdout line-by-line.
- [ ] Each line that is valid `PythonProgress` JSON is parsed and stored as a progress update in the queue (e.g. `update_progress(job_id, progress)`).
- [ ] On pipeline exit code 0: updates job status to `"completed"` with the `output_path` from the last `PythonProgress` line where `output` is set.
- [ ] On non-zero exit: updates status to `"failed"` with the last stderr content in `result.error`.

### US-005: ProgressReporter helper
**As a** pipeline author, **I want** to import `ProgressReporter` and call `reporter.update(step, pct)` **so that** my pipeline emits structured NDJSON progress without writing JSON serialization logic manually.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/progress.py` exports `ProgressReporter` with methods: `update(step: str, pct: float, **kwargs)` and `done(output_path: str)`.
- [ ] Each call to `update()` prints a single line of valid JSON matching `PythonProgress` to stdout.
- [ ] `done(output_path)` prints a final `PythonProgress` with `pct=100.0` and `output=output_path`.
- [ ] Existing pipelines that do not use `ProgressReporter` continue to work unchanged.

---

## Functional Requirements

- FR-1: All job models are Pydantic `BaseModel` with strict type annotations.
- FR-2: The SQLite database path defaults to `~/.config/parallax/jobs.db` and is overridable via `PARALLAX_DB_PATH` environment variable.
- FR-3: `submit_job()` must not block; the worker runs in a fully detached process (`start_new_session=True`).
- FR-4: The worker uses the `uv_path` field from `JobData` to resolve the `uv` binary.
- FR-5: `ProgressReporter` flushes stdout after every write (`flush=True`).
- FR-6: All new modules follow the lazy import pattern — no `torch`/`comfy.*` imports at module top level.

## Non-Goals

- No HTTP endpoints in this PRD — covered by PRD 002.
- No CLI commands in this PRD — covered by PRD 003.
- No MCP tools in this PRD — covered by PRD 004.
- No migration of existing TypeScript packages — they remain functional during the transition.

## Open Questions

- Should `JobQueue` support a `subscribe(job_id)` async generator for real-time progress, or is polling from the gateway sufficient? (SSE streaming pattern decision — impacts PRD 002.)
