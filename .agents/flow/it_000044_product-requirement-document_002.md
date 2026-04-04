# Requirement: parallax_ms REST API with Job Queue and SSE

## Context
`parallax_ms` currently has only a `/health` endpoint. This PRD defines a full job management REST API using Elysia that allows external consumers (web UIs, scripts, webhooks) to submit inference jobs, check their status, and stream real-time progress via Server-Sent Events (SSE). This package uses the shared job infrastructure from PRD 001 (`@parallax/sdk`).

## Goals
- Expose HTTP endpoints for submitting all five inference operations (create image/video/audio, edit image, upscale image).
- Provide a job status endpoint for polling.
- Provide an SSE endpoint per job for real-time progress streaming without polling.
- Provide job list and cancel endpoints for job management.

## User Stories

### US-001: Submit inference jobs via REST
**As an** external consumer (web UI, script, webhook), **I want** to POST a job request and receive a job ID immediately **so that** I can track a long-running inference without keeping the HTTP connection open.

**Acceptance Criteria:**
- [ ] `POST /jobs/create/image` accepts a JSON body matching `GenerateImageRequest` from `@parallax/sdk` plus a required `model: string` field.
- [ ] `POST /jobs/create/video` accepts a JSON body matching `GenerateVideoRequest` plus `model: string` and optional `input: string` (image path for i2v).
- [ ] `POST /jobs/create/audio` accepts a JSON body matching `GenerateAudioRequest` plus `model: string`.
- [ ] `POST /jobs/edit/image` accepts a JSON body matching `EditImageRequest` plus `model: string`.
- [ ] `POST /jobs/upscale/image` accepts `{ image_path: string, model: string, output?: string }`.
- [ ] All five endpoints call `submitJob()` from `@parallax/sdk/submit` and return `{ job_id: string, status: "queued" }` within 200ms.
- [ ] Returns `400` with a descriptive error body when required fields are missing.
- [ ] Visually verified: `curl -X POST http://localhost:3000/jobs/create/image -d '{"model":"sdxl","prompt":"a cat"}' -H 'Content-Type: application/json'` returns `{ job_id, status: "queued" }`.

### US-002: Get job status by ID
**As a** consumer, **I want** to `GET /jobs/:id` and receive the current state of a job **so that** I can poll status without an open SSE connection.

**Acceptance Criteria:**
- [ ] `GET /jobs/:id` returns a JSON object with: `id`, `status` (`waiting|active|completed|failed`), `progress` (0–100 number), `output` (string or null), `error` (string or null), `createdAt` (epoch ms), `startedAt` (epoch ms or null), `finishedAt` (epoch ms or null).
- [ ] Returns `404` with `{ error: "Job not found" }` when the ID does not exist.
- [ ] `bun typecheck` passes.

### US-003: Stream job progress via SSE
**As a** consumer, **I want** to `GET /jobs/:id/stream` and receive Server-Sent Events until the job completes or fails **so that** I get real-time progress without polling.

**Acceptance Criteria:**
- [ ] `GET /jobs/:id/stream` returns a `text/event-stream` response.
- [ ] While the job is active or waiting, emits `event: progress\ndata: {"pct":N,"step":"..."}` every 500ms.
- [ ] When the job completes, emits `event: completed\ndata: {"output":"<path>"}` then closes the stream.
- [ ] When the job fails, emits `event: failed\ndata: {"error":"<reason>"}` then closes the stream.
- [ ] Returns `404` before opening the stream if the job ID does not exist.
- [ ] `@elysiajs/stream` is added as a dependency.
- [ ] Visually verified in browser: EventSource connects, receives progress events, stream closes on completion.

### US-004: List jobs
**As a** consumer, **I want** to `GET /jobs` and see all recent jobs with their status **so that** I can inspect the queue state.

**Acceptance Criteria:**
- [ ] `GET /jobs` returns `{ jobs: JobSummary[], counts: { waiting, active, completed, failed } }`.
- [ ] `JobSummary` includes: `id`, `status`, `progress`, `model`, `action`, `media`, `createdAt`.
- [ ] Defaults to returning at most 50 most-recent jobs.
- [ ] Supports optional query param `?status=active|completed|failed|waiting` to filter.
- [ ] `bun typecheck` passes.

### US-005: Cancel a job
**As a** consumer, **I want** to `DELETE /jobs/:id` to cancel a running or waiting job **so that** I can abort inference that is no longer needed.

**Acceptance Criteria:**
- [ ] `DELETE /jobs/:id` calls `getQueue().cancel(jobId)` and returns `{ cancelled: true }`.
- [ ] Returns `404` if the job ID does not exist.
- [ ] Returns `409` with `{ error: "Job already completed" }` if the job is already in a terminal state.
- [ ] `bun typecheck` passes.

### US-006: Health endpoint enrichment
**As an** operator, **I want** `GET /health` to include queue statistics **so that** I can verify the queue is operational.

**Acceptance Criteria:**
- [ ] `GET /health` returns `{ status: "ok", queue: { waiting, active, completed, failed } }`.
- [ ] Does not break existing behavior (still returns HTTP 200).
- [ ] `bun typecheck` passes.

## Functional Requirements
- FR-1: All route handlers must use Elysia's native TypeScript input validation (no manual `if (!body.field)` checks).
- FR-2: The server listens on port `3000` by default, overridable via `PORT` environment variable.
- FR-3: CORS must be enabled for all origins in development (`@elysiajs/cors`).
- FR-4: Unhandled errors in route handlers must return `500` with `{ error: "Internal server error" }` — not crash the server.
- FR-5: `parallax_ms/package.json` must declare `@parallax/sdk` as `workspace:*` dependency if not already present.
- FR-6: All job submission endpoints must resolve the correct Python script path using the same `getScript()` / model registry logic already in `parallax_cli/src/models/registry.ts` — do not hardcode script paths.

## Non-Goals (Out of Scope)
- Authentication or API keys.
- Pagination beyond the 50-job default.
- WebSocket support — SSE is sufficient.
- Job result file serving (only the path string is returned).
- Forwarding requests to `server/FastAPI` — the job worker runs pipelines directly.

## Open Questions
- None.
