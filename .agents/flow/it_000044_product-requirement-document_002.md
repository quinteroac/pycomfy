# FastAPI Gateway (Python)

## Context

`parallax_ms` is currently an Elysia TypeScript gateway running on `:3000` that proxies to the FastAPI `server/` on `:5000`. With the job infrastructure moved to Python (PRD 001), the gateway can be consolidated into `server/` as a FastAPI sub-application — eliminating the separate Node process and the TS→Python boundary. This PRD replaces `parallax_ms` with a FastAPI router mounted inside the existing FastAPI app, exposing the same REST + SSE surface.

## Goals

- Expose REST endpoints for submitting all five inference operations (create image/video/audio, edit image, upscale image).
- Provide a job status endpoint for polling.
- Provide a Server-Sent Events (SSE) endpoint per job for real-time progress streaming.
- Provide job list and cancel endpoints.
- Mount the gateway router on the existing FastAPI app in `server/main.py`.

## User Stories

### US-001: Submit inference jobs via REST
**As an** external consumer (web UI, script, webhook), **I want** to POST a job request and receive a job ID immediately **so that** I can track a long-running inference without keeping the HTTP connection open.

**Acceptance Criteria:**
- [ ] `POST /jobs/create/image` accepts a JSON body with at minimum: `model: str`, `prompt: str`, and optional generation fields; returns `{"job_id": "<id>", "status": "queued"}` within 200ms.
- [ ] `POST /jobs/create/video` accepts a JSON body with `model: str`, `prompt: str`, and optional `input: str` (image path for i2v).
- [ ] `POST /jobs/create/audio` accepts a JSON body with `model: str`, `prompt: str`, and optional audio fields.
- [ ] `POST /jobs/edit/image` accepts a JSON body with `model: str`, `prompt: str`, `input: str` (input image path).
- [ ] `POST /jobs/upscale/image` accepts a JSON body with `model: str`, `prompt: str`, `input: str`.
- [ ] All five endpoints call `submit_job()` from `server/submit.py` and return the job ID in < 200ms.
- [ ] All request bodies are validated by Pydantic models defined in `server/schemas.py`.

### US-002: Job status polling
**As a** consumer, **I want** to call `GET /jobs/{job_id}` **so that** I can check the current status of a job without holding a streaming connection.

**Acceptance Criteria:**
- [ ] `GET /jobs/{job_id}` returns a JSON object with fields: `id`, `status`, `created_at`, `updated_at`, and `result` (null or `JobResult`).
- [ ] Returns HTTP 404 with `{"detail": "job not found"}` when the job ID does not exist.
- [ ] Status values match: `"queued"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`.

### US-003: SSE progress stream
**As a** consumer, **I want** to connect to `GET /jobs/{job_id}/stream` and receive real-time progress events **so that** I can display a live progress bar without polling.

**Acceptance Criteria:**
- [ ] `GET /jobs/{job_id}/stream` returns a `text/event-stream` response using FastAPI's `StreamingResponse`.
- [ ] Each event is a JSON-encoded `PythonProgress` object emitted as `data: <json>\n\n`.
- [ ] When the job reaches `"completed"` or `"failed"`, a final event with `step: "done"` or `step: "error"` is emitted and the stream closes.
- [ ] If the job does not exist, the endpoint returns HTTP 404 immediately (no stream opened).

### US-004: Job list and cancel
**As a** consumer, **I want** to list recent jobs and cancel a queued job **so that** I can manage the job queue from external tools.

**Acceptance Criteria:**
- [ ] `GET /jobs` returns a JSON array of the 50 most recent jobs, each with `id`, `status`, `created_at`, `updated_at`.
- [ ] `DELETE /jobs/{job_id}` cancels a job that is in `"queued"` status; returns `{"cancelled": true}`.
- [ ] `DELETE /jobs/{job_id}` returns HTTP 409 with `{"detail": "job is already running or completed"}` when the job is not in `"queued"` status.

### US-005: Health endpoint
**As a** deployment operator, **I want** `GET /health` to return service status **so that** load balancers and monitoring tools can verify the gateway is alive.

**Acceptance Criteria:**
- [ ] `GET /health` returns HTTP 200 with `{"status": "ok", "version": "<package_version>"}`.
- [ ] The endpoint does not depend on the job queue being available (no DB call).

---

## Functional Requirements

- FR-1: All gateway routes are defined in `server/gateway.py` as an `APIRouter` and mounted on the main FastAPI app in `server/main.py` with prefix `/` (or no prefix).
- FR-2: All request/response schemas are Pydantic models in `server/schemas.py` with strict type annotations.
- FR-3: SSE streaming uses `asyncio` and polls the job queue every 500ms maximum; polling interval is configurable via `PARALLAX_SSE_POLL_MS` env var.
- FR-4: CORS is enabled on the gateway for `*` origins (configurable via `PARALLAX_CORS_ORIGINS` env var).
- FR-5: The gateway runs on the same `uvicorn` instance as `server/main.py` — no new process or port.

## Non-Goals

- No authentication or API key validation in this PRD.
- No WebSocket support — SSE is sufficient.
- No migration tooling for existing `parallax_ms` TypeScript consumers.
- No CLI implementation — covered by PRD 003.

## Open Questions

- None.
