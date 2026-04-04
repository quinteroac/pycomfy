# Requirement: parallax_mcp Async Tools

## Context
The MCP server currently blocks until each inference subprocess completes — which can take 5+ minutes for video generation. AI clients (Claude, Gemini, Copilot) have tool timeout limits that cause failures on long operations. This PRD makes all five inference tools non-blocking by returning a job ID immediately, and introduces two new tools (`get_job_status`, `wait_for_job`) that allow AI agents to observe job progress on their own schedule.

## Goals
- Refactor all five inference tools to return a job ID within 200ms instead of blocking until completion.
- Add `get_job_status` tool for single-shot status checks.
- Add `wait_for_job` tool that polls internally and resolves only when the job reaches a terminal state (with a configurable timeout).
- Remove the subprocess-spawning pattern from `parallax_mcp` — all execution goes through the shared job layer.

## User Stories

### US-001: Non-blocking inference tools
**As an** AI agent using the MCP server, **I want** `create_image`, `create_video`, `create_audio`, `edit_image`, and `upscale_image` to return a job ID immediately **so that** I can avoid tool timeouts on long-running operations and continue with other reasoning while inference runs.

**Acceptance Criteria:**
- [ ] All five tools call `submitJob()` from `@parallax/sdk/submit` instead of spawning `parallax_cli` as a subprocess.
- [ ] Each tool returns within 200ms with a text response of the form: `job_id: <id>\nstatus: queued\nmodel: <model>`.
- [ ] The tool descriptions in `registerTool` are updated to mention that they return a job ID, not the output path directly.
- [ ] The `Bun.spawn` subprocess pattern is fully removed from `parallax_mcp/src/index.ts`.
- [ ] `bun typecheck` passes on `parallax_mcp`.
- [ ] Manually verified: calling `create_video` via Claude returns a job ID in under 1 second.

### US-002: get_job_status tool
**As an** AI agent, **I want** to call `get_job_status` with a job ID and receive the current state, progress percentage, and output path (when done) **so that** I can check whether inference has finished before taking further actions.

**Acceptance Criteria:**
- [ ] `get_job_status` is registered with input schema `{ job_id: z.string() }`.
- [ ] Returns a JSON-formatted text response with fields: `id`, `status`, `progress`, `output` (null or string), `error` (null or string), `model`, `action`, `media`.
- [ ] When `job_id` does not exist, returns `isError: true` with message `Job <id> not found`.
- [ ] `bun typecheck` passes.

### US-003: wait_for_job tool
**As an** AI agent, **I want** to call `wait_for_job` with a job ID and optional timeout **so that** I can block until inference finishes and get the output path in one tool call — useful when the agent wants a synchronous-style result but without causing MCP transport timeouts.

**Acceptance Criteria:**
- [ ] `wait_for_job` is registered with input schema `{ job_id: z.string(), timeout_seconds: z.number().optional().default(600) }`.
- [ ] Polls `getQueue().getJob(id)` every 2 seconds until the job reaches `completed` or `failed`, or the timeout is exceeded.
- [ ] On `completed`: returns `{ status: "completed", output: "<path>", duration_seconds: N }`.
- [ ] On `failed`: returns `isError: true` with `{ status: "failed", error: "<reason>" }`.
- [ ] On timeout: returns `isError: true` with `{ status: "timeout", job_id: "<id>", message: "Job did not complete within <N> seconds. Use get_job_status to check later." }`.
- [ ] Closes the queue connection before returning.
- [ ] `bun typecheck` passes.

### US-004: Update tool descriptions for agent discoverability
**As an** AI agent discovering MCP tools, **I want** the tool descriptions to reflect the async pattern **so that** I know to call `get_job_status` or `wait_for_job` after submitting a job.

**Acceptance Criteria:**
- [ ] Each inference tool description ends with: `Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.`
- [ ] `get_job_status` description reads: `Check the current status and progress of a submitted inference job. Returns status, progress percentage (0-100), and output path when completed.`
- [ ] `wait_for_job` description reads: `Block until a submitted inference job completes. Polls internally every 2 seconds. Default timeout: 600 seconds. Returns output path on success.`
- [ ] `bun typecheck` passes.

## Functional Requirements
- FR-1: `@parallax/sdk` must be added as a `workspace:*` dependency in `parallax_mcp/package.json`.
- FR-2: `getQueue().close()` must be called after every tool handler that calls `getQueue()` — do not leave SQLite handles open across tool calls.
- FR-3: The `CLI_DIR` constant and all `Bun.spawn(["bun", "run", "src/index.ts", ...args])` calls must be removed.
- FR-4: All model validation (does the model exist for the given action/media?) must use `validateModel()` from `@parallax/cli/models/registry` or equivalent — do not silently submit jobs with invalid models.
- FR-5: Script path resolution for `ParallaxJobData.script` must use `getScript()` from the model registry — same logic as the CLI.

## Non-Goals (Out of Scope)
- SSE streaming from MCP — polling via `get_job_status` is sufficient.
- `list_jobs` MCP tool — AI agents don't need to browse job history.
- MCP progress notifications (MCP `progressToken`) — deferred to a future iteration.
- Any changes to the MCP server transport (stdio remains).

## Open Questions
- None.
