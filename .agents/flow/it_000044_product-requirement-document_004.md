# Python MCP Server (fastmcp)

## Context

`parallax_mcp` is currently a TypeScript MCP server using `@modelcontextprotocol/sdk`. All five inference tools block until the pipeline subprocess completes, causing timeout failures in AI clients. Since the job infrastructure is moving to Python (PRD 001), the MCP server can be reimplemented with `fastmcp` — a Python-native MCP framework — making all tools non-blocking by returning job IDs immediately, and adding `get_job_status` / `wait_for_job` tools that let AI agents observe progress on their own schedule.

## Goals

- Implement a Python MCP server under `mcp/` using `fastmcp`.
- Migrate all five inference tools to return a job ID within 200ms instead of blocking.
- Add `get_job_status` and `wait_for_job` tools.
- Register the server as a `uv run` entry point in `pyproject.toml`.

## User Stories

### US-001: Non-blocking inference tools
**As an** AI agent using the MCP server, **I want** `create_image`, `create_video`, `create_audio`, `edit_image`, and `upscale_image` to return a job ID immediately **so that** I can avoid tool timeouts on long-running operations and continue reasoning while inference runs.

**Acceptance Criteria:**
- [ ] All five tools call `submit_job()` from `server/submit.py` instead of blocking.
- [ ] Each tool returns within 200ms with a text response of the form: `job_id: <id>\nstatus: queued\nmodel: <model>`.
- [ ] The tool `description` fields mention that they return a job ID and that `wait_for_job` should be used to get the output path.
- [ ] A unit test verifies that each tool returns a string containing `job_id:` in under 500ms (with a mock queue).

### US-002: `get_job_status` tool
**As an** AI agent, **I want** to call `get_job_status(job_id)` **so that** I can check whether a job is still running, completed, or failed without blocking my reasoning loop.

**Acceptance Criteria:**
- [ ] `get_job_status` accepts `job_id: str` as its only parameter.
- [ ] Returns a text response with fields: `status`, `model`, `created_at`, and `output_path` (if completed) or `error` (if failed).
- [ ] Returns `status: not_found` (not an error/exception) when the job ID does not exist in the queue.

### US-003: `wait_for_job` tool
**As an** AI agent, **I want** to call `wait_for_job(job_id, timeout_seconds)` **so that** I can block my current tool call until the job finishes and get the output path directly.

**Acceptance Criteria:**
- [ ] `wait_for_job` accepts `job_id: str` and `timeout_seconds: int` (default 600).
- [ ] Polls the job queue every 2 seconds until the job status is `"completed"` or `"failed"`.
- [ ] On `"completed"`: returns `output: <path>` as text response.
- [ ] On `"failed"`: returns `error: <message>` as text response.
- [ ] On timeout: returns `error: timeout after <N>s` as text response (does not raise).

### US-004: MCP server entry point
**As a** developer configuring Claude Desktop or another MCP client, **I want** to point the client at `uv run parallax-mcp` **so that** the Python MCP server starts as a stdio process without needing a separate Bun install.

**Acceptance Criteria:**
- [ ] `mcp/__main__.py` exists and starts the `fastmcp` server in stdio mode.
- [ ] `pyproject.toml` registers a `parallax-mcp` script entry point pointing to `mcp.main:main`.
- [ ] `uv run parallax-mcp` starts without error (no GPU required — tools return early with mock job IDs in test mode).
- [ ] The server name reported to MCP clients is `"parallax-mcp"` with version matching `pyproject.toml`.

---

## Functional Requirements

- FR-1: MCP server is implemented with `fastmcp`; `mcp/` directory at repo root contains `main.py`, `tools/`, and `__main__.py`.
- FR-2: All tools import `submit_job` and `get_queue` from `server/submit.py` and `server/queue.py` — no direct `comfy.*` imports in `mcp/`.
- FR-3: Tool input schemas are defined as Pydantic models or typed function signatures (fastmcp auto-generates the JSON schema).
- FR-4: The MCP server runs as a stdio process (`fastmcp` stdio transport) — not HTTP.
- FR-5: `wait_for_job` uses `asyncio.sleep(2)` between polls — never busy-waits.
- FR-6: All tool docstrings are present and descriptive (fastmcp uses them as tool descriptions).

## Non-Goals

- No HTTP/SSE transport for MCP in this PRD — stdio only.
- No migration guide for existing `parallax_mcp` TypeScript consumers.
- No new inference capabilities beyond what exists in the current TS MCP server.

## Open Questions

- Should `fastmcp` be pinned to a specific version, or is the latest stable acceptable? (fastmcp is evolving rapidly — pinning reduces surprises.)
