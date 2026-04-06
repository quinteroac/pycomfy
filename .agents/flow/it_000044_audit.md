# Audit Report — Iteration 000044, PRD 004 (Python MCP Server)

## Executive Summary

The MCP server entry-point scaffold (US-004) is fully implemented: `mcp/main.py` instantiates FastMCP in stdio mode, `mcp/__main__.py` delegates to it, `pyproject.toml` registers `parallax-mcp = "mcp.main:main"`, the server name is `"parallax-mcp"` with the version read from package metadata, and all US-004 tests pass. However, the actual MCP tools required by US-001 through US-003 are entirely absent — there is no `mcp/tools/` subdirectory, and no `create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`, `get_job_status`, or `wait_for_job` registrations anywhere in `mcp/`. FR-1 partially complies (missing `tools/`); FR-2, FR-3, FR-5, and FR-6 cannot comply because no tools exist.

---

## Verification by FR

| FR | Assessment | Notes |
|----|------------|-------|
| FR-1 | **partially_comply** | `mcp/main.py` and `mcp/__main__.py` exist and use fastmcp. The required `mcp/tools/` subdirectory is missing entirely. |
| FR-2 | **does_not_comply** | No tool files exist in `mcp/` that import `submit_job` or `get_queue`. |
| FR-3 | **does_not_comply** | No tool functions or Pydantic input models are defined anywhere in `mcp/`. |
| FR-4 | **comply** | `mcp/main.py` calls `mcp.run(transport="stdio")` — stdio transport correctly configured. |
| FR-5 | **does_not_comply** | `wait_for_job` is not implemented; `asyncio.sleep(2)` poll logic is absent. |
| FR-6 | **does_not_comply** | No tool functions exist, therefore no tool docstrings exist. |

---

## Verification by US

| US | Assessment | Notes |
|----|------------|-------|
| US-001 | **does_not_comply** | None of the five inference tools are implemented in `mcp/`. AC01–AC04 all unmet: no `submit_job()` calls, no 200ms job-ID response, no descriptive docstrings, no MCP unit tests. |
| US-002 | **does_not_comply** | `get_job_status` tool does not exist. AC01–AC03 all unmet. |
| US-003 | **does_not_comply** | `wait_for_job` tool does not exist. AC01–AC05 all unmet. |
| US-004 | **comply** | All four ACs met: `__main__.py` delegates to `main()` (AC01); script entry registered in `pyproject.toml` (AC02); server starts on EOF stdin (AC03); server name `"parallax-mcp"` with correct version (AC04). All tests in `test_mcp_us004_it044.py` pass. |

---

## Minor Observations

1. `mcp/__init__.py` namespace-package shim is sophisticated but untested. A regression test would protect the fragile `sys.path`/`__path__` manipulation.
2. FR-1 specifies a `tools/` subdirectory; all implementations should live there (e.g. `mcp/tools/inference.py`, `mcp/tools/jobs.py`) and be imported into `main.py`.
3. US-001-AC04 requires timing assertions (< 500ms per tool with a mock queue). These tests are absent even as stubs.
4. `server/submit.py:submit_job()` uses `asyncio.run()` internally, which will conflict when called from an already-running asyncio event loop (as in fastmcp async tool handlers). Tool implementations must call `await queue.enqueue(data)` directly rather than the synchronous wrapper.
5. fastmcp is pinned to `>=3.2.0`. The PRD open question about pinning to a specific minor version for reproducibility remains unresolved.

---

## Conclusions and Recommendations

Three of four user stories (US-001, US-002, US-003) and four of six functional requirements (FR-2, FR-3, FR-5, FR-6) are unimplemented.

---

## Refactor Plan

### Priority 1 — Create `mcp/tools/` with inference tools (US-001, FR-1, FR-2, FR-3, FR-6)

**File: `mcp/tools/inference.py`**

Implement five async tool functions registered with the FastMCP instance:

- `create_image(model, prompt, width, height, steps, cfg, seed, negative_prompt, output, models_dir)` → calls `submit_job()` async-safely, returns `f"job_id: {job_id}\nstatus: queued\nmodel: {model}"`
- `create_video(model, prompt, input, width, height, steps, cfg, seed, length, output, models_dir)`
- `create_audio(model, prompt, steps, cfg, seed, length, bpm, lyrics, output, models_dir)`
- `edit_image(model, prompt, input, width, height, steps, cfg, seed, output, models_dir)`
- `upscale_image(model, prompt, input, width, height, steps, cfg, seed, output, models_dir)`

All tools must:
- Use async-safe `submit_job` (call `queue.enqueue()` directly with `await` rather than `asyncio.run()`).
- Return within 200ms (enqueue is I/O-bound to SQLite, well within budget).
- Include a docstring mentioning the returned job ID and `wait_for_job`.

**File: `mcp/tools/jobs.py`**

- `get_job_status(job_id: str)` — queries queue, returns status/model/created_at/output_path or not_found.
- `wait_for_job(job_id: str, timeout_seconds: int = 600)` — polls with `asyncio.sleep(2)`, returns `output: <path>`, `error: <msg>`, or `error: timeout after <N>s`.

### Priority 2 — Register tools in `mcp/main.py`

Import both tool modules after creating the `mcp` instance so FastMCP auto-registers decorated functions:

```python
from mcp.tools import inference, jobs  # noqa: F401  – registers all tools
```

### Priority 3 — Add async-safe submit helper in `server/submit.py`

Add `async def submit_job_async(data: JobData) -> str` that calls `await (await get_queue()).enqueue(data)` without `asyncio.run()`, for use in fastmcp async tool handlers.

### Priority 4 — Unit tests for US-001, US-002, US-003

Create `tests/test_mcp_us001_it044.py`, `tests/test_mcp_us002_it044.py`, `tests/test_mcp_us003_it044.py` with a mock queue. US-001 tests must assert response format and timing (< 500ms).
