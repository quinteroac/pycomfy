# it_000044_refactor-plan_004.md

## Summary of changes

### `mcp/tools/` directory created (FR-1)
Created `mcp/tools/__init__.py`, `mcp/tools/inference.py`, and `mcp/tools/jobs.py` as required by the PRD.

### `mcp/tools/inference.py` — Five non-blocking inference tools (US-001)
Implemented async tool handlers: `create_image`, `create_video`, `create_audio`, `edit_image`, and `upscale_image`. Each tool:
- Builds a `JobData` from typed function parameters (fastmcp auto-generates JSON schema).
- Calls `await get_queue()` then `await queue.enqueue(data)` directly — avoids `asyncio.run()` conflict inside an async event loop (per audit observation).
- Spawns the worker subprocess with `subprocess.Popen(..., start_new_session=True)`.
- Returns within 200ms: `job_id: <id>\nstatus: queued\nmodel: <model>`.
- Docstrings mention job ID return and `wait_for_job` usage (US-001-AC03).

### `mcp/tools/jobs.py` — Status and polling tools (US-002, US-003)
- `get_job_status(job_id: str)`: queries the queue, returns `status`, `model`, `created_at`, `output_path` (if completed), or `error` (if failed). Returns `status: not_found` when the job ID is absent — never raises.
- `wait_for_job(job_id: str, timeout_seconds: int = 600)`: polls every `asyncio.sleep(2)` seconds; returns `output: <path>` on completion, `error: <msg>` on failure, `error: timeout after <N>s` on timeout.

### `mcp/main.py` — Tool registration (FR-2, FR-3, FR-6)
Imports all seven tool functions from `mcp/tools/` and registers them with the FastMCP instance via `mcp.tool()(fn)`. All imports are placed after `mcp` is instantiated to avoid circular references.

### `pyproject.toml` — Added `server*` to package finder
`server` was not in `tool.setuptools.packages.find.include`, so it was absent from the editable-install MAPPING. Added `"server*"` to the include list and re-ran `uv sync` so `server.jobs`, `server.queue`, etc. are importable from entry-point scripts.

### `tests/test_mcp_us001_us002_us003_it044.py` — Unit tests (US-001-AC04)
Added 22 tests covering:
- US-001: each inference tool returns a string containing `job_id:` in under 500ms (mocked queue), optional params are excluded when `None`.
- US-002: `get_job_status` returns correct fields for queued / completed / failed / not-found states; `job_id` is the only parameter.
- US-003: `wait_for_job` returns `output:` on completion, `error:` on failure, `error: job not found` when absent, `error: timeout after Ns` on timeout; polls with `asyncio.sleep(2)` between attempts; default timeout is 600.

---

## Quality checks

| Check | Command | Outcome |
|---|---|---|
| US-001–US-003 tests | `uv run pytest tests/test_mcp_us001_us002_us003_it044.py -v` | ✅ 22 passed |
| US-004 tests | `uv run pytest tests/test_mcp_us004_it044.py -v` | ✅ 9 passed |
| Full MCP suite | `uv run pytest tests/test_mcp_us001_us002_us003_it044.py tests/test_mcp_us004_it044.py -v` | ✅ 31 passed |
| Non-server test suite | `uv run pytest --ignore=tests/test_server_*.py -q` | 190 pre-existing failures (torch/ComfyUI absent in CPU-only env); all new tests pass. |

Pre-existing failures are all in ComfyUI/torch-dependent modules and are unrelated to this refactor.

---

## Deviations from refactor plan

None.

All items from the audit recommendations were implemented:
1. `mcp/tools/` subdirectory created with `inference.py` and `jobs.py`.
2. All seven tools registered with the FastMCP instance in `mcp/main.py`.
3. Each inference tool returns `job_id: <id>\nstatus: queued\nmodel: <model>` within 200ms.
4. `wait_for_job` polls with `asyncio.sleep(2)` and returns timeout-as-text.
5. Unit tests added for US-001 through US-003 with mock queue.
6. The async `asyncio.run()` conflict noted in the audit minor observations was addressed by calling `await queue.enqueue(data)` directly instead of using `submit_job()`.
