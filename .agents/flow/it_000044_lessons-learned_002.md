# Lessons Learned — Iteration 000044

## US-001 — Submit inference jobs via REST

**Summary:** Implemented 5 FastAPI REST endpoints (`POST /jobs/create/image`, `/jobs/create/video`, `/jobs/create/audio`, `/jobs/edit/image`, `/jobs/upscale/image`) in `server/app.py`, Pydantic request/response schemas in `server/schemas.py`, and 48 tests in `tests/test_server_routes_us001_it044.py`. All tests pass (48/48).

**Key Decisions:**
- Created `server/schemas.py` with 6 Pydantic models: `CreateImageRequest`, `CreateVideoRequest`, `CreateAudioRequest`, `EditImageRequest`, `UpscaleImageRequest`, and `JobResponse`.
- Created `server/app.py` with a FastAPI `app` instance and 5 route handlers; each handler builds a `JobData` and calls `submit_job()` from `server/submit.py`.
- `_make_args()` helper strips `None` values and the `model` field from request dicts, passing remaining fields as `args` to `JobData`.
- Script path is derived from model name as a convention: `comfy_diffusion/pipelines/{media}/{model}/run.py`.
- `_uv_path()` uses `shutil.which("uv")` to find the uv binary, falling back to `sys.executable`.

**Pitfalls Encountered:**
- `fastapi` was not installed in the test environment (`/home/victor/AI/pycomfy/.venv`). Required manual install via `uv pip install fastapi httpx --python /home/victor/AI/pycomfy/.venv/bin/python3`. This should be documented for future agents.
- The project `.venv` is a sparse development venv (only aiosqlite, mypy, pytest), while tests run against the **pycomfy** venv at `/home/victor/AI/pycomfy/.venv`. Always test with the `python3` from that path.
- `test_server_queue_us002.py` has 19 pre-existing failures due to `aiosqlite` not being installed in the test venv — this is unrelated to this user story.

**Useful Context for Future Agents:**
- Tests run via `/home/victor/AI/pycomfy/.venv/bin/python3 -m pytest` (not via `uv run pytest`). The test environment is the pycomfy venv, not the project's own `.venv`.
- `fastapi` 0.135.3 and `starlette` 1.0.0 are now installed in the pycomfy venv. Use `starlette.testclient.TestClient` for endpoint testing (not `fastapi.testclient`).
- The `server/app.py` FastAPI `app` object is the entry point for future route additions. Follow the existing pattern: schema → `_make_args()` → `JobData` → `submit_job()` → `JobResponse`.
- `pyproject.toml` `server` extra currently only declares `aiosqlite`; if FastAPI becomes a permanent dependency, add `fastapi>=0.135` to the `server` extras and run `uv lock --upgrade-package fastapi`.

## US-002 — Job status polling

**Summary:** Implemented `GET /jobs/{job_id}` endpoint in `server/app.py`, added `JobResult` and `JobStatusResponse` Pydantic models to `server/schemas.py`, and wrote 16 tests in `tests/test_server_routes_us002_it044.py`. All 16 tests pass.

**Key Decisions:**
- Added `JobResult` (with optional `output_path` and `error` fields) and `JobStatusResponse` (with `id`, `status`, `created_at`, `updated_at`, `result`) to `server/schemas.py`.
- The route handler is synchronous (like other routes), using `asyncio.run()` to call the async `get_queue()` — consistent with the `submit_job()` pattern.
- DB stores status as `"pending"` for newly submitted jobs; the GET endpoint maps `"pending"` → `"queued"` to satisfy AC03 status contract.
- `result` column in DB is a JSON string or NULL; parsed with `json.loads()` when present.

**Pitfalls Encountered:**
- `JobResult` already existed in `server/jobs.py` (with only `output_path: str`). A new, more flexible `JobResult` (with optional `output_path` and `error`) was defined in `server/schemas.py` for the public API — the one in `jobs.py` is internal and not affected.
- Tests must mock `server.app.get_queue` (not `server.queue.get_queue`) since `app.py` imports `get_queue` directly into its namespace.

**Useful Context for Future Agents:**
- Run tests with `/home/victor/AI/pycomfy/.venv/bin/python3 -m pytest` — not `uv run pytest`.
- `server/schemas.py` now exports: `JobResult`, `JobStatusResponse`, `JobResponse`, and all request models. Import from there for response types.
- Status mapping: DB `"pending"` → API `"queued"`; `"running"`, `"completed"`, `"failed"`, `"cancelled"` pass through unchanged.
- The `result` field in the DB row is a JSON string when set. Always call `json.loads()` before constructing `JobResult`.

## US-003 — SSE progress stream

**Summary:** Implemented `GET /jobs/{job_id}/stream` as an async FastAPI endpoint returning a `StreamingResponse` with `media_type="text/event-stream"`. The endpoint polls the DB for progress updates and emits `data: <json>\n\n` SSE events until the job reaches `"completed"` or `"failed"`, then emits a final `step: "done"` or `step: "error"` event and closes. 16 tests cover all ACs.

**Key Decisions:**
- Made the route `async def` (not sync with `asyncio.run()`) so it can yield from an async generator directly into `StreamingResponse`.
- The initial 404 check (`await queue.get(job_id)` before the generator) ensures the response is a proper 404 HTTPException, not an empty stream.
- Duplicate progress events are suppressed by comparing `progress_json` to `last_progress_json` — only changed values are emitted.
- `PythonProgress` imported from `server.jobs` (already had the model); `StreamingResponse` imported from `fastapi.responses`.
- `asyncio.sleep(0.5)` is the polling interval; tests bypass it by making the mock queue immediately return terminal status (or by mocking `asyncio.sleep` with `AsyncMock`).

**Pitfalls Encountered:**
- Route ordering: `/jobs/{job_id}/stream` must not conflict with `/jobs/{job_id}`. In FastAPI/Starlette, `{job_id}` only matches a single path segment, so there is no ambiguity. The stream route was added after `get_job_status` without issues.
- The mock queue's `get()` is called **twice** per completed job: once before the generator starts (the existence check in the route handler) and once inside the generator loop. `side_effect` lists must include at least 2 entries for terminal-status tests.

**Useful Context for Future Agents:**
- Tests run via `/home/victor/AI/pycomfy/.venv/bin/python3 -m pytest` — not `uv run pytest`.
- Mock `server.app.get_queue` (not `server.queue.get_queue`) — it's imported directly into `app.py`'s namespace.
- `AsyncMock` works correctly for both `get_queue` (which returns a queue object) and `queue.get` (which returns row dicts). Use `side_effect=` with a list to simulate multiple polling calls.
- When the job is in a terminal state immediately, `asyncio.sleep` is never reached, so no need to mock it for those test cases.
- `starlette.testclient.TestClient` (sync) can consume SSE responses via `client.get()` when the generator terminates: `response.text` contains the full SSE body.

## US-004 — Job list and cancel

**Summary:** Implemented `GET /jobs` (returns 50 most recent jobs as `JobListItem` array) and `DELETE /jobs/{job_id}` (cancels queued jobs or returns 409). Added `JobListItem` Pydantic model to `server/schemas.py`, added two route handlers to `server/app.py`, and wrote 25 tests in `tests/test_server_routes_us004_it044.py`. All 25 tests pass.

**Key Decisions:**
- `GET /jobs` reuses the existing `queue.list_jobs(limit=50)` method already present in `server/queue.py`.
- `DELETE /jobs/{job_id}` reads the job status directly via `queue.get()` and calls `queue.update_status(job_id, "cancelled")` rather than using `queue.cancel()`, because the existing `cancel()` method allows cancelling running jobs (only blocks on "done"/"failed"/"cancelled"), which would violate AC03.
- Route ordering: `GET /jobs` is registered before `GET /jobs/{job_id}` to avoid the literal path being shadowed by the parameterized one. FastAPI handles this correctly regardless, but explicit ordering is cleaner.
- Status mapping: DB `"pending"` → API `"queued"` applied in both `GET /jobs` and the existing `GET /jobs/{job_id}` route for consistency.
- Added `JobListItem` (id, status, created_at, updated_at) to `schemas.py` — deliberately separate from `JobStatusResponse` (which also carries `result`).

**Pitfalls Encountered:**
- `queue.cancel()` in `server/queue.py` only blocks cancellation for `("done", "failed", "cancelled")` — it would also cancel "running" jobs, violating AC03. Always check the exact cancellation guard logic before delegating to a helper method.
- The `get_queue` singleton is called twice in `cancel_job` (once for the `get` check, once for `update_status`). The `side_effect` mock function returns the same mock queue object on each call, which works cleanly.

**Useful Context for Future Agents:**
- Tests run via `/home/victor/AI/pycomfy/.venv/bin/python3 -m pytest` — not `uv run pytest`.
- `server/schemas.py` now exports `JobListItem` in addition to all previous models.
- When adding new list/query endpoints that need the queue, follow the `asyncio.run(_inner())` pattern used by all sync routes in `app.py`.
- `queue.list_jobs(limit=50)` is already implemented in `server/queue.py` — no DB changes were required.

## US-005 — Health endpoint

**Summary:** Added `GET /health` to `server/app.py`. Returns `{"status": "ok", "version": "<pkg_version>"}` using `importlib.metadata.version("comfy-diffusion")`. No DB dependency. 4 tests in `tests/test_server_routes_us005_it044.py`, all passing.

**Key Decisions:**
- Used `importlib.metadata.version` (stdlib, no extra deps) to read the package version at call time.
- Route registered before all `/jobs/*` routes so it doesn't conflict.
- No schema model needed — simple `dict` return type suffices.

**Pitfalls Encountered:**
- None. This was a minimal, self-contained endpoint with no DB or model-loading dependency.

**Useful Context for Future Agents:**
- `_pkg_version` is imported at module top as `from importlib.metadata import version as _pkg_version` — mock it as `server.app._pkg_version` in tests.
- Tests run via `/home/victor/AI/pycomfy/.venv/bin/python3 -m pytest` — not `uv run pytest`.
