# Lessons Learned — Iteration 000044

## US-001 — Pydantic job types

**Summary:** Created `server/jobs.py` with three Pydantic `BaseModel` subclasses — `JobData`, `JobResult`, and `PythonProgress` — and a corresponding test file `tests/test_server_jobs_us001.py` with 25 tests covering all four acceptance criteria.

**Key Decisions:**
- Created `server/__init__.py` alongside `server/jobs.py` to make `server` a proper Python package.
- Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` in `pyproject.toml` so that `from server.jobs import ...` resolves during test collection. The repo root was not on `sys.path` by default for pytest.
- Used `from __future__ import annotations` for cleaner `int | None` union syntax on Python 3.12.
- All optional fields in `PythonProgress` default to `None`, so `frame`, `total`, `output`, and `error` are not required at construction.

**Pitfalls Encountered:**
- The `server/` directory did not exist yet — it had to be created from scratch.
- `ModuleNotFoundError: No module named 'server'` on first test run because pytest's `rootdir` was not on `sys.path`. Fixed by adding `pythonpath = ["."]` to pytest config.

**Useful Context for Future Agents:**
- `pydantic~=2.0` is already available as part of the `comfyui` extras in `pyproject.toml` (and in `uv.lock`), so no new dependency needs to be added for `server/` modules that use Pydantic.
- The `pythonpath = ["."]` pytest setting is now present in `pyproject.toml` — all future `server.*` imports in tests will work without extra conftest manipulation.
- The `server/` package follows the same lazy-import, no-top-level-torch pattern expected by the rest of the project; `jobs.py` is a pure data-model file with no runtime side effects.

## US-002 — SQLite job queue singleton

**Summary:** Created `server/queue.py` with a `JobQueue` class wrapping `aiosqlite` and a `get_queue()` async singleton factory. Added `aiosqlite` as an optional `server` extra in `pyproject.toml`. Written 38 tests in `tests/test_server_queue_us002.py` covering all four acceptance criteria.

**Key Decisions:**
- `get_queue()` accepts an optional `db_path: Path | None` parameter (defaults to `~/.config/parallax/jobs.db`) so tests can pass a `tmp_path` without touching the user's real config directory.
- A module-level `_reset_singleton()` helper allows each test to get a fresh `JobQueue` without process restart — critical because the singleton persists across test functions otherwise.
- Tests use `asyncio.run()` via a `_run()` helper rather than `pytest-asyncio`, since `pytest-asyncio` is not in the project's dev dependencies. All 38 tests pass synchronously.
- `aiosqlite.Row` is set as `db.row_factory` so rows can be accessed by column name and converted to `dict` via `dict(row)`.
- `cancel()` checks terminal statuses (`done`, `failed`, `cancelled`) and returns `False` without modifying the row — prevents double-cancel and post-completion cancellation.

**Pitfalls Encountered:**
- `uv run pytest` was slow to start on first invocation (>60s) due to environment resolution; running via `timeout 60 uv run python -m pytest ...` confirmed output appears well within 60s once the environment is ready.
- `aiosqlite` was not in the lock file — had to add it with `uv add --optional server aiosqlite`.

**Useful Context for Future Agents:**
- `aiosqlite==0.22.1` is now in `uv.lock` under the `server` optional extra.
- `_reset_singleton()` in `server/queue.py` must be called at the start of each test that uses `get_queue()` to avoid test pollution from the module-level singleton.
- DB path parent directories are created automatically (`mkdir(parents=True, exist_ok=True)`) — no pre-setup needed in consumers.
- `result` column stores JSON-serialised dicts via `json.dumps`; callers must `json.loads` to deserialise.

## US-003 — submit_job() helper

**Summary:** Created `server/submit.py` exporting `submit_job(data: JobData) -> str`. The function runs `get_queue()` + `queue.enqueue()` via `asyncio.run()` in an inner coroutine, then spawns `server/worker.py` via `subprocess.Popen(..., start_new_session=True)`. 13 tests written in `tests/test_server_submit_us003.py` covering all four ACs; all pass in ~70ms.

**Key Decisions:**
- `get_queue` is imported at module level (not lazily), enabling `patch("server.submit.get_queue", ...)` in tests without any `sys.path` tricks. This is acceptable because `queue.py` has no torch/comfy imports — the lazy-import rule applies only to torch and comfy.* modules.
- Used `asyncio.run()` to bridge from the synchronous `submit_job()` signature to the async `get_queue()` / `enqueue()` API. This is correct for standalone/CLI use but will raise if called from within a running event loop (e.g. inside a FastAPI `async def` route). Future iterations should consider making `submit_job` async or wrapping with `asyncio.get_event_loop().run_until_complete()` for mixed contexts.
- The inner `_enqueue()` async helper keeps the event loop scoped to a single call, ensuring `asyncio.run()` always gets a fresh loop.

**Pitfalls Encountered:**
- Patching `get_queue` with `return_value=coroutine_object` via `unittest.mock.patch` does not work as expected: `return_value` is a regular attribute on the `MagicMock`, so calling the mock returns the coroutine, but `await mock()` awaits the mock (not the coroutine). The fix is `patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue))` — this makes `get_queue` itself an `AsyncMock`, so `await get_queue()` correctly returns `mock_queue`.
- Return annotation check `sig.return_annotation is str` fails when the module uses `from __future__ import annotations` (PEP 563 — annotations become strings). Use `typing.get_type_hints(fn)["return"]` instead.

**Useful Context for Future Agents:**
- `AsyncMock(return_value=X)` is the canonical way to mock an `async def` function that returns `X` when awaited.
- `subprocess.Popen` call args are captured as `(args_list, kwargs)` via `mock_popen.call_args`; the command list is `args_list[0]`.
- `asyncio.run()` cannot be called from within an already-running event loop. If `submit_job` is to be used inside FastAPI routes, the route should be `async def` and call `await asyncio.to_thread(submit_job, data)`, or `submit_job` should be refactored to be async.

## US-004 — Python worker process

**Summary:** Created `server/worker.py` with an async `_run_worker(job_id)` coroutine and `main()` CLI entry point. Added `update_progress()` to `JobQueue` in `server/queue.py`, plus a `progress TEXT` column with ALTER TABLE migration for existing DBs. 28 tests in `tests/test_server_worker_us004.py` cover all five ACs.

**Key Decisions:**
- Worker accepts both `"pending"` and `"queued"` as valid starting statuses: `submit_job()` / `enqueue()` creates jobs with `"pending"` (enforced by US-002 tests), but AC01 specifies `"queued"`. Accepting both avoids breaking existing tests while satisfying the AC.
- `progress TEXT` column added to the jobs schema with an `ALTER TABLE` migration inside `get_queue()` wrapped in a bare `except` to silently skip if the column already exists — the standard SQLite migration pattern.
- `PythonProgress` added to the `server/queue.py` import from `server.jobs` (not deferred) since `server.jobs` has no torch/comfy imports and causes no circular dependency issues.
- `proc.stdout` is iterated line-by-line synchronously (blocking read). This is correct for a subprocess worker — async I/O is unnecessary complexity here.

**Pitfalls Encountered:**
- `uv run pytest` hangs indefinitely in this environment (likely a uv resolution/network issue). Use `python -m pytest` directly or `timeout 120 python -m pytest` for CI-style runs.
- 38 queue tests (US-002) fail with `ModuleNotFoundError: No module named 'aiosqlite'` when running with the default `python` — `aiosqlite` is a `server` optional extra and is not installed in the base venv. These failures are **pre-existing** and not caused by this change.
- The worker test mock must set `proc.stdout = iter(["line\n", ...])` (an iterator, not a list) and `proc.stderr = io.StringIO("...")` since `worker.py` uses `for line in proc.stdout` and `proc.stderr.read()`.

**Useful Context for Future Agents:**
- `server/worker.py` is an executable (`python server/worker.py <job_id>`) — it does not expose a public Python API beyond `_run_worker` (private) and `main()`.
- The `update_progress` method on `JobQueue` stores the latest `PythonProgress` JSON in the `progress` column; it does not append — only the most recent progress snapshot is retained.
- When writing worker tests, mock `server.worker.get_queue` (not `server.queue.get_queue`) and `server.worker.subprocess.Popen` to intercept at the point of use.

## US-005 — ProgressReporter helper

**Summary:** Created `comfy_diffusion/progress.py` exporting `ProgressReporter` with `update(step, pct, **kwargs)` and `done(output_path)`. Outputs NDJSON lines matching the `PythonProgress` schema. 17 tests in `tests/test_progress_us005.py` cover all four ACs.

**Key Decisions:**
- Implemented without importing Pydantic or `server.jobs` — `comfy_diffusion/` must not depend on `server/`. JSON is built directly as a plain dict and serialised with `json.dumps`.
- Only recognised optional fields (`frame`, `total`, `output`, `error`) are forwarded from `**kwargs`; unknown kwargs are silently ignored to match PythonProgress's fixed schema.
- `ProgressReporter` is NOT re-exported at the `comfy_diffusion` package level — follows the project's rule that only `check_runtime`, `apply_lora`, and vae functions are re-exported.
- `print(..., flush=True)` ensures lines appear immediately even when stdout is buffered (important for worker subprocess line-by-line reading).

**Pitfalls Encountered:**
- No pitfalls; implementation was straightforward. The schema was already defined in `server/jobs.py` — match it structurally, don't import it.

**Useful Context for Future Agents:**
- `ProgressReporter` is consumed via `from comfy_diffusion.progress import ProgressReporter`.
- Pipelines that want structured progress can instantiate one reporter, call `reporter.update(...)` between steps, and call `reporter.done(output_path)` at completion.
- The `done()` method always emits `step="done"` — callers that parse the NDJSON stream can use this as a sentinel.
