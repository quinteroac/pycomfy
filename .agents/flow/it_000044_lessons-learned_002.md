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
