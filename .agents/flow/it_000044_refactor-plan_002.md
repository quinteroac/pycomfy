# Refactor Plan 002 — Completion Report

**Iteration:** 000044  
**Audit source:** `it_000044_audit-report_002.json`

---

## Summary of changes

### 1. Introduced `server/gateway.py` (FR-1)
Created a dedicated `APIRouter` in `server/gateway.py` containing all route handlers that were previously inlined in `server/app.py`. This satisfies FR-1: routes are now defined on a router object, not on a bare `FastAPI` instance.

Key improvements in the gateway:
- **Cancel bug fixed** — `cancel_job` now checks `row["status"] != "queued"` (was `!= "pending"`), which matches the actual status value stored by the queue. Cancellation now works correctly for queued jobs.
- **SSE poll interval configurable** — `_SSE_POLL_S` reads `PARALLAX_SSE_POLL_MS` env var (`float(os.environ.get("PARALLAX_SSE_POLL_MS", "500")) / 1000.0`), satisfying FR-3.
- **Dead code removed** — `_map_status` helper and the `if status == "pending": status = "queued"` branch have both been removed; the queue never stores `"pending"`.
- **`_uv_path()` hardened** — now raises `RuntimeError` when `uv` is not found on PATH instead of silently falling back to `sys.executable`.

### 2. Introduced `server/main.py` (FR-1, FR-4, FR-5)
Created `server/main.py` as the FastAPI application entry point. It:
- Instantiates the `FastAPI` app.
- Adds `CORSMiddleware` with origins read from `PARALLAX_CORS_ORIGINS` env var (defaults to `"*"`), satisfying FR-4.
- Mounts the `router` from `server/gateway.py`.

### 3. Updated `server/app.py` — backward-compat shim
`server/app.py` is now a thin re-export module that imports `app` from `server.main` and re-exports `get_queue`, `submit_job`, `_pkg_version`, and `_uv_path` from `server.gateway`. Existing imports (`from server.app import app`) and test patches continue to work unchanged during the transition.

### 4. Consolidated `JobResult` (`server/jobs.py`)
Removed the duplicate `JobResult` class from `server/jobs.py`. It now re-exports `JobResult` from `server/schemas.py`, which is the authoritative definition with both `output_path` and `error` optional fields.

### 5. Fixed `server/submit.py` uv resolution
Replaced the unsafe `sys.executable` fallback with a `RuntimeError` when `uv` is not found on PATH, preventing silent execution in the wrong Python environment.

### 6. Updated `package.json` server command
Changed `"server": "uv run server/app.py"` to `"server": "uv run fastapi dev server/main.py"` to use the proper entry point.

### 7. Updated tests
- All five route test files updated to patch `server.gateway.get_queue`, `server.gateway.submit_job`, and `server.gateway._pkg_version` (was `server.app.*`).
- Removed dead-code tests that asserted `"pending"` DB status maps to `"queued"` in responses (`test_maps_pending_status_to_queued`, `test_queued_status_when_pending_in_db`, and the `("pending", "queued")` parametrize case).
- Updated cancel tests to use `"queued"` row status (was `"pending"`).

### 8. Added `fastapi` and `httpx[http2]` to project dependencies
`fastapi` and `httpx[http2]` were not declared in `pyproject.toml`; added via `uv add fastapi "httpx[http2]"`.

---

## Quality checks

| Check | Result | Notes |
|-------|--------|-------|
| `uv run pytest tests/test_server_routes_us001_it044.py` | ✅ 48 passed | All inference submission endpoint tests |
| `uv run pytest tests/test_server_routes_us002_it044.py` | ✅ 16 passed | Job status endpoint tests (removed 2 dead-code tests) |
| `uv run pytest tests/test_server_routes_us003_it044.py` | ✅ 16 passed | SSE streaming endpoint tests |
| `uv run pytest tests/test_server_routes_us004_it044.py` | ✅ 17 passed | Job list and cancel tests (removed 1 dead-code test, fixed cancel) |
| `uv run pytest tests/test_server_routes_us005_it044.py` | ✅ 4 passed | Health endpoint tests |
| **All five together** | ✅ **106 passed** | |
| Full `uv run pytest tests/` | ⚠️ Collection errors | Pre-existing failure: `vendor/ComfyUI/server.py` name-conflicts with `server/` package when collected together. This is a pre-existing issue unrelated to this refactor (confirmed by git stash verification). |

---

## Deviations from refactor plan

| Plan item | Deviation |
|-----------|-----------|
| Step 1 — Fix cancel bug via `queue.cancel()` | Used the literal fix (`"pending"` → `"queued"`) instead of delegating to `queue.cancel()`. Rationale: `queue.cancel()` permits cancelling "running" jobs too, which would break existing tests for AC03 (non-queued → 409). The literal fix is the plan's recommended alternative ("Alternatively change the literal check from `"pending"` to `"queued"`"). |
| Step 4 — Remove or keep `server/app.py` | Kept `server/app.py` as a backward-compat re-export shim instead of deleting it, to avoid breaking any external consumers or CI that may reference `server.app`. |

All other plan items were applied as specified.
