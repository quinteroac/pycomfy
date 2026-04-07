# Lessons Learned — Iteration 000047

## US-001 — Access the frontend via the FastAPI server

**Summary:** Added static file serving for the compiled React frontend at `/ui` in the FastAPI server. `GET /ui` and `GET /ui/` both serve `index.html`; `/ui/assets/*` serves hashed JS/CSS bundles. Existing API routes are unaffected.

**Key Decisions:**
- Added `_mount_frontend_ui(app, dist_dir)` as a reusable helper in `server/main.py` so tests can create a fresh app instance with a mock dist directory rather than patching module-level state.
- Used `StaticFiles(directory=dist_dir, html=True)` for the mount (handles `/ui/` → `index.html` automatically) plus an explicit `@app.get("/ui")` route to satisfy the no-redirect requirement for `/ui` without trailing slash.
- `_FRONTEND_DIST` is resolved from env var `PARALLAX_FRONTEND_DIR` (if set) or defaults to `frontend/dist` relative to the repo root. Mount only happens when the directory exists, keeping CI from failing when frontend is not built.

**Pitfalls Encountered:**
- `StaticFiles` mount alone redirects `/ui` → `/ui/` (307). To return 200 directly on `/ui`, an explicit route must be registered *before* the mount. FastAPI's route resolution gives explicit routes priority over mounts.
- The app object is set up at module import time, so tests cannot patch `_FRONTEND_DIST` after import to change which routes exist. The solution is to expose `_mount_frontend_ui` and let tests construct a fresh FastAPI app.

**Useful Context for Future Agents:**
- `server/main.py` now imports `StaticFiles` and `FileResponse`; keep the lazy-import convention in mind for any future additions to `gateway.py` (those follow the lazy-import pattern), but `main.py` is the bootstrap file so eager imports are acceptable there.
- The frontend build output is `frontend/dist/` (Vite, `bun run build`). The built assets have content-hashed filenames like `index-Bc-gZLBk.js`.
- Tests for this feature use a fixture that creates a minimal mock `dist/` in `tmp_path`, registers the frontend routes on a fresh `FastAPI` instance, and returns a `TestClient`. This pattern avoids any coupling to the real built frontend.
