# Server Integration — Serve Frontend via FastAPI

## Context

Once the React+Bun frontend is built into a `dist/` directory, it needs to be served to the browser. Rather than running a separate static file server, the existing FastAPI application at `server/` will mount the `dist/` directory as static files, making the frontend accessible at `/ui` on the same `:5000` port. This avoids CORS issues and keeps the deployment surface minimal.

## Goals

- Serve the pre-built frontend from the FastAPI server with zero additional processes.
- Avoid breaking existing API routes.
- Make the frontend path configurable so `parallax frontend install` (PRD 003) can point to a custom directory.

## User Stories

### US-001: Access the frontend via the FastAPI server
**As a** user, **I want** to open `http://localhost:5000/ui` in a browser and see the Parallax chat interface **so that** I do not need to run a separate static file server.

**Acceptance Criteria:**
- [ ] Navigating to `http://localhost:5000/ui` returns `index.html` from the mounted frontend directory.
- [ ] All hashed JS and CSS assets referenced by `index.html` are served correctly from `/ui/assets/`.
- [ ] Navigating to `http://localhost:5000/ui/` (trailing slash) also returns `index.html`.
- [ ] Existing API routes (`/create/*`, `/jobs/*`) continue to respond correctly — no route collision with `/ui`.

### US-002: Configure the frontend directory path
**As an** operator, **I want** the frontend directory to be configurable via an environment variable **so that** `parallax frontend install` can point the server to the installed location without modifying source code.

**Acceptance Criteria:**
- [ ] The server reads `PARALLAX_FRONTEND_PATH` from the environment.
- [ ] If `PARALLAX_FRONTEND_PATH` is set and the path exists, the server mounts it at `/ui`.
- [ ] If `PARALLAX_FRONTEND_PATH` is not set, the server falls back to `frontend/dist/` relative to the repo root (development default).
- [ ] If neither path exists, the server starts normally without mounting `/ui` and logs a warning: `"Frontend not found at {path} — /ui will not be served."`

---

## Functional Requirements

- FR-1: Mount the frontend directory using FastAPI's `StaticFiles` at the `/ui` prefix.
- FR-2: The `index.html` is served for the root of the mount (`/ui` and `/ui/`).
- FR-3: `PARALLAX_FRONTEND_PATH` env var controls the mounted directory; fallback is `frontend/dist/` relative to the server working directory.
- FR-4: The mount is added in `server/main.py` after all API routers are registered, so API routes take precedence.
- FR-5: The server must not crash at startup if the frontend directory is absent — it logs a warning and skips the mount.

## Non-Goals

- Building the frontend (that is Bun's job, handled outside the server).
- Downloading or installing the frontend (covered by PRD 003).
- Reverse-proxy or CDN configuration.
- SPA client-side routing fallback (the frontend is a single-page app with no client-side routes beyond `/ui`).

## Open Questions

- None.
