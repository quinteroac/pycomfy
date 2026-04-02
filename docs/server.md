# server (parallax-worker)

FastAPI service that wraps `comfy_diffusion` and exposes it over HTTP. Acts as the Python inference worker for the Elysia gateway (`@parallax/ms`).

## Role in the monorepo

Bridge between the TypeScript application layer and the Python core. Receives job requests from `@parallax/ms`, runs inference via `comfy_diffusion`, and returns results.

## Location

`server/` at repo root.

## Stack

- Python 3.12+
- FastAPI + uvicorn
- `comfy_diffusion` (local, from repo root)

## Running

```bash
# via uv
uv run server/app.py

# or from root shortcut
bun run server
```

Starts on `http://0.0.0.0:5000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness probe — returns `{"status": "ok"}` |

> More endpoints to be added: `POST /infer`, `GET /models`, etc.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port to listen on |
| `MODELS_DIR` | — | Path to model weights directory |

## Dependencies

```toml
# to be added to pyproject.toml optional group
fastapi >= 0.115.0
uvicorn[standard] >= 0.32.0
```
