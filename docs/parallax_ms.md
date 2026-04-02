# @parallax/ms (parallax media server)

Elysia HTTP gateway — the public-facing API for the Parallax platform. Receives requests from the CLI and MCP server, queues jobs, and proxies inference work to the FastAPI worker (`server/`).

## Role in the monorepo

Central API hub. All external consumers (CLI, MCP, future web clients) talk exclusively to this service. It never runs inference directly — it delegates to `server/` on `:5000`.

## Location

`packages/parallax_ms/`

## Stack

- Bun runtime
- Elysia ^1.x
- `@parallax/sdk` for shared types

## Running

```bash
# from repo root
bun run ms

# or from package dir
cd packages/parallax_ms && bun run dev
```

Starts on `http://localhost:3000`.

## Planned Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/v1/jobs` | Submit a generation job |
| `GET` | `/v1/jobs/:id` | Get job status |
| `GET` | `/v1/jobs/:id/events` | SSE stream — live job updates |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/outputs/:file` | Serve generated output files |

## Dependencies

```json
{
  "elysia": "^1.3.0",
  "@parallax/sdk": "workspace:*"
}
```
