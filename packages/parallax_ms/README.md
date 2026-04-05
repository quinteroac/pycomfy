# @parallax/ms

HTTP microservice for the Parallax media-generation pipeline. Provides a REST API that submits inference jobs to a background queue and exposes endpoints for status polling and SSE streaming.

Built with [Elysia](https://elysiajs.com/) on Bun.

## Starting the server

```bash
# From the monorepo root
bun run packages/parallax_ms/src/index.ts

# Or
cd packages/parallax_ms && bun run start

# Watch mode (dev)
bun run dev
```

Default port: `3000` (override with the `PORT` env var).

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | HTTP port | `3000` |
| `PARALLAX_RUNTIME_DIR` | Root directory for Python scripts | `PARALLAX_REPO_ROOT` or `cwd` |
| `PARALLAX_REPO_ROOT` | Fallback script root | `cwd` |
| `PARALLAX_UV_PATH` | Path to the `uv` executable | `uv` |

## API reference

### `GET /health`

Returns server status and queue statistics.

```json
{ "status": "ok", "queue": { "waiting": 0, "active": 1, "completed": 12, "failed": 0 } }
```

---

### `GET /jobs`

List recent jobs.

**Query params:**

| Param | Values | Description |
|-------|--------|-------------|
| `status` | `waiting` \| `active` \| `completed` \| `failed` | Filter by status (optional) |

**Response:** array of job summaries with `id`, `status`, `progress`, `model`, `action`, `media`, `createdAt`, `duration`.

---

### `GET /jobs/:id`

Get the current status of a specific job.

**Response:**

```json
{
  "id": "42",
  "status": "completed",
  "progress": 100,
  "model": "sdxl",
  "action": "create",
  "media": "image",
  "output": "output.png",
  "error": null,
  "createdAt": 1712000000000,
  "startedAt": 1712000001000,
  "finishedAt": 1712000019000
}
```

Returns `404` if the job is not found.

---

### `GET /jobs/:id/stream`

Server-Sent Events stream that emits progress events until the job completes.

**Events:**

| Event | Payload | When |
|-------|---------|------|
| `progress` | `{ pct: number, step: string }` | While job is running (every 500 ms) |
| `completed` | `{ output: string }` | When job finishes successfully |
| `failed` | `{ error: string }` | When job fails |

Returns `404` if the job is not found.

---

### `DELETE /jobs/:id`

Cancel a waiting or active job.

**Responses:**

| Status | Body | Meaning |
|--------|------|---------|
| 200 | `{ "cancelled": true }` | Successfully cancelled |
| 404 | `{ "error": "Job not found" }` | Unknown ID |
| 409 | `{ "error": "Job already completed" }` | Already in terminal state |

---

### `POST /jobs/create/image`

Submit an image generation job.

**Body:**

```json
{
  "model": "sdxl",
  "prompt": "a red cube on a white background",
  "negative_prompt": "blurry",
  "width": 1024,
  "height": 1024,
  "steps": 25
}
```

**Response:** `{ "job_id": "42", "status": "queued" }`

Available models: `sdxl`, `anima`, `z_image`, `flux_klein`, `qwen`

---

### `POST /jobs/create/video`

Submit a video generation job.

**Body:**

```json
{
  "model": "wan22",
  "prompt": "ocean wave at sunset",
  "input": "/path/to/frame.png",
  "width": 832,
  "height": 480,
  "frames": 81,
  "steps": 4
}
```

`input` is optional — when provided enables image-to-video mode.

Available models: `ltx2`, `ltx23`, `wan21`, `wan22`

---

### `POST /jobs/create/audio`

Submit an audio generation job.

**Body:**

```json
{
  "model": "ace_step",
  "prompt": "epic orchestral, cinematic, dramatic",
  "duration_seconds": 60,
  "steps": 8
}
```

Available models: `ace_step`

---

### `POST /jobs/edit/image`

Submit an image editing job.

**Body:**

```json
{
  "model": "qwen",
  "image_path": "/path/to/photo.png",
  "prompt": "make the sky pink",
  "steps": 4
}
```

Available models: `flux_4b_base`, `flux_4b_distilled`, `flux_9b_base`, `flux_9b_distilled`, `flux_9b_kv`, `qwen`

---

### `POST /jobs/upscale/image`

Submit an image upscaling job.

**Body:**

```json
{
  "model": "esrgan",
  "image_path": "/path/to/low_res.png",
  "output": "upscaled.png"
}
```

Available models: `esrgan`, `latent_upscale`

---

## Typical workflow

```bash
# Submit a job
curl -X POST http://localhost:3000/jobs/create/image \
  -H "Content-Type: application/json" \
  -d '{"model":"sdxl","prompt":"sunset over mountains"}'
# → {"job_id":"42","status":"queued"}

# Poll status
curl http://localhost:3000/jobs/42

# Or stream progress
curl -N http://localhost:3000/jobs/42/stream

# Cancel if needed
curl -X DELETE http://localhost:3000/jobs/42
```

## Development

```bash
bun run dev        # Watch mode
bun run typecheck  # Type-check only
bun test           # Run tests
```
