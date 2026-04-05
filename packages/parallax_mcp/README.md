# @parallax/mcp

Model Context Protocol (MCP) server that exposes the Parallax media-generation pipeline as tools callable by AI agents (Claude, Cursor, etc.).

## Overview

The MCP server wraps all Parallax inference actions behind a standard MCP interface. Each tool submits a job to the background queue and returns a `job_id`. Use `get_job_status` or `wait_for_job` to track completion.

## Starting the server

```bash
# From the monorepo root
bun run packages/parallax_mcp/src/index.ts

# Or via the parallax CLI
parallax mcp
```

The server uses stdio transport — it is launched by the MCP host (e.g. Claude Desktop, Cursor) as a subprocess.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PARALLAX_RUNTIME_DIR` | Root directory for Python scripts | `PARALLAX_REPO_ROOT` or `cwd` |
| `PARALLAX_REPO_ROOT` | Fallback root directory | `cwd` |
| `PARALLAX_UV_PATH` | Path to the `uv` executable | `uv` |
| `PYCOMFY_MODELS_DIR` | Default models directory | — |

## Tools

### `create_image`

Generate an image. Returns a `job_id`.

| Input | Type | Description |
|-------|------|-------------|
| `model` | string | `sdxl`, `anima`, `z_image`, `flux_klein`, `qwen` |
| `prompt` | string | Text prompt |
| `negativePrompt` | string? | What to avoid |
| `width` | string? | Width in pixels |
| `height` | string? | Height in pixels |
| `steps` | string? | Sampling steps |
| `cfg` | string? | CFG guidance scale |
| `seed` | string? | Random seed |
| `output` | string? | Output file path (default: `output.png`) |
| `modelsDir` | string? | Overrides `PYCOMFY_MODELS_DIR` |

---

### `create_video`

Generate a video (text-to-video or image-to-video). Returns a `job_id`.

| Input | Type | Description |
|-------|------|-------------|
| `model` | string | `ltx2`, `ltx23`, `wan21`, `wan22` |
| `prompt` | string | Text prompt |
| `input` | string? | Input image path (enables image-to-video) |
| `width` | string? | Width in pixels |
| `height` | string? | Height in pixels |
| `length` | string? | Number of frames |
| `steps` | string? | Sampling steps |
| `cfg` | string? | CFG guidance scale |
| `seed` | string? | Random seed |
| `output` | string? | Output file path (default: `output.mp4`) |
| `modelsDir` | string? | Overrides `PYCOMFY_MODELS_DIR` |

---

### `create_audio`

Generate audio. Returns a `job_id`.

| Input | Type | Description |
|-------|------|-------------|
| `model` | string | `ace_step` |
| `prompt` | string | Text tags describing the audio |
| `length` | string? | Duration in seconds |
| `steps` | string? | Sampling steps |
| `cfg` | string? | CFG guidance scale |
| `bpm` | string? | Beats per minute |
| `lyrics` | string? | Lyrics text (`ace_step`) |
| `seed` | string? | Random seed |
| `output` | string? | Output file path (default: `output.wav`) |
| `modelsDir` | string? | Overrides `PYCOMFY_MODELS_DIR` |

---

### `edit_image`

Edit an existing image with a prompt. Returns a `job_id`.

| Input | Type | Description |
|-------|------|-------------|
| `model` | string | `flux_4b_base`, `flux_4b_distilled`, `flux_9b_base`, `flux_9b_distilled`, `flux_9b_kv`, `qwen` |
| `prompt` | string | Desired edits |
| `input` | string | Input image path |
| `subjectImage` | string? | Subject reference image (`flux_9b_kv` only) |
| `width` | string? | Output width |
| `height` | string? | Output height |
| `steps` | string? | Sampling steps |
| `cfg` | string? | CFG guidance scale |
| `seed` | string? | Random seed |
| `output` | string? | Output file path (default: `output.png`) |
| `image2` | string? | Second input image (`qwen` only) |
| `image3` | string? | Third input image (`qwen` only) |
| `noLora` | boolean? | Disable LoRA (`qwen` only) |
| `modelsDir` | string? | Overrides `PYCOMFY_MODELS_DIR` |

---

### `upscale_image`

Upscale an image. Returns a `job_id`.

| Input | Type | Description |
|-------|------|-------------|
| `model` | string | `esrgan`, `latent_upscale` |
| `prompt` | string | Text prompt |
| `input` | string | Input image path |
| `checkpoint` | string? | Base checkpoint filename (or `PYCOMFY_CHECKPOINT`) |
| `esrganCheckpoint` | string? | ESRGAN checkpoint (`esrgan`) |
| `latentUpscaleCheckpoint` | string? | Latent upscale checkpoint |
| `negativePrompt` | string? | Negative prompt |
| `width` | string? | Output width |
| `height` | string? | Output height |
| `steps` | string? | Sampling steps |
| `cfg` | string? | CFG guidance scale |
| `seed` | string? | Random seed |
| `output` | string? | Output file path (default: `output.png`) |
| `outputBase` | string? | Intermediate base image path |
| `modelsDir` | string? | Overrides `PYCOMFY_MODELS_DIR` |

---

### `get_job_status`

Poll the status of a submitted job.

| Input | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job ID returned by a create/edit/upscale tool |

Returns: `{ id, status, progress, output, error, model, action, media }`

Status values: `waiting` | `active` | `completed` | `failed`

---

### `wait_for_job`

Block until a job completes (polls every 2 seconds). Returns the output path on success.

| Input | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job ID to wait for |
| `timeout_seconds` | number? | Max wait time in seconds (default: 600) |

---

## Typical agent workflow

```
1. create_image(model="sdxl", prompt="a sunset over the ocean")
   → { job_id: "42" }

2. wait_for_job(job_id="42")
   → { status: "completed", output: "output.png", duration_seconds: 18 }
```

Or non-blocking:

```
1. create_image(...)      → job_id: "42"
2. get_job_status("42")   → { status: "active", progress: 60 }
3. get_job_status("42")   → { status: "completed", output: "output.png" }
```

## Development

```bash
bun run dev        # Watch mode
bun run typecheck  # Type-check only
```
