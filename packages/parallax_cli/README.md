# @parallax/cli

Command-line interface for the Parallax media-generation pipeline (images, videos, audio).

## Installation

```bash
# From the monorepo root
bun install
```

## Usage

```bash
# Run directly with bun
bun run packages/parallax_cli/src/index.ts --help

# Or via the bin entry-point after linking (see Global installation below)
parallax --help
```

## Commands

### `parallax create image`

Generate an image from a text prompt.

```bash
parallax create image --model <name> --prompt <text> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model <name>` | Model to use: `sdxl`, `anima`, `z_image`, `flux_klein`, `qwen` | required |
| `--prompt <text>` | Text prompt describing the image | required |
| `--negative-prompt <text>` | What to avoid in the output | — |
| `--width <pixels>` | Image width | model default |
| `--height <pixels>` | Image height | model default |
| `--steps <n>` | Sampling steps | model default |
| `--cfg <value>` | CFG guidance scale | model default |
| `--seed <n>` | Random seed for reproducibility | — |
| `--output <path>` | Output file path | `output.png` |
| `--models-dir <path>` | Models directory (overrides `PYCOMFY_MODELS_DIR`) | — |
| `--async` | Queue and return job ID immediately | — |

**Model defaults:**

| model     | width | height | steps | cfg  |
|-----------|-------|--------|-------|------|
| sdxl      | 1024  | 1024   | 25    | 7.5  |
| anima     | 1024  | 1024   | 30    | 4.0  |
| z_image   | 1024  | 1024   | 8     | 7.0  |

**Examples:**

```bash
parallax create image --model sdxl --prompt "a red cube on a white background"
parallax create image --model anima --prompt "fantasy landscape" --width 1280 --height 720
parallax create image --model sdxl --prompt "portrait" --async
```

---

### `parallax create video`

Generate a video from a text prompt (text-to-video) or from an image (image-to-video).

```bash
parallax create video --model <name> --prompt <text> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model <name>` | Model to use: `ltx2`, `ltx23`, `wan21`, `wan22` | required |
| `--prompt <text>` | Text prompt describing the video | required |
| `--input <path>` | Input image for image-to-video (optional) | — |
| `--width <pixels>` | Video width | model default |
| `--height <pixels>` | Video height | model default |
| `--length <frames>` | Number of frames | model default |
| `--steps <n>` | Sampling steps | model default |
| `--cfg <value>` | CFG guidance scale | model default |
| `--seed <n>` | Random seed | — |
| `--output <path>` | Output file path | `output.mp4` |
| `--models-dir <path>` | Models directory | — |
| `--async` | Queue and return job ID immediately | — |

**Model defaults:**

| model | width | height | length | fps | steps | cfg |
|-------|-------|--------|--------|-----|-------|-----|
| ltx2  | 1280  | 720    | 97     | 24  | 20    | 4.0 |
| ltx23 | 768   | 512    | 97     | 25  | —     | 1.0 |
| wan21 | 832   | 480    | 33     | 16  | 30    | 6.0 |
| wan22 | 832   | 480    | 81     | —   | 4     | 1.0 |

**Examples:**

```bash
parallax create video --model wan22 --prompt "a wave crashing on a beach"
parallax create video --model ltx2 --prompt "timelapse sunset" --input frame.png
parallax create video --model wan21 --prompt "flying eagle" --async
```

---

### `parallax create audio`

Generate audio from a text prompt.

```bash
parallax create audio --model <name> --prompt <text> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model <name>` | Model to use: `ace_step` | required |
| `--prompt <text>` | Text prompt (tags) describing the audio | required |
| `--length <seconds>` | Duration in seconds | 120 |
| `--steps <n>` | Sampling steps | 8 |
| `--cfg <value>` | CFG guidance scale | 1.0 |
| `--bpm <n>` | Beats per minute | 120 |
| `--lyrics <text>` | Lyrics text | — |
| `--seed <n>` | Random seed | — |
| `--output <path>` | Output file path | `output.wav` |
| `--models-dir <path>` | Models directory | — |
| `--unet <path>` | UNet path (overrides `PYCOMFY_ACE_UNET`) | — |
| `--vae <path>` | VAE path (overrides `PYCOMFY_ACE_VAE`) | — |
| `--text-encoder-1 <path>` | Text encoder 1 (overrides `PYCOMFY_ACE_TEXT_ENCODER_1`) | — |
| `--text-encoder-2 <path>` | Text encoder 2 (overrides `PYCOMFY_ACE_TEXT_ENCODER_2`) | — |
| `--async` | Queue and return job ID immediately | — |

**Examples:**

```bash
parallax create audio --model ace_step --prompt "epic cinematic orchestra, dramatic"
parallax create audio --model ace_step --prompt "jazz piano, relaxing" --bpm 90 --length 60
```

---

### `parallax edit image`

Edit an existing image using a text prompt.

```bash
parallax edit image --model <name> --prompt <text> --input <path> [options]
```

| Option | Description |
|--------|-------------|
| `--model <name>` | Model: `flux_4b_base`, `flux_4b_distilled`, `flux_9b_base`, `flux_9b_distilled`, `flux_9b_kv`, `qwen` |
| `--prompt <text>` | Desired edits description |
| `--input <path>` | Input image file |
| `--subject-image <path>` | Subject reference image (`flux_9b_kv` only) |
| `--width <pixels>` | Output width (default: 1024) |
| `--height <pixels>` | Output height (default: 1024) |
| `--steps <n>` | Sampling steps |
| `--cfg <value>` | CFG guidance scale |
| `--seed <n>` | Random seed |
| `--output <path>` | Output file (default: `output.png`) |
| `--image2 <path>` | Second input image (`qwen` only) |
| `--image3 <path>` | Third input image (`qwen` only) |
| `--no-lora` | Disable LoRA (`qwen` only) |
| `--models-dir <path>` | Models directory |
| `--async` | Non-blocking mode |

**Examples:**

```bash
parallax edit image --model qwen --prompt "make the sky pink" --input photo.png
parallax edit image --model flux_9b_kv --prompt "change outfit" --input person.png --subject-image ref.png
```

---

### `parallax upscale image`

Upscale an image using a super-resolution model.

```bash
parallax upscale image --model <name> --prompt <text> --input <path> [options]
```

| Option | Description |
|--------|-------------|
| `--model <name>` | Model: `esrgan`, `latent_upscale` |
| `--prompt <text>` | Text prompt |
| `--input <path>` | Input image file |
| `--checkpoint <file>` | Base checkpoint (or `PYCOMFY_CHECKPOINT`) |
| `--esrgan-checkpoint <file>` | ESRGAN checkpoint (`esrgan` model, or `PYCOMFY_ESRGAN_CHECKPOINT`) |
| `--latent-upscale-checkpoint <file>` | Latent upscale checkpoint (`latent_upscale` model) |
| `--negative-prompt <text>` | Negative prompt |
| `--width <pixels>` | Output width (default: 768) |
| `--height <pixels>` | Output height (default: 768) |
| `--steps <n>` | Sampling steps (default: 20) |
| `--cfg <value>` | CFG guidance scale (default: 7) |
| `--seed <n>` | Random seed |
| `--output <path>` | Upscaled output (default: `output.png`) |
| `--output-base <path>` | Intermediate base image (default: `output_base.png`) |
| `--models-dir <path>` | Models directory |
| `--async` | Non-blocking mode |

**Examples:**

```bash
parallax upscale image --model esrgan --prompt "sharp photo" --input low_res.png \
  --checkpoint v1-5.safetensors --esrgan-checkpoint RealESRGAN_x4plus.pth
```

---

### `parallax jobs`

Manage background jobs (submitted with `--async`).

```bash
parallax jobs list                  # Show recent jobs (newest first, max 20)
parallax jobs status <id>           # One-shot status block
parallax jobs status <id> --json    # Status as JSON
parallax jobs watch <id>            # Live-watch a job until it finishes
parallax jobs open <id>             # Open the output file in the default app
parallax jobs cancel <id>           # Cancel a running or waiting job
```

---

### `parallax mcp`

Start the MCP (Model Context Protocol) server. Used by AI agents to call Parallax tools programmatically.

```bash
parallax mcp
```

See [`packages/parallax_mcp`](../parallax_mcp/README.md) for full tool descriptions.

---

### `parallax install`

Install runtime dependencies (Python environments via `uv`).

---

## Async mode

All generation commands support `--async`, which submits the job to a background queue and immediately prints a job ID:

```bash
parallax create image --model sdxl --prompt "sunset" --async
# → job_id: 42

parallax jobs watch 42   # block until done
parallax jobs open  42   # open the result file
```

## Global installation

To invoke `parallax` as a global command, link the package with `bun link`:

```bash
cd packages/parallax_cli
bun link

which parallax   # should resolve to the bun shims directory
parallax --help
```

Unlink when finished:

```bash
bun unlink @parallax/cli
```

## Development

```bash
# Watch mode
bun run dev

# Type-check only
bun run typecheck
```

## Building a standalone binary

```bash
bun run build:linux   # dist/parallax-linux
bun run build:mac     # dist/parallax-macos
bun run build:win     # dist/parallax-windows.exe
```
