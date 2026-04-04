# Requirement: `edit image` and `upscale image` CLI Commands

## Context

`parallax edit image` exists in `edit.ts` but always calls `notImplemented`, and `registry.ts`
only lists `["qwen"]` with no script mappings. All runtime scripts already exist under
`packages/parallax_cli/runtime/image/edit/` for flux (5 variants), qwen, and two SD upscale
pipelines. The gap is purely in the TypeScript CLI layer: registry entries, arg builders, and
action handlers need to be wired up following the same pattern as `create.ts`.

Additionally, the two SD upscale scripts (`esrgan_upscale.py`, `latent_upscale.py`) are
structurally different from image-edit pipelines (they require a separate base checkpoint and
an upscale model), so they are exposed under a new top-level `parallax upscale image` command
rather than under `edit`.

## Goals

- `parallax edit image --model <flux_*|qwen> --prompt "..." --input photo.jpg` reaches the
  correct Python runtime script end-to-end.
- `parallax upscale image --model <esrgan|latent_upscale> ...` reaches its Python runtime
  script end-to-end.
- Both commands follow the exact same pattern as `create.ts`:
  `validate → resolveModelsDir → buildArgs → spawnPipeline`.
- TypeScript typecheck passes after all changes.

## User Stories

### US-001: Registry declares `edit image` and `upscale image` models and scripts

**As a** developer, **I want** `registry.ts` to be the single source of truth for edit/upscale
model names and their script paths **so that** adding future models only requires editing one
file.

**Acceptance Criteria:**
- [ ] `MODELS["edit image"]` is updated to:
  `["flux_4b_base", "flux_4b_distilled", "flux_9b_base", "flux_9b_distilled", "flux_9b_kv", "qwen"]`.
- [ ] A new `EDIT_IMAGE_SCRIPTS` map is added, keyed by model name, with the following values:

  | model | script |
  |---|---|
  | `flux_4b_base` | `runtime/image/edit/flux/4b_base.py` |
  | `flux_4b_distilled` | `runtime/image/edit/flux/4b_distilled.py` |
  | `flux_9b_base` | `runtime/image/edit/flux/9b_base.py` |
  | `flux_9b_distilled` | `runtime/image/edit/flux/9b_distilled.py` |
  | `flux_9b_kv` | `runtime/image/edit/flux/9b_kv.py` |
  | `qwen` | `runtime/image/edit/qwen/edit_2511.py` |

- [ ] `MODELS["upscale image"]` is added: `["esrgan", "latent_upscale"]`.
- [ ] A new `UPSCALE_IMAGE_SCRIPTS` map is added:

  | model | script |
  |---|---|
  | `esrgan` | `runtime/image/edit/sd/esrgan_upscale.py` |
  | `latent_upscale` | `runtime/image/edit/sd/latent_upscale.py` |

- [ ] `getScript("edit", "image", model)` returns the correct entry from `EDIT_IMAGE_SCRIPTS`.
- [ ] `getScript("upscale", "image", model)` returns the correct entry from
  `UPSCALE_IMAGE_SCRIPTS`.
- [ ] Typecheck / lint passes.

---

### US-002: `edit image` action wires up flux and qwen models

**As a** CLI user, **I want** `parallax edit image --model flux_4b_base --prompt "..." --input
photo.jpg` to actually invoke the Python script **so that** I can edit images from the terminal.

**Acceptance Criteria:**
- [ ] `edit.ts` imports `spawnPipeline`, `resolveModelsDir`, `readConfig`, `getScript`, and
  `buildEditImageArgs` (from `models/image.ts` or a new `models/edit_image.ts`).
- [ ] The `edit image` action handler follows the pattern:
  `validateModel → getScript → resolveModelsDir → buildEditImageArgs → spawnPipeline`.
- [ ] `--input` is a `requiredOption` and is validated to exist on disk before invoking the
  script (same guard as `--input` in `create video`).
- [ ] `buildEditImageArgs` maps CLI options to Python script args per model:
  - **All flux variants** (`flux_4b_base`, `flux_4b_distilled`, `flux_9b_base`,
    `flux_9b_distilled`): forwards `--models-dir`, `--prompt`, `--image <input>`, `--width`,
    `--height`, `--steps`, `--seed`, `--output`. Does **not** forward `--cfg` (script does not
    accept it).
  - **`flux_9b_kv`**: same as above, plus forwards `--subject-image` from the CLI
    `--subject-image` option. Both `--input` and `--subject-image` are validated to exist.
  - **`qwen`**: forwards `--models-dir`, `--image <input>`, `--prompt`, `--steps`, `--cfg`,
    `--seed`; maps `--output` to `--output-prefix` (strip `.png` suffix if present); forwards
    optional `--image2`, `--image3`; forwards `--no-lora` flag when set.
- [ ] Running `parallax edit image --model flux_4b_base --prompt "test" --input photo.jpg`
  invokes `uv run python runtime/image/edit/flux/4b_base.py --models-dir <dir> --prompt test
  --image photo.jpg ...` without error (verified in terminal).
- [ ] Running `parallax edit image --model qwen --prompt "change background" --input photo.jpg`
  invokes `uv run python runtime/image/edit/qwen/edit_2511.py ...` without error.
- [ ] Typecheck / lint passes.

---

### US-003: New `upscale image` command wires up esrgan and latent_upscale

**As a** CLI user, **I want** `parallax upscale image --model esrgan --checkpoint model.safetensors
--esrgan-checkpoint RealESRGAN_x4plus.safetensors --prompt "..."` to invoke the upscale Python
script **so that** I can upscale images from the terminal.

**Acceptance Criteria:**
- [ ] A new file `packages/parallax_cli/src/commands/upscale.ts` exports `registerUpscale`.
- [ ] `upscale image` command is structured identically to `create image` (validate →
  resolveModelsDir → buildArgs → spawnPipeline).
- [ ] CLI options for `upscale image`:
  - `--model <name>` (required) — `esrgan` or `latent_upscale`.
  - `--checkpoint <file>` (required) — base checkpoint filename (read from
    `PYCOMFY_CHECKPOINT` env if not supplied; validated non-empty before spawning).
  - `--esrgan-checkpoint <file>` — required when `--model esrgan`; read from
    `PYCOMFY_ESRGAN_CHECKPOINT` env if not supplied.
  - `--latent-upscale-checkpoint <file>` — required when `--model latent_upscale`; read from
    `PYCOMFY_LATENT_UPSCALE_CHECKPOINT` env if not supplied.
  - `--prompt <text>` (required).
  - `--negative-prompt <text>` (optional).
  - `--width <pixels>` (optional, default `"768"`).
  - `--height <pixels>` (optional, default `"768"`).
  - `--steps <n>` (optional, default `"20"`).
  - `--cfg <value>` (optional, default `"7"`).
  - `--seed <n>` (optional).
  - `--output <path>` (optional, default `"output.png"`).
  - `--output-base <path>` (optional, default `"output_base.png"`) — intermediate base image
    before upscaling.
  - `--models-dir <path>` (optional — overrides `PYCOMFY_MODELS_DIR`).
- [ ] `buildUpscaleImageArgs` in `models/image.ts` (or `models/edit_image.ts`) correctly maps
  CLI options to the esrgan/latent_upscale script flags, including model-specific required flags.
- [ ] The handler exits with an error if the model-specific required flag
  (`--esrgan-checkpoint` / `--latent-upscale-checkpoint`) is absent and the corresponding env
  var is not set.
- [ ] Running `parallax upscale image --model esrgan --checkpoint my.safetensors
  --esrgan-checkpoint RealESRGAN_x4plus.safetensors --prompt "test"` invokes
  `uv run python runtime/image/edit/sd/esrgan_upscale.py ...` without error (verified in
  terminal).
- [ ] Typecheck / lint passes.

---

### US-004: `index.ts` registers the upscale command

**As a** CLI user, **I want** `parallax upscale` to appear as a valid top-level command
**so that** it is discoverable via `parallax --help`.

**Acceptance Criteria:**
- [ ] `index.ts` imports `registerUpscale` from `./commands/upscale`.
- [ ] `registerUpscale(program)` is called alongside `registerCreate` and `registerEdit`.
- [ ] `parallax --help` lists `upscale` as a command.
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- FR-1: `getScript(action, media, model)` in `registry.ts` handles `action = "edit"` and
  `action = "upscale"` for `media = "image"`.
- FR-2: `EDIT_IMAGE_SCRIPTS` and `UPSCALE_IMAGE_SCRIPTS` are `Partial<Record<string, string>>`
  (same type as `IMAGE_SCRIPTS`).
- FR-3: `buildEditImageArgs` encapsulates all model-specific arg differences (no inline model
  logic in `edit.ts`).
- FR-4: `buildUpscaleImageArgs` encapsulates all model-specific arg differences (no inline
  model logic in `upscale.ts`).
- FR-5: The qwen `--output-prefix` mapping strips a trailing `.png` extension from the CLI
  `--output` value so `--output output.png` becomes `--output-prefix output`.
- FR-6: `flux_9b_kv` requires both `--input` and `--subject-image`; the handler validates
  both exist on disk before spawning.
- FR-7: The `edit video` command in `edit.ts` is left unchanged (still calls `notImplemented`).

## Non-Goals (Out of Scope)

- Implementing or modifying any Python pipeline code.
- Adding per-model defaults table to `--help` output (covered by PRD-001).
- `parallax upscale video` or any other upscale media type.
- `parallax_ms` and `parallax_mcp` changes.
- `flux_klein` or `qwen` create-image commands (already in `create.ts`).

## Open Questions

None.
