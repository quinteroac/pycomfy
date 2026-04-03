# Requirement: Image Generation via parallax-cli (sdxl, anima, z_image)

## Context

`parallax create image` already accepts all the necessary flags (`--model`, `--prompt`,
`--negative-prompt`, `--width`, `--height`, `--steps`, `--cfg`, `--seed`, `--output`)
but every model currently calls `notImplemented()`. The Python pipeline scripts for
`sdxl`, `anima`, and `z_image` already exist under `examples/image/generation/` with
a full `argparse` interface that mirrors these flags.

The goal is to wire the CLI to those scripts: the CLI action handler spawns a
`uv run python <script>` subprocess, passing every flag through, and exits with the
child's exit code. No new HTTP layer is needed; this is a pure CLI-to-subprocess
connection.

## Goals

- `parallax create image --model sdxl --prompt "..."` produces a PNG on disk.
- `parallax create image --model anima --prompt "..."` produces a PNG on disk.
- `parallax create image --model z_image --prompt "..."` produces a PNG on disk.
- The models directory is resolved from the `PYCOMFY_MODELS_DIR` environment variable
  (consistent with the existing example scripts).
- The `--output` flag in the CLI maps directly to `--output` in the subprocess.

## User Stories

### US-001: sdxl image generation via CLI

**As a** developer, **I want** to run `parallax create image --model sdxl --prompt "..."` **so that** the SDXL base+refiner pipeline generates a PNG at the given output path.

**Acceptance Criteria:**
- [ ] Running the command with a valid `PYCOMFY_MODELS_DIR` and `--prompt` spawns
      `uv run python examples/image/generation/sdxl/t2i.py` with the correct flags.
- [ ] `--negative-prompt`, `--width`, `--height`, `--steps`, `--cfg`, `--seed`,
      `--output` are forwarded verbatim to the subprocess.
- [ ] The CLI exits with the subprocess exit code (0 on success, non-zero on error).
- [ ] If `PYCOMFY_MODELS_DIR` is unset and no `--models-dir` flag is present, the CLI
      prints a clear error message and exits with code 1 before spawning the subprocess.
- [ ] Typecheck / lint passes (`bun tsc --noEmit`).

### US-002: anima image generation via CLI

**As a** developer, **I want** to run `parallax create image --model anima --prompt "..."` **so that** the Anima t2i pipeline generates a PNG at the given output path.

**Acceptance Criteria:**
- [ ] Running the command spawns
      `uv run python examples/image/generation/anima/t2i.py` with the correct flags.
- [ ] `--negative-prompt`, `--width`, `--height`, `--steps`, `--cfg`, `--seed`,
      `--output` are forwarded verbatim to the subprocess.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-003: z_image image generation via CLI

**As a** developer, **I want** to run `parallax create image --model z_image --prompt "..."` **so that** the Z-Image Turbo pipeline generates a PNG at the given output path.

**Acceptance Criteria:**
- [ ] Running the command spawns
      `uv run python examples/image/generation/z_image/turbo.py` with the correct flags.
- [ ] `--width`, `--height`, `--steps`, `--seed`, `--output` are forwarded verbatim.
- [ ] `--negative-prompt` and `--cfg` are **not** forwarded (z_image turbo has no
      negative prompt or CFG parameter); the CLI silently ignores them.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-004: models-dir resolution

**As a** developer, **I want** the CLI to resolve the models directory automatically **so that** I don't have to repeat the path on every command.

**Acceptance Criteria:**
- [ ] The CLI reads `PYCOMFY_MODELS_DIR` from the environment if no `--models-dir`
      flag is provided.
- [ ] A new optional `--models-dir <path>` flag on `create image` takes precedence
      over the env var when provided.
- [ ] If neither is set, the CLI prints
      `Error: --models-dir or PYCOMFY_MODELS_DIR is required` and exits with code 1.
- [ ] The resolved path is passed as `--models-dir <path>` to the subprocess.
- [ ] Typecheck / lint passes.

## Functional Requirements

- FR-1: The `create image` action handler in `packages/parallax_cli/src/index.ts`
  must dispatch to one of three scripts based on `opts.model`:
  - `sdxl`    → `examples/image/generation/sdxl/t2i.py`
  - `anima`   → `examples/image/generation/anima/t2i.py`
  - `z_image` → `examples/image/generation/z_image/turbo.py`
- FR-2: The script path must be resolved using the `PARALLAX_REPO_ROOT` environment
  variable as the repo root. If `PARALLAX_REPO_ROOT` is not set, the CLI must print
  `Error: PARALLAX_REPO_ROOT is required` and exit with code 1.
- FR-3: The subprocess must be spawned with `uv run python <script>` so the correct
  virtual environment is used automatically.
- FR-4: `--models-dir` (FR-4a: env var `PYCOMFY_MODELS_DIR`, FR-4b: explicit flag)
  must be resolved before spawning and passed as `--models-dir <path>` to the script.
- FR-5: `stdin`, `stdout`, and `stderr` of the child process must be inherited so
  progress output is visible to the user.
- FR-6: The CLI process must exit with the child's exit code.
- FR-7: z_image flags `--negative-prompt` and `--cfg` must be omitted from the
  subprocess command (the turbo script does not accept them).

## Non-Goals (Out of Scope)

- Implementing `flux_klein` or `qwen` image generation (remains `notImplemented`).
- Adding a `--variant` flag to select sdxl turbo vs. sdxl base+refiner.
- Adding HTTP routes to `parallax_ms` or `server/app.py`.
- Modifying any Python pipeline or example script.
- Adding automated tests for this iteration (critical-path tests are written in the
  Refactor phase per project convention).

## Open Questions

None.
