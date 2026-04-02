# Requirement: parallax-cli Foundation

## Context

`@parallax/cli` exists as a scaffold (`src/index.ts` with a single TODO comment) but has no real commands.
The CLI must become the primary user-facing entry point for running comfy-diffusion pipelines from the terminal.
This iteration establishes the **command structure only** — no pipeline execution yet.
Subsequent iterations will wire each `--model` variant to its corresponding `comfy_diffusion/pipelines/*` module.

## Goals

- Define the full command/subcommand tree: `parallax create|edit image|video|audio`
- Declare all flags (`--model`, `--prompt`, `--input`, `--output`, and media-specific params) with types, defaults, and descriptions
- Validate required flags and known model values, exiting with code `1` and a clear error message on failure
- Ship as an installable global binary (`bun link` / `bun install -g`)
- Provide complete `--help` at every command level

## User Stories

### US-001: Top-level help
**As an** end user, **I want** to run `parallax --help` **so that** I can see all available commands and a short description of each.

**Acceptance Criteria:**
- [ ] `parallax --help` prints the tool name, version, description, and the list of subcommands (`create`, `edit`)
- [ ] `parallax` with no arguments shows help and exits with code `0`
- [ ] Typecheck / lint passes

### US-002: `create` subcommand help
**As an** end user, **I want** to run `parallax create --help` **so that** I can see the supported media types and their flags.

**Acceptance Criteria:**
- [ ] `parallax create --help` prints usage for `parallax create <media> [options]`
- [ ] Media types listed: `image`, `video`, `audio`
- [ ] Typecheck / lint passes

### US-003: `edit` subcommand help
**As an** end user, **I want** to run `parallax edit --help` **so that** I can see the supported media types for editing and their flags.

**Acceptance Criteria:**
- [ ] `parallax edit --help` prints usage for `parallax edit <media> [options]`
- [ ] Media types listed: `image`, `video`
- [ ] Typecheck / lint passes

### US-004: Media-level help (`create image`, `create video`, `create audio`, `edit image`, `edit video`)
**As an** end user, **I want** to run e.g. `parallax create image --help` **so that** I can see which flags are available for that specific media type.

**Acceptance Criteria:**
- [ ] Each of the 5 media commands shows its specific flags with type, default value (if any), and description:
  - **create image**: `--model` (required), `--prompt` (required), `--negative-prompt`, `--width`, `--height`, `--steps`, `--cfg`, `--seed`, `--output`
  - **create video**: `--model` (required), `--prompt` (required), `--width`, `--height`, `--length` (frames), `--steps`, `--cfg`, `--seed`, `--output`
  - **create audio**: `--model` (required), `--prompt` (required), `--length` (seconds), `--steps`, `--seed`, `--output`
  - **edit image**: `--model` (required), `--prompt` (required), `--input` (required), `--steps`, `--cfg`, `--seed`, `--output`
  - **edit video**: `--model` (required), `--prompt` (required), `--input` (required), `--width`, `--height`, `--length`, `--steps`, `--cfg`, `--seed`, `--output`
- [ ] Each media command's help footer includes the list of available models for that command, e.g.:
  `Available models: sdxl, anima, z_image, flux_klein, qwen`
- [ ] Typecheck / lint passes

### US-005: Known-model validation
**As an** end user, **I want** to receive a clear error when I pass an unknown `--model` value **so that** I know which models are supported.

**Acceptance Criteria:**
- [ ] When `--model` is not in the known list for that media type, the CLI prints:
  `Error: unknown model "<value>" for create image. Known models: sdxl, anima, z_image, flux_klein, qwen`
  (exact list varies per media type — see FR-4)
- [ ] Process exits with code `1`
- [ ] Typecheck / lint passes

### US-006: Required-flag validation
**As an** end user, **I want** to receive a clear error when I omit a required flag **so that** I know what's missing.

**Acceptance Criteria:**
- [ ] Omitting `--model` or `--prompt` on any `create` or `edit` command prints:
  `Error: --model is required` / `Error: --prompt is required`
- [ ] Omitting `--input` on any `edit` command prints: `Error: --input is required`
- [ ] Process exits with code `1`
- [ ] Typecheck / lint passes

### US-007: Stub execution
**As an** end user running a valid command, **I want** to see a "not yet implemented" message (instead of a crash) **so that** the CLI is usable as soon as a model is added.

**Acceptance Criteria:**
- [ ] A fully-valid command (all required flags present, model known) prints:
  `[parallax] create image --model sdxl — not yet implemented (coming soon)`
- [ ] Process exits with code `0`
- [ ] Typecheck / lint passes

### US-008: Installable binary
**As a** developer setting up the toolchain, **I want** to install the CLI globally with `bun link` **so that** `parallax` is available as a system command.

**Acceptance Criteria:**
- [ ] `bun link` in `packages/parallax_cli/` succeeds without errors
- [ ] After linking, running `parallax --help` from any directory works
- [ ] `package.json` `bin` field points to the correct entry file
- [ ] Typecheck / lint passes

## Functional Requirements

- **FR-1:** The CLI uses `commander` (already declared in `package.json`) as the parsing framework.
- **FR-2:** Command tree: `parallax` → `create` → `image | video | audio`; `parallax` → `edit` → `image | video`.
- **FR-3:** All flags are declared with `.option()` / `.requiredOption()` using long-form names (e.g. `--negative-prompt`). Short aliases are optional.
- **FR-3a:** Each media-level command appends an `addHelpText('after', ...)` block listing the available models, e.g.: `\nAvailable models: sdxl, anima, z_image, flux_klein, qwen`. The list is derived from the same model registry used for validation (FR-4) so they stay in sync.
- **FR-4:** Known model registry per media+action:
  - `create image`: `sdxl`, `anima`, `z_image`, `flux_klein`, `qwen`
  - `create video`: `ltx2`, `ltx23`, `wan21`, `wan22`
  - `create audio`: `ace_step`
  - `edit image`: `qwen`
  - `edit video`: `wan21`, `wan22`
- **FR-5:** Model validation runs inside the command action handler before any pipeline call. If invalid, print to `stderr` and `process.exit(1)`.
- **FR-6:** Required-flag validation: `--model` and `--prompt` are required on all 5 commands; `--input` is additionally required on both `edit` commands.
- **FR-7:** Stub action: when all validations pass, print a single line to `stdout`:
  `[parallax] <action> <media> --model <model> — not yet implemented (coming soon)`.
- **FR-8:** `package.json` `bin.parallax` must point to `./src/index.ts` (already correct) and the shebang `#!/usr/bin/env bun` must be present at line 1.
- **FR-9:** The `@parallax/sdk` `GenerateImageRequest` / `GenerateImageResponse` types are kept but not used yet; do not delete them. New SDK types for video and audio requests/responses may be added as placeholder interfaces.

## Non-Goals (Out of Scope)

- Actual pipeline execution — no calls to `comfy_diffusion` or HTTP to `parallax_ms` in this iteration
- `download` subcommand (model manifest download) — future iteration
- Config file (`~/.parallax/config.json`) — future iteration
- Shell autocomplete generation — future iteration
- Short-form flag aliases (e.g. `-m` for `--model`) — can be added later
- `parallax edit audio` — not a planned use case

## Open Questions

- None
