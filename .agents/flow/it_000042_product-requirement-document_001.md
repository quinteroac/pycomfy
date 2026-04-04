# Requirement: Per-Model CLI Defaults

## Context

The CLI (`parallax_cli`) currently defines a single set of global default values for `create image`, `create video`, and `create audio` commands (e.g. `steps=20`, `cfg=7`, `width=832`). These defaults do not match the defaults in each pipeline's `run()` function — the canonical source of truth. As a result, invoking the CLI without specifying parameters produces different output than calling the Python pipeline directly, and `--help` shows misleading values (e.g. `steps=60` for audio when `ace_step` uses 8). Each model also has structurally different optimal parameters (ltx2/ltx23 output 24–25 fps, wan2.x outputs 16 fps).

## Goals

- Per-model defaults in `registry.ts` are the single source of truth for all CLI parameter defaults, derived from the Python `run()` signatures.
- When a user invokes `parallax create video --model ltx2` without specifying `--width`, the pipeline receives `1280` (ltx2's default), not `832` (the old global default).
- `--help` footer shows a per-model defaults table so users see the correct values for the model they intend to use.

## User Stories

### US-001: Registry stores per-model defaults

**As a** developer or CLI user, **I want** each model's default parameter values to be declared in `registry.ts` **so that** there is one place to update when a pipeline's defaults change.

**Acceptance Criteria:**
- [ ] `registry.ts` defines a `ModelDefaults` interface with fields: `width?`, `height?`, `length?`, `fps?`, `steps?`, `cfg?` (all optional; absent means the param is not applicable for that model).
- [ ] Every model that has a `create` script has a corresponding `ModelDefaults` entry, populated from its `run()` signature (see table below).
- [ ] A `getModelDefaults(media, model)` helper is exported from `registry.ts` and returns `ModelDefaults | undefined`.
- [ ] Typecheck / lint passes.

**Reference defaults table (from Python `run()` signatures):**

| media | model | width | height | length | fps | steps | cfg | notes |
|-------|-------|-------|--------|--------|-----|-------|-----|-------|
| image | sdxl | 1024 | 1024 | — | — | 25 | 7.5 | |
| image | anima | 1024 | 1024 | — | — | 30 | 4.0 | |
| image | z_image | 1024 | 1024 | — | — | 4 | — | no `--cfg`, no `--negative-prompt` |
| video | ltx2 | 1280 | 720 | 97 | 24 | 20 | 4.0 | cfg flag is `--cfg-pass1` |
| video | ltx23 | 768 | 512 | 97 | 25 | — | 1.0 | distilled: no `--steps` arg |
| video | wan21 | 832 | 480 | 33 | 16 | 30 | 6.0 | |
| video | wan22 | 832 | 480 | 81 | — | 4 | 1.0 | no `fps` param in `run()` |
| audio | ace_step | — | — | 120 | — | 8 | 1.0 | `length` maps to `--duration` |

---

### US-002: CLI applies per-model defaults at runtime

**As a** CLI user, **I want** `parallax create video --model ltx2` (without `--width`) to use ltx2's width `1280` **so that** I get the pipeline's optimal output without having to memorize model-specific values.

**Acceptance Criteria:**
- [ ] In `create.ts`, the static commander defaults for `--width`, `--height`, `--length`, `--steps`, `--cfg`, and `--fps` (where applicable) are removed (i.e. not hardcoded in `.option(...)` calls).
- [ ] In each action handler, after the model is validated, `getModelDefaults(media, model)` is called and any param that is `undefined` (not supplied by the user) is filled with the model default before being passed to `buildArgs`.
- [ ] Running `parallax create video --model ltx2 --prompt "test"` passes `--width 1280 --height 720 --length 97 --steps 20 --cfg-pass1 4` to the Python script.
- [ ] Running `parallax create video --model wan21 --prompt "test"` passes `--width 832 --height 480 --length 33 --steps 30 --cfg 6` to the Python script.
- [ ] Running `parallax create audio --model ace_step --prompt "test"` passes `--duration 120 --steps 8 --cfg 1` to the Python script.
- [ ] Explicitly supplied flags override model defaults (e.g. `--steps 50` is forwarded as `--steps 50`).
- [ ] Typecheck / lint passes.

---

### US-003: `--help` footer shows per-model defaults table

**As a** CLI user, **I want** `parallax create video --help` to include a table of per-model defaults **so that** I can see the correct values for each model without reading Python source.

**Acceptance Criteria:**
- [ ] `create video --help` shows an "Defaults per model" section (via `addHelpText("after", ...)`) listing width, height, length, fps, steps, and cfg for ltx2, ltx23, wan21, and wan22.
- [ ] `create image --help` shows a similar table for sdxl, anima, and z_image.
- [ ] `create audio --help` shows the ace_step defaults (duration, steps, cfg, bpm).
- [ ] Table values match the entries in `ModelDefaults` from US-001 (no duplication; table is generated from `registry.ts` data).
- [ ] Visually verified in terminal (`parallax create video --help` output reviewed).

## Functional Requirements

- FR-1: `registry.ts` exports a `ModelDefaults` interface and a `getModelDefaults(media: string, model: string): ModelDefaults | undefined` function.
- FR-2: `ModelDefaults` fields are all optional (`?`) so models that lack a parameter (e.g. z_image has no `cfg`) simply omit it.
- FR-3: Commander option declarations in `create.ts` must not specify static defaults for dimension/sampling params — default resolution is `undefined` so the handler can detect omission.
- FR-4: Each action handler in `create.ts` applies model defaults before calling `buildArgs`, so `buildArgs` always receives fully-resolved values.
- FR-5: `addHelpText("after", ...)` in `create.ts` generates the defaults table dynamically from `getModelDefaults` calls, not from hardcoded strings.
- FR-6: Existing `cfgFlag` and `omitSteps` behaviour in `VIDEO_MODEL_CONFIG` and `buildVideoArgs` is preserved unchanged.
- FR-7: The `edit` commands are out of scope (all are `notImplemented`); their default values are not touched.

## Non-Goals (Out of Scope)

- Changing any Python pipeline code.
- Per-model sub-commands (no CLI restructure).
- `parallax_mcp` and `parallax_ms` changes (deferred to a future iteration).
- `edit image` / `edit video` commands (currently `notImplemented`).
- Adding new models or pipeline scripts.

## Open Questions

None.
