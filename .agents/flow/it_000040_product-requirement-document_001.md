# Requirement: Refactor parallax_cli — Compiled Binary, Interactive Install, Persistent Config & Layer Separation

## Context

`packages/parallax_cli` is a 310-line monolithic `src/index.ts` file. It hardcodes `PARALLAX_REPO_ROOT`, duplicates `--models-dir` validation in three action handlers, and embeds model-specific logic directly in command handlers. It has no persistent configuration, no interactive installer, and cannot be distributed as a self-contained binary. This iteration restructures the package into a clean layered architecture to support compiled binary distribution (`bun build --compile`), a guided `parallax install` UX (via `@clack/prompts`), and persistent user config at `~/.config/parallax/config.json` — while preserving full backward compatibility with `PARALLAX_REPO_ROOT` + `PYCOMFY_MODELS_DIR` env vars for CI usage.

## Goals

- Enable distribution of `parallax` as a self-contained compiled binary (Linux x64, macOS arm64)
- Provide a guided `parallax install` command with interactive clack UI for first-time setup
- Persist user configuration in `~/.config/parallax/config.json` so env vars are not required on every invocation
- Decouple model registry, argument builders, config I/O, and subprocess runner from command handlers
- All existing black-box CLI tests continue to pass after the refactor

## User Stories

### US-001: Compiled binary distribution
**As a** developer, **I want** to run `bun run build` and get a self-contained `parallax` binary (Linux + macOS) **so that** I can distribute and run the CLI without Bun installed on the target machine.

**Acceptance Criteria:**
- [ ] `package.json` includes `build:linux`, `build:mac`, and `build` scripts using `bun build --compile`
- [ ] `@clack/prompts` is added to `dependencies` in `package.json`
- [ ] `bun run build` completes without errors and produces binaries in `dist/`
- [ ] `bun run typecheck` passes with zero errors

---

### US-002: Persistent configuration layer (`config.ts`)
**As a** CLI user, **I want** my `repoRoot` and `modelsDir` saved after install **so that** I don't need to export env vars on every terminal session.

**Acceptance Criteria:**
- [ ] `src/config.ts` exports `readConfig()`, `writeConfig(config)`, and `configExists()` functions
- [ ] Config is stored as JSON at `~/.config/parallax/config.json` with fields: `repoRoot`, `modelsDir`, `uvPath`, `installedAt`
- [ ] `readConfig()` merges persisted config with env vars; env vars (`PARALLAX_REPO_ROOT`, `PYCOMFY_MODELS_DIR`) take precedence over stored values (backward compat for CI)
- [ ] `bun run typecheck` passes

---

### US-003: Decoupled subprocess runner (`runner.ts`)
**As a** maintainer, **I want** `spawnPipeline` to receive `repoRoot` and `uvPath` as parameters **so that** it no longer reads `process.env.PARALLAX_REPO_ROOT` directly and can work with values sourced from the config layer.

**Acceptance Criteria:**
- [ ] `src/runner.ts` exports `spawnPipeline(scriptRelPath, args, config)` accepting a resolved config object
- [ ] No direct reference to `process.env.PARALLAX_REPO_ROOT` inside `runner.ts`
- [ ] All existing CLI commands still work when `PARALLAX_REPO_ROOT` is set as env var (backward compat via `readConfig()`)

---

### US-004: Centralized model registry (`models/registry.ts`)
**As a** maintainer, **I want** all model lists, script paths, and model config in a single module **so that** adding a new model requires editing exactly one file.

**Acceptance Criteria:**
- [ ] `src/models/registry.ts` exports `MODELS`, `IMAGE_SCRIPTS`, `VIDEO_MODEL_CONFIG`, `AUDIO_SCRIPTS` (moved from `index.ts`)
- [ ] Exports `getModels(action, media): string[]`, `getScript(action, media, model): string | undefined`, `getModelConfig(media, model): ModelConfig | undefined`
- [ ] `index.ts` no longer contains model data — all model lookups go through `registry.ts`

---

### US-005: Per-media argument builders (`models/image.ts`, `models/video.ts`, `models/audio.ts`)
**As a** maintainer, **I want** model-specific argument logic isolated per media type **so that** action handlers contain no inline special-cases.

**Acceptance Criteria:**
- [ ] `src/models/image.ts` exports `buildArgs(opts, modelsDir): string[]`; handles the `z_image` omit-negative-prompt/cfg special case
- [ ] `src/models/video.ts` exports `buildArgs(opts, modelsDir): string[]`; handles `omitSteps`, `cfgFlag`, and `--image` vs `--input` mapping
- [ ] `src/models/audio.ts` exports `buildArgs(opts, modelsDir): string[]`; handles `--prompt → --tags` and `--length → --duration` mapping
- [ ] Action handlers in `commands/create.ts` call only `buildArgs()` and `spawnPipeline()` — no inline model logic

---

### US-006: Refactored command handlers (`commands/create.ts`, `commands/edit.ts`)
**As a** maintainer, **I want** action handlers reduced to validate → resolve config → buildArgs → spawnPipeline **so that** each handler is under 20 lines and free of model-specific conditionals.

**Acceptance Criteria:**
- [ ] `src/commands/create.ts` and `src/commands/edit.ts` export `registerCreate(program)` and `registerEdit(program)` respectively
- [ ] `--models-dir` resolution (flag > config > env) is handled once per handler, not duplicated
- [ ] All existing `parallax create image/video/audio` and `parallax edit image/video` commands preserve their flags, defaults, and behavior

---

### US-007: Interactive installer (`commands/install.ts`)
**As a** first-time user, **I want** to run `parallax install` and be guided through environment setup **so that** I don't need to manually configure paths and env vars.

**Acceptance Criteria:**
- [ ] `src/commands/install.ts` exports `registerInstall(program)` and registers a `install` command
- [ ] Interactive TTY flow (using `@clack/prompts`):
  - If config exists: asks for confirmation before reinstalling
  - Checks for `uv` in PATH; if not found, downloads it to `~/.local/bin`
  - Prompts for install directory (default: `~/.parallax`)
  - Runs `uv venv` and `uv sync --extra <variant>` (cuda or cpu) with spinner feedback
  - Prompts for models directory (default: `~/parallax-models`)
  - Writes config via `writeConfig()` with `installedAt` timestamp
  - Shows outro: `"Listo. Ejecuta: parallax create image --help"`
- [ ] Non-interactive mode: `parallax install --non-interactive --install-dir <path> --models-dir <path> --variant cuda|cpu` accepts all settings as flags, skips all prompts
- [ ] If `!process.stdout.isTTY` (no TTY detected), falls back to non-interactive mode automatically using flag values or defaults
- [ ] `parallax install --non-interactive` runs to completion without crash (on a machine with `uv` in PATH)

---

### US-008: Clean entry point (`index.ts`)
**As a** maintainer, **I want** `index.ts` to contain only program setup and command registration **so that** the entry point is under 20 lines.

**Acceptance Criteria:**
- [ ] `src/index.ts` imports and calls `registerInstall`, `registerCreate`, `registerEdit` only
- [ ] No model data, no inline logic, no `spawnPipeline` call in `index.ts`
- [ ] `parallax --help` still lists all commands correctly

---

### US-009: Existing tests continue to pass
**As a** maintainer, **I want** the refactor to be transparent to existing black-box tests **so that** the CLI contract is not broken.

**Acceptance Criteria:**
- [ ] `bun test` (or equivalent runner for `src/index.test.ts`) passes with zero failures
- [ ] All existing CLI invocations in the test file work unchanged with `PARALLAX_REPO_ROOT` set as env var

## Functional Requirements

- **FR-1:** `src/config.ts` — `readConfig()` merges `~/.config/parallax/config.json` (if exists) with `process.env` overrides; returns a `ParallaxConfig` object
- **FR-2:** `src/config.ts` — `writeConfig(config: ParallaxConfig)` creates `~/.config/parallax/` if absent and writes JSON atomically
- **FR-3:** `src/runner.ts` — `spawnPipeline(scriptRelPath, args, config)` resolves the full script path as `join(config.repoRoot, scriptRelPath)` and spawns `[config.uvPath ?? "uv", "run", "python", ...]`
- **FR-4:** `src/models/registry.ts` — is the single source of truth; all model validation, help-text generation, and script resolution use this module
- **FR-5:** `commands/install.ts` — detects TTY via `process.stdout.isTTY`; falls back to non-interactive when false
- **FR-6:** `commands/install.ts` — CI flags `--non-interactive`, `--install-dir`, `--models-dir`, `--variant` allow fully scripted installation
- **FR-7:** Backward compatibility — `PARALLAX_REPO_ROOT` and `PYCOMFY_MODELS_DIR` env vars continue to take precedence over stored config values in all commands
- **FR-8:** `package.json` — adds `@clack/prompts ^0.9.x` to `dependencies` and `build:linux`, `build:mac`, `build:win`, `build` scripts
- **FR-9:** File structure after refactor matches exactly the layout defined in `CLI_REFACTOR_PLAN.md` (T1–T9)

## Non-Goals (Out of Scope)

- Auto-update mechanism for the binary (not in this iteration)
- Windows binary (`build:win` script is added to `package.json` but not tested)
- Unit tests for `config.ts`, `buildArgs()`, or `install.ts` — only black-box tests required in this iteration
- Changing the public CLI interface (flags, command names, defaults) — this is a refactor, not a feature addition
- Modifying Python pipeline scripts or any code outside `packages/parallax_cli/`
- Support for Bun versions below the current stable

## Open Questions

- None
