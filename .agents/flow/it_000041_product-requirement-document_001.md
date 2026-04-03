# Requirement: CLI Runtime Bundle — Move examples/ into parallax_cli/runtime/

## Context
The CLI (`@parallax/cli`) currently resolves Python pipeline scripts using paths relative to
`PARALLAX_REPO_ROOT` (e.g. `examples/image/generation/sdxl/t2i.py`). This means the CLI only
works when run from inside the source repository. To support distribution as a standalone binary,
the Python scripts need to be co-located with the CLI package, bundled into `dist/`, and copied to
a well-known user directory (`~/.config/parallax/runtime/`) during `parallax install`.

## Goals
- Decouple the CLI from `PARALLAX_REPO_ROOT` for script resolution.
- Make the compiled binary fully self-contained: `dist/parallax-<platform>` + `dist/runtime/`.
- After `parallax install`, all pipeline scripts live at `~/.config/parallax/runtime/` and the CLI
  resolves them from there via the stored `runtimeDir` config key.

## User Stories

### US-001: Move examples/ into packages/parallax_cli/runtime/
**As a** developer, **I want** the pipeline Python scripts to live inside the CLI package at
`packages/parallax_cli/runtime/` **so that** they are logically owned by the CLI and can be
bundled with the CLI build artifact.

**Acceptance Criteria:**
- [ ] Directory `packages/parallax_cli/runtime/` is created with the same subdirectory structure
      currently in `examples/` (i.e. `runtime/image/`, `runtime/video/`, `runtime/audio/`,
      `runtime/text/`).
- [ ] All Python script files from `examples/` are moved (not copied) to their equivalent paths
      under `packages/parallax_cli/runtime/`.
- [ ] The original `examples/` directory is removed (or left empty) from the repo root.
- [ ] Typecheck / lint passes.

### US-002: Update registry.ts paths to use runtime/ prefix
**As a** developer, **I want** `registry.ts` to reference scripts using `runtime/` paths instead
of `examples/` paths **so that** all path constants reflect the new location.

**Acceptance Criteria:**
- [ ] `IMAGE_SCRIPTS`, `VIDEO_MODEL_CONFIG`, `AUDIO_SCRIPTS`, and any other path constants in
      `registry.ts` are updated from `examples/...` to `runtime/...`.
- [ ] No remaining references to `examples/` exist in any TypeScript source file under
      `packages/parallax_cli/src/`.
- [ ] Typecheck / lint passes.

### US-003: runner.ts resolves scripts from runtimeDir config key
**As a** CLI user, **I want** `spawnPipeline` to resolve script paths from the installed
`runtimeDir` **so that** commands work without `PARALLAX_REPO_ROOT` after `parallax install`.

**Acceptance Criteria:**
- [ ] `ParallaxConfig` interface gains a `runtimeDir?: string` field in `config.ts`.
- [ ] `runner.ts` resolves scripts as `join(runtimeDir, scriptRelPath)` when `runtimeDir` is set
      in config.
- [ ] If `runtimeDir` is not set, `runner.ts` falls back to `join(repoRoot, scriptRelPath)` for
      backward compatibility during development.
- [ ] `spawnPipeline` prints a clear error and exits with code 1 if neither `runtimeDir` nor
      `repoRoot` is configured.
- [ ] Typecheck / lint passes.

### US-004: parallax install copies runtime/ to ~/.config/parallax/runtime/
**As a** CLI user, **I want** `parallax install` to copy the bundled `runtime/` directory to
`~/.config/parallax/runtime/` **so that** pipeline scripts are available after installation
without requiring the source repository.

**Acceptance Criteria:**
- [ ] The existing `install.ts` command is extended (not replaced) with a step that copies
      `runtime/` from the binary's sibling directory (resolved via `import.meta.dir` at dev time
      or `dirname(process.execPath)` at runtime) to `~/.config/parallax/runtime/`.
- [ ] The copy is recursive and overwrites existing files (idempotent — safe to re-run).
- [ ] After the copy, `writeConfig` stores `runtimeDir: join(homedir(), ".config", "parallax", "runtime")`.
- [ ] The interactive installer shows a spinner/log line while copying (e.g. "Copying runtime
      scripts…").
- [ ] The non-interactive path also performs the copy and logs to stdout.
- [ ] Typecheck / lint passes.

### US-005: Build scripts include runtime/ in dist/
**As a** developer, **I want** the `bun run build` scripts to copy `packages/parallax_cli/runtime/`
into `dist/runtime/` **so that** the distributed artifact is self-contained.

**Acceptance Criteria:**
- [ ] `package.json` build scripts are updated so that after compiling the binary, `runtime/` is
      copied recursively to `dist/runtime/`.
- [ ] Running `bun run build:linux` produces both `dist/parallax-linux` and `dist/runtime/` with
      all Python scripts.
- [ ] Running `bun run build:mac` produces both `dist/parallax-macos` and `dist/runtime/`.
- [ ] `dist/runtime/` is added to `.gitignore` (or already ignored via `dist/`).
- [ ] Typecheck / lint passes.

## Functional Requirements
- FR-1: `packages/parallax_cli/runtime/` mirrors the subdirectory tree currently in `examples/`
        (same relative paths for every `.py` file).
- FR-2: `registry.ts` path constants use `runtime/` prefix; no `examples/` references remain in
        any TS source.
- FR-3: `ParallaxConfig` gains `runtimeDir?: string`; `runner.ts` prefers `runtimeDir` over
        `repoRoot` for script resolution.
- FR-4: `parallax install` (both interactive and non-interactive flows) copies `runtime/`
        recursively to `~/.config/parallax/runtime/` and stores `runtimeDir` in config.
- FR-5: `dist/runtime/` is produced by all `bun run build:*` targets.
- FR-6: The copy step in `install` is idempotent — running `parallax install` multiple times does
        not corrupt the runtime directory.

## Non-Goals (Out of Scope)
- Changing the Python script logic or pipeline implementations inside `runtime/`.
- Adding new pipeline scripts or models in this iteration.
- Removing `PARALLAX_REPO_ROOT` / `repoRoot` support — it remains valid as a dev-mode fallback.
- Signing or packaging the binary as an installer (`.deb`, `.dmg`, etc.).
- Automatic updates of `runtime/` when scripts change (no hot-reload).

## Open Questions
- None
