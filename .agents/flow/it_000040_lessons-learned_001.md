# Lessons Learned — Iteration 000040

## US-001 — Compiled binary distribution

**Summary:** Added `build:linux`, `build:mac`, and `build` scripts to `packages/parallax_cli/package.json` using `bun build --compile`, added `@clack/prompts` to dependencies, and ran `bun install` to lock the dependency.

**Key Decisions:**
- Used `--target=bun-linux-x64` and `--target=bun-darwin-x64` for cross-platform compilation from the same machine.
- Output binaries go to `dist/parallax-linux` and `dist/parallax-macos` (no extension on Linux binary is conventional for CLIs).
- `@clack/prompts` pinned to `latest` since no specific version was required.

**Pitfalls Encountered:**
- None significant. The `bun build --compile` command is straightforward for Bun-based CLIs.

**Useful Context for Future Agents:**
- The `dist/` directory is gitignored (standard practice); binaries are build artifacts, not committed.
- Both Linux and macOS cross-compilation work from a Linux host with Bun — no macOS machine needed.
- `bun run typecheck` runs `tsc --noEmit` — it validates all TypeScript types including imported packages, so any new dependency that ships broken types will fail this check.

## US-002 — Persistent configuration layer (`config.ts`)

**Summary:** Created `packages/parallax_cli/src/config.ts` exporting `readConfig()`, `writeConfig(config)`, and `configExists()`. Config is stored as JSON at `~/.config/parallax/config.json`. Env vars `PARALLAX_REPO_ROOT` and `PYCOMFY_MODELS_DIR` override stored values at read time.

**Key Decisions:**
- Used Node.js `fs` builtins (`existsSync`, `mkdirSync`, `readFileSync`, `writeFileSync`) and `os.homedir()` — no new dependencies needed.
- `readConfig()` uses spread merge (`{ ...stored, ...envOverrides }`) so env vars cleanly win over stored values without mutating the stored object.
- Malformed JSON in the config file is silently treated as empty (try/catch) to avoid crashing on corrupt state.

**Pitfalls Encountered:**
- `Bun.file(path).toString()` returns `"[object Object]"` — it gives a `BunFile` object, not the file contents. Use `readFileSync(path, "utf-8")` from Node's `fs` module for synchronous file reads in tests.

**Useful Context for Future Agents:**
- The config file location `~/.config/parallax/config.json` is the single source of truth — if future stories need to change this path, update it in `config.ts` only.
- `writeConfig` creates the `~/.config/parallax/` directory recursively if it doesn't exist, so callers don't need to check.
- Tests back up and restore the real `~/.config/parallax/config.json` so they won't corrupt a developer's config if run on a machine that already has one.

## US-003 — Decoupled subprocess runner (`runner.ts`)

**Summary:** Created `packages/parallax_cli/src/runner.ts` exporting `spawnPipeline(scriptRelPath, args, config)`. Updated `index.ts` to import `readConfig` from `config.ts` and `spawnPipeline` from `runner.ts`, passing the resolved config object at each call site. Removed the local `spawnPipeline` function from `index.ts`.

**Key Decisions:**
- `runner.ts` accepts `ParallaxConfig` (imported as a type) — `repoRoot` and `uvPath` come from the config object. `uvPath` defaults to `"uv"` via destructuring default.
- `index.ts` calls `readConfig()` inline at each `spawnPipeline(script, args, readConfig())` call site. This is safe because `readConfig()` is pure/idempotent.
- Backward compat is fully handled by `readConfig()` in `config.ts` which already reads `PARALLAX_REPO_ROOT` and `PYCOMFY_MODELS_DIR` from env and merges them over stored config values.

**Pitfalls Encountered:**
- When editing `index.ts` to replace the local function, an edit accidentally dropped the `});` closing and `create\n  .command("video")` lines, causing a TypeScript syntax error. Always verify the structure around multi-line replacements.
- AC03 test assertions using `.not.toContain("PARALLAX_REPO_ROOT is required")` fail when `uv` is not in PATH: Bun includes source code context in exception stack traces, and the string appears as a code excerpt inside the trace. Fix: use `.not.toMatch(/^Error: PARALLAX_REPO_ROOT is required/)` to match only the console.error output at the start of stderr, not within a stack trace body.

**Useful Context for Future Agents:**
- `runner.ts` must never reference `process.env` directly — it always receives values through the `ParallaxConfig` parameter.
- When `uv` is not in PATH, Bun throws an ENOENT exception whose stderr output begins with source-context line numbers (e.g. `"13 |  if (!repoRoot)"`), not with `"Error:"`. Use this characteristic to distinguish spawn failures from CLI validation errors in tests.
- The 11 `ace_step model component flags` test failures in `index.test.ts` are pre-existing (they test a yet-to-be-implemented story) and are unrelated to this story.

## US-004 — Centralized model registry (`models/registry.ts`)

**Summary:** Created `packages/parallax_cli/src/models/registry.ts` exporting `MODELS`, `IMAGE_SCRIPTS`, `VIDEO_MODEL_CONFIG`, `AUDIO_SCRIPTS` (moved from `index.ts`), plus three helper functions: `getModels(action, media)`, `getScript(action, media, model)`, `getModelConfig(media, model)`. Updated `index.ts` to remove all model data and import exclusively from `registry.ts`. The `VideoModelConfig` interface was renamed to `ModelConfig` per AC02.

**Key Decisions:**
- `getScript` for video always returns the `t2v` path by default; callers must inspect `getModelConfig` and check `.i2v` themselves to select the i2v script. This keeps the API simple while still exposing all the data needed.
- `VIDEO_MODEL_CONFIG` is no longer imported directly in `index.ts` — the video action uses `getModelConfig("video", model)` instead, so all data access is centralized.
- `modelsFooter` and `validateModel` helper functions in `index.ts` were updated to accept `(action, media)` params and delegate to `getModels()`.

**Pitfalls Encountered:**
- Bun 1.3.4 has a bug where `expect.stringContaining(...)` used inside `toMatchObject` corrupts the actual object property in the test's view, causing the subsequent `.toBe()` assertion on `getScript()` to receive `StringContaining "ltx2"` instead of the real string value. Workaround: avoid asymmetric matchers (`expect.stringContaining`) inside `toMatchObject` when the same constant is read again in a later test. Use explicit property access + `.toContain()` instead.

**Useful Context for Future Agents:**
- The `models/registry.ts` pattern means that adding a new model now requires only editing `registry.ts` — no changes to `index.ts` are needed for model data.
- `getModelConfig` currently only returns configs for `media === "video"`. If future media types (audio, image) gain per-model config objects, extend `getModelConfig` in `registry.ts` only.
- Do not use `expect.stringContaining()` inside `toMatchObject()` in Bun 1.3.4 tests — it causes the matcher object to leak into subsequent `.toBe()` assertions on the same property.
