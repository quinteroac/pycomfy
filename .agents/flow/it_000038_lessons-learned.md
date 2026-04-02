# Lessons Learned — Iteration 000038

## US-001 — Top-level help

**Summary:** Added `create` and `edit` stub subcommands to `packages/parallax_cli/src/index.ts`, ensured `parallax` (no args) shows help and exits 0, and added a version preamble to the help output. Tests live in `packages/parallax_cli/src/index.test.ts` using Bun's built-in test runner.

**Key Decisions:**
- Commander's `--help` does not show the literal version string by default (only `-V, --version` option label). Used `.addHelpText("before", "parallax v0.1.0\n")` to surface the version in help output, satisfying US-001-AC01 literally.
- No-args help uses `program.help()` which prints to stdout and exits with code 0 — matches AC-02 exactly.
- Stub commands use `process.exit(1)` + `console.error` to signal "not yet implemented", keeping future implementation slots clean.

**Pitfalls Encountered:**
- `bun-types` was not installed when typecheck was first run (`Cannot find type definition file for 'bun-types'`). Running `bun install` in the package directory resolved it.
- Commander's default `--help` output does not embed the version number inline; the version is only accessible via `-V`/`--version`. Adding a `addHelpText("before", ...)` preamble was required to satisfy the AC.

**Useful Context for Future Agents:**
- Bun tests for the CLI spawn the CLI as a subprocess via `Bun.spawn(["bun", "run", CLI, ...args])`. This gives accurate exit-code and stdout/stderr capture.
- The `tsconfig.json` in `packages/parallax_cli` has `"include": ["src"]` which covers both `index.ts` and `index.test.ts` by default.
- `bun test` is the test runner for TypeScript packages; run from the package directory or pass the file path explicitly.
- When adding new commands in future stories, follow the same pattern: `.command("name").description(...).argument(...).option(...).action(...)`.

## US-002 — `create` subcommand help

**Summary:** Refactored the `create` command in `packages/parallax_cli/src/index.ts` to accept `<media>` as the argument with choices `["image", "video", "audio"]`, replacing the old `<prompt>` argument. Added `--prompt`, `--output`, and `--duration` flags. Added two tests for US-002 in `src/index.test.ts`.

**Key Decisions:**
- Used Commander's `Argument` class with `.choices([...])` to declare the media types. This makes Commander display `<media> (choices: "image", "video", "audio")` in the help output automatically — no manual formatting needed.
- Imported `Argument` alongside `Command` from the `commander` package (already a dependency).
- Made `--output` optional (no default) since the appropriate extension depends on the media type.
- Added `--duration` for video/audio, scoped only to this command.

**Pitfalls Encountered:**
- None significant. Commander's `Argument.choices()` API is straightforward in v12.

**Useful Context for Future Agents:**
- `parallax create --help` now outputs `Usage: parallax create [options] <media>` with the choices listed inline. The `[options]` part is generated automatically by Commander.
- When implementing the `action()` for this command, destructure `_media` as `"image" | "video" | "audio"` for type safety.
- The existing test pattern (spawning the CLI via `Bun.spawn`) works for subcommand help as well — pass `["create", "--help"]` as args.

## US-003 — `edit` subcommand help

**Summary:** Refactored the `edit` command in `packages/parallax_cli/src/index.ts` from a fixed `<image> <prompt>` argument pair to a `<media>` choice argument with options `["image", "video"]`. Added `--input`, `--prompt`, and `--output` flags. Added two US-003 tests in `src/index.test.ts`.

**Key Decisions:**
- Used the same `Argument` + `.choices([...])` pattern established by the `create` command (US-002). Commander auto-renders the choices inline in help output.
- Media types for `edit` are `image` and `video` only (no `audio`) — per the acceptance criteria.
- Removed the old positional `<image>` and `<prompt>` arguments; input path and prompt are now `--input` and `--prompt` options for consistency with the `create` command style.

**Pitfalls Encountered:**
- None. The pattern from US-002 transferred directly.

**Useful Context for Future Agents:**
- `parallax edit --help` now outputs `Usage: parallax edit [options] <media>` with choices `"image"`, `"video"` shown inline by Commander.
- When implementing the `action()` for this command, destructure `_media` as `"image" | "video"` for type safety.
- The `Argument` import was already present from the `create` command — no new imports needed.

## US-004 — Media-level help (create image, create video, create audio, edit image, edit video)

**Summary:** Refactored `create` and `edit` from single argument-based commands to nested sub-command trees (`create → image|video|audio`, `edit → image|video`). Each sub-command declares its media-specific flags and a help footer listing available models via `addHelpText("after", ...)`.

**Key Decisions:**
- Converted from `Argument("<media>").choices([...])` to proper sub-commands, which is the only way to have per-media-type flags with different sets of options.
- Used `.usage("<media> [options]")` on both the `create` and `edit` parent commands to preserve the `<media>` string in their help output, keeping existing US-002 and US-003 tests passing without any changes.
- Defined a single `MODELS` record as the source of truth for both the help footer text and future model validation (FR-4).
- Used `.requiredOption()` for `--model`, `--prompt`, and `--input` (on edit commands) so they appear with `(required)` in the generated help and commander enforces them at parse time.
- Extracted a shared `NOT_IMPLEMENTED` arrow function to avoid repeating the `console.error + process.exit(1)` stub in every action handler.

**Pitfalls Encountered:**
- `Argument` import was no longer needed after the refactor; removing it was necessary to keep the typecheck clean.
- Commander generates usage as `parallax create [options] [command]` by default for parent commands with sub-commands. The `.usage()` override is required; without it the US-002 and US-003 tests fail on `toContain("<media>")`.

**Useful Context for Future Agents:**
- The `MODELS` record (key = `"action media"`, value = `string[]`) is the canonical registry for known models per command. When implementing US-005 (model validation), import or duplicate this structure in the action handlers.
- `parallax create image --help` correctly shows all flags plus the `Available models:` footer — verified by running `bun test`.
- When US-006 (required-flag validation with custom error messages) is implemented, the `.requiredOption()` calls may need to be downgraded to `.option()` + manual checks in the action handler to produce the exact custom error format.

## US-005 — Known-model validation

**Summary:** Added `validateModel(key, model)` helper to `index.ts` that checks `opts.model` against the `MODELS` registry and prints `Error: unknown model "<value>" for <key>. Known models: ...` to stderr then exits 1. All five commands (`create image/video/audio`, `edit image/video`) were updated from `.action(NOT_IMPLEMENTED)` to `.action((opts) => { validateModel(key, opts.model); NOT_IMPLEMENTED(); })`. Six new tests added under the `parallax CLI — known-model validation (US-005)` suite.

**Key Decisions:** Validation happens inside the action handler (after Commander parses and enforces required options), not at option-definition time. This keeps Commander's built-in `--model <name>` required-flag enforcement intact and lets custom model-error logic run cleanly.

**Pitfalls Encountered:** An earlier edit accidentally dropped the `it("US-001-AC01:...` line from the test file. Verified by viewing the file after each edit. Always view the file after non-trivial edits to catch accidental deletions.

**Useful Context for Future Agents:** The `MODELS` record key format is `"action media"` (e.g. `"create image"`). The error message format is `Error: unknown model "<value>" for <action> <media>. Known models: <comma-separated list>`. Tests pass all required options (`--prompt`, `--input` where needed) so Commander's required-flag check doesn't interfere with model validation testing.

## US-006 — Required-flag validation

**Summary:** Added user-friendly error messages for missing required CLI flags (`--model`, `--prompt`, `--input`). Commander's built-in `requiredOption` already enforces the required constraint and exits with code 1; the only change needed was to reformat the error message from Commander's default `error: required option '--model <name>' not specified` to the expected `Error: --model is required`.

**Key Decisions:** Used Commander's `configureOutput({ writeErr })` on the root `program` instance with a regex transform. This single hook cascades to all subcommands so no per-subcommand changes were needed. The regex `/error: required option '(--[a-z-]+)[^']*' not specified/` captures the flag name and rewrites the message.

**Pitfalls Encountered:** None — Commander's `requiredOption` already handles exit code 1 correctly. The only work was message reformatting.

**Useful Context for Future Agents:** The `configureOutput` hook is set once on `program` and applies to all nested subcommands. If new subcommands are added with `requiredOption`, they automatically inherit the custom error format at no extra cost. The regex expects lowercase kebab-case flag names (`--[a-z-]+`), which matches all current flags.
