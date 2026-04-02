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
