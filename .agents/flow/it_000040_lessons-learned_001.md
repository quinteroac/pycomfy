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
