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
