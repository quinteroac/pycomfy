# Refactor Completion Report — it_000040 (pass 001)

## Summary of changes

### 1. `package.json` — FR-8 fixes
- Added `build:win` script: `bun build --compile --target=bun-windows-x64 ./src/index.ts --outfile dist/parallax-windows.exe`
- Pinned `@clack/prompts` from `latest` to `^0.9.x` (resolved to `0.9.1`)

### 2. `src/commands/install.ts` — US-007 interactive installer completion
- **Default models-dir** corrected from `~/.parallax/models` to `~/parallax-models` (per PRD spec)
- **Reinstall confirmation**: `runInteractive()` now checks `configExists()` and presents a `confirm` prompt before proceeding when an existing config is found
- **uv PATH detection**: added `detectOrInstallUv()` helper that checks `~/.local/bin/uv` first, then `Bun.which("uv")`, and finally downloads uv via the official install script (`curl -LsSf https://astral.sh/uv/install.sh | sh`) if neither location is available
- **uv environment setup**: after prompts, `runInteractive()` runs `uv venv <installDir>/.venv` and `uv sync --extra <variant>` using `@clack/prompts` spinner; errors are surfaced with cancel() and exit code 1
- **`uvPath` stored in config**: resolved `uvPath` is now passed to `applyConfig()` and persisted to `~/.config/parallax/config.json`
- **Outro message** corrected to `Listo. Ejecuta: parallax create image --help` (per PRD spec)

### 3. `src/utils.ts` — shared `resolveModelsDir`
- Extracted the duplicated `resolveModelsDir(flag?: string): string` helper into `src/utils.ts` alongside the existing `formatRequiredFlagError` utility
- Added `import { readConfig } from "./config"` to support the new export

### 4. `src/commands/create.ts` — de-duplication
- Removed local `resolveModelsDir` function
- Added `import { resolveModelsDir } from "../utils"`

### 5. `src/commands/edit.ts` — de-duplication
- Removed local `resolveModelsDir` function and the now-unused `readConfig` import
- Cleaned imports to only what is actually used (`Command`, `getModels`)

---

## Quality checks

| Check | Command | Result |
|---|---|---|
| Dependency install | `bun install` (in `packages/parallax_cli`) | ✅ Pass — `@clack/prompts@0.9.1` installed |
| TypeScript typecheck | `bun run typecheck` | ✅ Pass — 0 errors |
| Test suite | `bun test` | ✅ Pass — 277 tests pass, 0 fail across 8 files |

---

## Deviations from refactor plan

None. All five recommended refactor actions from the audit conclusions were applied:
1. `build:win` script added ✅
2. `@clack/prompts` pinned to `^0.9.x` ✅
3. `uv venv` / `uv sync` steps added to `runInteractive()` ✅
4. Default models-dir changed to `~/parallax-models` ✅
5. Outro message updated ✅
6. `resolveModelsDir()` extracted to `src/utils.ts` ✅
