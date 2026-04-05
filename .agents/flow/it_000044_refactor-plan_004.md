# Refactor Plan — Iteration 000044, Pass 004

## Summary of changes

### 1. Canonicalized `IMAGE_SCRIPTS` in `@parallax/cli/src/models/registry.ts` (FR-5)
Added the two `create image` models that were present in the MCP's inline map but absent from the canonical registry:
- `flux_klein → "runtime/image/generation/flux/4b_distilled.py"`
- `qwen → "runtime/image/generation/qwen/layered_t2l.py"`

`IMAGE_SCRIPTS` is now the single source of truth for all create-image script paths, eliminating the drift risk identified by the audit.

### 2. Added `@parallax/cli` as workspace dependency (FR-4, FR-5)
Updated `packages/parallax_mcp/package.json` to declare `"@parallax/cli": "workspace:*"` in dependencies, enabling the MCP to import from the canonical registry.

### 3. Removed all inline script maps from `parallax_mcp/src/index.ts` (FR-5)
Deleted the five duplicated maps (`IMAGE_CREATE_SCRIPTS`, `IMAGE_EDIT_SCRIPTS`, `IMAGE_UPSCALE_SCRIPTS`, `VIDEO_CREATE_SCRIPTS`, `AUDIO_CREATE_SCRIPTS`) that shadowed the CLI registry.

Replaced them with direct imports from `@parallax/cli/src/models/registry`:
```ts
import { getScript, getModels, getModelConfig } from "@parallax/cli/src/models/registry";
```

All five tool handlers (`create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`) now resolve script paths via `getScript()` / `getModelConfig()`.

### 4. Added model validation before `submitJob()` (FR-4)
Each of the five inference tools now calls `getModels(action, media)` and checks whether the requested model is in the valid list. If not, it returns `{ isError: true }` immediately with a descriptive message — no silent submission with an empty script path.

### 5. Updated registry tests to match new canonical state
Two test assertions in `packages/parallax_cli/tests/models/registry.test.ts` that had been written as "not yet implemented" placeholders were updated to assert the correct (now-implemented) values for `flux_klein` and `qwen` in `IMAGE_SCRIPTS` and `getScript()`.

---

## Quality checks

| Check | Command | Result |
|---|---|---|
| Type check — `@parallax/mcp` | `cd packages/parallax_mcp && bun run typecheck` | ✅ Pass (exit 0) |
| Type check — `@parallax/cli` | `cd packages/parallax_cli && bun run typecheck` | ✅ Pass (exit 0) |
| Test suite — `@parallax/cli` | `cd packages/parallax_cli && bun test` | ✅ 406 pass / 120 fail (pre-existing failures) |

**Pre-existing test failures (unchanged):** The 120 failures were present before this refactor. They relate to `MODEL_DEFAULTS` mismatches for `ltx23` (steps expected undefined but is 20) and `z_image` (cfg expected undefined but is 7), and to unrelated test files. My changes reduced the failure count from 122 to 120 (2 newly fixed registry tests).

---

## Deviations from refactor plan

None. All four recommendations from the audit's `conclusions_and_recommendations` were implemented:

1. ✅ `@parallax/cli` added as `workspace:*` dependency in `parallax_mcp/package.json`.
2. ✅ Inline script maps replaced with `getScript()` from `@parallax/cli/models/registry`.
3. ✅ Model validation added before `submitJob()` using `getModels()` — returns `isError: true` for unknown models.
4. ✅ `flux_klein` and `qwen` create-image entries added to `IMAGE_SCRIPTS` in `registry.ts`, making them canonical.

The minor observation about `(job as any).failedReason` in `wait_for_job` was not changed — BullMQ's `Job` type does expose `failedReason` as a typed property, but accessing it requires importing the BullMQ `Job` type. This is a cosmetic concern and not part of the FR-4/FR-5 scope; it remains as-is to avoid introducing a new BullMQ type import in this targeted pass.
