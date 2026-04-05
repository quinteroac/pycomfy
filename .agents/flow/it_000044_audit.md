# Audit Report — Iteration 000044 / PRD 004

## 1. Executive Summary

The parallax_mcp async refactor is substantially complete. All five inference tools have been converted to non-blocking mode via `submitJob()`, `Bun.spawn` has been eliminated, the two new tools (`get_job_status`, `wait_for_job`) are implemented correctly, tool descriptions follow the async pattern, `bun typecheck` passes, and queue handles are properly closed.

Two functional requirements are not met: **FR-4** (model validation) is absent — invalid models are silently accepted and submitted with an empty script path — and **FR-5** (script path resolution via `getScript()` from `@parallax/cli/models/registry`) is not implemented; instead, the MCP duplicates the script registry inline, introducing drift risk. The MCP's inline maps already include `flux_klein` and `qwen` create-image entries absent from the canonical registry, confirming divergence is present.

---

## 2. Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ✅ comply | `@parallax/sdk` listed as `workspace:*` in `parallax_mcp/package.json`. |
| FR-2 | ✅ comply | `get_job_status` uses `getJobStatus()` SDK wrapper (closes queue internally); `wait_for_job` closes queue in `finally`. |
| FR-3 | ✅ comply | No `CLI_DIR` constant or `Bun.spawn` calls found in `parallax_mcp/src/index.ts`. |
| FR-4 | ❌ does_not_comply | No model validation performed. Unknown models silently fall through with empty script `""`. Neither `validateModel()` nor `getModels()` is called. `@parallax/cli` is not a declared dependency. |
| FR-5 | ❌ does_not_comply | Script resolution uses inline hardcoded maps duplicated from the CLI, not `getScript()` from `@parallax/cli/models/registry`. `@parallax/cli` is not a declared dependency. Inline maps diverge from canonical registry (`flux_klein`, `qwen` create-image paths are present in MCP but absent from `IMAGE_SCRIPTS` in the registry). |

---

## 3. Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | All five tools use `submitJob()`, return the correct `job_id/status/model` format, descriptions updated, `Bun.spawn` removed, typecheck passes. AC-06 (manual timing with Claude) accepted as manual gate. |
| US-002 | ✅ comply | `get_job_status` registered with correct schema, returns all required JSON fields, handles missing job with `isError: true`, typecheck passes. |
| US-003 | ✅ comply | `wait_for_job` has correct schema (timeout defaults to 600), polls every 2s, handles completed/failed/timeout correctly, closes queue in `finally`, typecheck passes. |
| US-004 | ✅ comply | All five inference tool descriptions end with the required text; `get_job_status` and `wait_for_job` descriptions match required text exactly. |

---

## 4. Minor Observations

- `wait_for_job` uses `(job as any).failedReason` — a type cast that bypasses BullMQ's typed API. Consider typing via BullMQ's `Job` type directly.
- `IMAGE_CREATE_SCRIPTS` in the MCP includes `flux_klein` and `qwen` mappings absent from the canonical `IMAGE_SCRIPTS` in `@parallax/cli/models/registry.ts` — a maintenance risk.
- `getScriptBase()` / `getUvPath()` helpers duplicate logic owned by the CLI's `runner.ts`. These should eventually migrate to a shared utility in `@parallax/sdk`.
- US-001-AC06 (manual timing verification with Claude) cannot be verified by static analysis.

---

## 5. Conclusions and Recommendations

The core async refactor is successful. All four user stories comply. The two failing FRs represent a design gap: the MCP was connected to `@parallax/sdk` for job submission but not to `@parallax/cli` for model validation and script resolution.

**Recommended actions:**
1. Add `@parallax/cli` as a `workspace:*` dependency in `parallax_mcp/package.json`.
2. Replace inline script maps with `getScript()` from `@parallax/cli/models/registry`.
3. Add model validation before `submitJob()` using `getModels()` from the same registry; return `isError: true` for unknown models.
4. Add the `flux_klein` and `qwen` create-image entries to `IMAGE_SCRIPTS` in the canonical registry so they are the single source of truth.

---

## 6. Refactor Plan

### Task 1 — Add `@parallax/cli` dependency to `parallax_mcp`
**File:** `packages/parallax_mcp/package.json`
- Add `"@parallax/cli": "workspace:*"` to `dependencies`.
- Run `bun install` to update lockfile.

### Task 2 — Add missing entries to canonical registry
**File:** `packages/parallax_cli/src/models/registry.ts`
- Add to `IMAGE_SCRIPTS`: `flux_klein: "runtime/image/generation/flux/4b_distilled.py"` and `qwen: "runtime/image/generation/qwen/layered_t2l.py"`.
- Verify `getScript("create", "image", "flux_klein")` and `getScript("create", "image", "qwen")` return the correct paths.

### Task 3 — Replace inline script maps with `getScript()` + `getModelConfig()`
**File:** `packages/parallax_mcp/src/index.ts`
- Remove `IMAGE_CREATE_SCRIPTS`, `IMAGE_EDIT_SCRIPTS`, `IMAGE_UPSCALE_SCRIPTS`, `VIDEO_CREATE_SCRIPTS`, `AUDIO_CREATE_SCRIPTS` constants.
- Import `{ getScript, getModelConfig, getModels }` from `@parallax/cli/models/registry`.
- In each tool handler, resolve `script` via `getScript(action, media, model)`.
- For video, use `getModelConfig("video", model)` to decide `i2v` vs `t2v`.

### Task 4 — Add model validation to all five inference tools
**File:** `packages/parallax_mcp/src/index.ts`
- At the top of each tool handler, call `getModels(action, media)` and check that `input.model` is included.
- If not found, return `{ isError: true, content: [{ type: "text", text: "Unknown model '<model>' for <action> <media>. Valid models: <list>" }] }`.

### Task 5 — typecheck & regression
- Run `bun run typecheck` in `packages/parallax_mcp`.
- Run existing tests if any exist (`bun test` in the relevant packages).
