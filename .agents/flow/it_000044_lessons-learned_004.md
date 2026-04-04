# Lessons Learned — Iteration 000044

## US-001 — Non-blocking inference tools

**Summary:** Replaced all five `Bun.spawn` subprocess patterns in `parallax_mcp/src/index.ts` with direct `submitJob()` calls from `@parallax/sdk/submit`. Each tool now returns `job_id: <id>\nstatus: queued\nmodel: <model>` immediately without blocking. Tool descriptions were updated to mention the job ID return. All test files were rewritten to verify the new non-blocking behavior.

**Key Decisions:**
- Followed the exact same pattern already used in `parallax_ms/src/index.ts`, which had already solved the same problem for the HTTP gateway. The MCP simply mirrors that pattern.
- Added script registries (`IMAGE_CREATE_SCRIPTS`, `IMAGE_EDIT_SCRIPTS`, `IMAGE_UPSCALE_SCRIPTS`, `VIDEO_CREATE_SCRIPTS`, `AUDIO_CREATE_SCRIPTS`) directly in `parallax_mcp/src/index.ts`, mirroring the CLI registry and `parallax_ms`. This keeps the three consumers in sync — update all three when adding new models.
- Audio arg remapping (`--prompt` → `--tags`, `--length` → `--duration`) was preserved, consistent with `parallax_cli/src/models/audio.ts` and the worker convention.
- `getScriptBase()` and `getUvPath()` follow the same env-var resolution pattern as `parallax_ms`.
- `getModelsDir(override?)` falls back through the override → `PYCOMFY_MODELS_DIR` env var → empty string (Python scripts handle missing models-dir themselves).

**Pitfalls Encountered:**
- The existing tests (from previous iterations) all asserted `Bun.spawn` was present and checked for `CLI_DIR`. All five test files needed complete rewrites. The old tests were effectively "verify we're using the wrong pattern" tests.
- `edit_image` requires model-specific arg building (qwen uses `--output-prefix` instead of `--output`, and `--image` instead of `--input`; `flux_9b_kv` gets an extra `--subject-image`). This logic was ported from `parallax_cli/src/models/image.ts::buildEditImageArgs`.
- `upscale_image` uses `--input` (not `--image`) as the primary input flag, consistent with the CLI.

**Useful Context for Future Agents:**
- The script registry in `parallax_mcp/src/index.ts` is a **third copy** of the model registry (alongside `parallax_cli/src/models/registry.ts` and `parallax_ms/src/index.ts`). When adding new models or pipelines, all three must be updated.
- `submitJob()` from `@parallax/sdk/submit` internally spawns `packages/parallax_cli/src/_run.ts <jobId>` as a detached process. The MCP calls it and returns immediately — the actual inference runs in the background worker.
- The `ParallaxJobData` interface fields are: `action`, `media`, `model`, `script` (Python path relative to `scriptBase`), `args` (Python script args), `scriptBase`, `uvPath`.
- `bun typecheck` passes cleanly on `parallax_mcp` with the new implementation.

## US-002 — get_job_status tool

**Summary:** Added `get_job_status` MCP tool to `parallax_mcp/src/index.ts`. The tool accepts `{ job_id: z.string() }`, calls `getJobStatus()` from `@parallax/sdk`, and returns a JSON-formatted text response with fields `id`, `status`, `progress`, `output`, `error`, `model`, `action`, `media`. Missing jobs return `isError: true` with "Job <id> not found". Also updated `ParallaxJobStatus` in the SDK to include the `media` field.

**Key Decisions:**
- `getJobStatus` was already implemented in `@parallax/sdk/src/status.ts` and exported from `@parallax/sdk`. Only needed to import it and register the tool.
- Added `media: string | null` to `ParallaxJobStatus` type and extracted it from `job.data` alongside the existing `model` and `action` fields. This was the only SDK change required.
- Used `isError: true` in the MCP return shape (standard MCP error convention) for the not-found case, matching AC03's exact requirement.
- Test strategy follows the same source-level structural check pattern used by all other `parallax_mcp` tests — reading `index.ts` as a string and asserting presence of key identifiers.

**Pitfalls Encountered:**
- `ParallaxJobStatus` in the SDK was missing `media`, even though `ParallaxJobData` has it. The AC requires `media` in the MCP response, so the SDK type had to be updated before writing the tool.
- `getJobStatus` needed to be imported as a named import from `@parallax/sdk` (not `@parallax/sdk/status`) because the sub-path export for `status` is not declared separately.

**Useful Context for Future Agents:**
- `getJobStatus(id)` returns `ParallaxJobStatus | null` (null = job not found). The MCP tool maps null → `isError: true`.
- `ParallaxJobStatus` now includes all fields from `ParallaxJobData` relevant to callers: `model`, `action`, `media`. Fields like `script`, `args`, `scriptBase`, `uvPath` are intentionally not exposed.
- The SDK's `getJobStatus` closes the queue connection after each call. Fine for one-shot MCP use, but inefficient for bulk polling.

## US-003 — wait_for_job tool

**Summary:** Added `wait_for_job` MCP tool to `parallax_mcp/src/index.ts`. The tool accepts `{ job_id: z.string(), timeout_seconds: z.number().optional().default(600) }`, polls `getQueue().getJob(id)` every 2 seconds until the job completes, fails, or times out. Returns `{ status, output, duration_seconds }` on success, `isError: true` with `{ status: "failed", error }` on failure, and `isError: true` with `{ status: "timeout", job_id, message }` on timeout. Queue is closed in a `finally` block.

**Key Decisions:**
- Imported `getQueue` from `@parallax/sdk` (it was already exported alongside `getJobStatus`). No SDK changes were needed.
- Used a `try/finally` block to guarantee `queue.close()` is called in all code paths (success, failure, and timeout).
- Tracked `startedAt = Date.now()` before the loop to compute `duration_seconds` on completion.
- The `getQueue()` singleton is re-used within the polling loop — only one connection is opened per `wait_for_job` call.

**Pitfalls Encountered:**
- `getQueue()` is a module-level singleton in `@parallax/sdk`. Calling `queue.close()` closes the shared instance. This is fine for one-shot MCP calls but means subsequent calls in the same process would re-create the queue. The `finally` pattern used here matches what `getJobStatus` already does internally.
- `input.timeout_seconds` can be `undefined` even with `.default(600)` in the Zod schema because the MCP input type may not apply the default at runtime; using `?? 600` as a fallback is safer.

**Useful Context for Future Agents:**
- The `getQueue()` singleton from `@parallax/sdk/src/queue.ts` uses `bunqueue` internally. Each time `close()` is called, the next `getQueue()` call still returns the same cached instance (the instance variable is NOT reset on close). Verify behavior before assuming re-connection works automatically.
- Test strategy follows the existing source-level structural check pattern: `readFileSync` the source and assert key strings are present. This avoids needing a running Redis/queue for tests.
- All 21 tests pass and `bun typecheck` is clean.

## US-004 — Update tool descriptions for agent discoverability

**Summary:** Updated all 7 tool descriptions in `packages/parallax_mcp/src/index.ts`. The 5 inference tools (`create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`) now end with the canonical suffix. `get_job_status` and `wait_for_job` have exact verbatim descriptions from the ACs.

**Key Decisions:**
- Description changes were pure string edits — no logic changes required.
- Added `tests/tool_descriptions.test.ts` to assert the exact new descriptions for all 7 tools.
- Updated 5 existing test assertions in `create_image.test.ts`, `create_video.test.ts`, `create_audio.test.ts`, `edit_image.test.ts`, and `upscale_image.test.ts` that were asserting the old `"Returns a job ID immediately"` text.

**Pitfalls Encountered:**
- The US-001 tests (in the 5 inference tool test files) each had an AC03 assertion checking for the old description text `"Returns a job ID immediately"`. These had to be updated alongside the source changes to avoid regressions. Always grep tests for any string being changed in source.

**Useful Context for Future Agents:**
- The canonical inference tool description suffix is: `Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.`
- When updating tool descriptions, remember to also update any test files that assert the old description text.
- All 180 tests pass and `bun typecheck` is clean after changes.
