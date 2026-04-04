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
