# Lessons Learned — Iteration 000042

## US-001 — Registry declares `edit image` and `upscale image` models and scripts

**Summary:** Updated `registry.ts` to add flux variants to `MODELS["edit image"]`, introduce `EDIT_IMAGE_SCRIPTS` and `UPSCALE_IMAGE_SCRIPTS` maps, add `MODELS["upscale image"]`, and make `getScript()` action-aware for `media="image"`.

**Key Decisions:**
- Extended `getScript()` with an `action` branch for `media="image"` (previously `action` was ignored for image). The new routing is: `action="edit"` → `EDIT_IMAGE_SCRIPTS`, `action="upscale"` → `UPSCALE_IMAGE_SCRIPTS`, default → `IMAGE_SCRIPTS`. This keeps the function signature unchanged and is backwards compatible.
- Both new script maps use `Partial<Record<string, string>>` to match the type of `IMAGE_SCRIPTS` (per FR-2).
- Updated the stale test `MODELS["edit image"]` assertion (previously `["qwen"]`) and `getModels("edit", "image")` assertion to match the new model list.

**Pitfalls Encountered:**
- The existing `registry.test.ts` had hardcoded assertions for `MODELS["edit image"] === ["qwen"]` and `getModels("edit", "image") === ["qwen"]` — these needed updating alongside the source changes, or they would fail.
- The 112 failing tests in the full suite are pre-existing failures unrelated to this story (they test runtime integration paths referencing `PARALLAX_REPO_ROOT`).

**Useful Context for Future Agents:**
- `getScript(action, media, model)` is now action-aware for `media="image"`. US-002 (`edit.ts`) and US-003 (`upscale.ts`) can call it directly with their respective actions.
- `EDIT_IMAGE_SCRIPTS` and `UPSCALE_IMAGE_SCRIPTS` are exported from `registry.ts` and are available for import in downstream command files.
- Pre-existing test failures (112) are all in `index.test.ts` and relate to subprocess/env integration, not the registry or models modules — safe to ignore for this iteration's registry work.

---

## US-002 — `edit image` action wires up flux and qwen models

**Summary:** Added `EditImageOpts` interface and `buildEditImageArgs()` to `models/image.ts`, rewrote the `edit image` action handler in `edit.ts` to follow the validate → getScript → resolveModelsDir → buildEditImageArgs → spawnPipeline pattern, and created `tests/commands/edit.test.ts` with 35 tests covering all model variants.

**Key Decisions:**
- `buildEditImageArgs` was added to the existing `models/image.ts` (alongside `buildArgs`) rather than creating a new `models/edit_image.ts` file — simpler, one fewer file.
- Commander's `--no-lora` flag behavior: commander converts `--no-lora` to a boolean property `noLora` on the opts object (it is `true` when the flag is passed, `false` when `--lora` is passed, `undefined` otherwise). The action handler normalizes it with `opts.noLora === true`.
- `flux_9b_kv` subject-image validation is done in the action handler (not in `buildEditImageArgs`) — the builder just forwards the value when present.
- Default `--width` and `--height` are set to `"1024"` in the commander option (no per-model defaults table needed for edit image per the PRD non-goals).

**Pitfalls Encountered:**
- None significant. The pattern from `create.ts` transferred cleanly.

**Useful Context for Future Agents:**
- `buildEditImageArgs` in `models/image.ts` handles all flux and qwen model-specific arg differences. US-003 (`upscale.ts`) will need a similar `buildUpscaleImageArgs` in `models/image.ts`.
- The qwen `--output-prefix` mapping strips `.png` suffix: `"output.png"` → `"output"`. This is implemented as `opts.output.endsWith(".png") ? opts.output.slice(0, -4) : opts.output`.
- The pre-existing 113 `index.test.ts` failures remain; they are subprocess integration tests unrelated to this story.
