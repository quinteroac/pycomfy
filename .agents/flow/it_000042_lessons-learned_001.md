# Lessons Learned — Iteration 000042

## US-001 — Registry stores per-model defaults

**Summary:** Added a `ModelDefaults` interface with optional fields (`width?`, `height?`, `length?`, `fps?`, `steps?`, `cfg?`) to `registry.ts`, a `MODEL_DEFAULTS` constant keyed by `[media][model]`, and a `getModelDefaults(media, model)` helper function. Tests were added to the existing `tests/models/registry.test.ts` file.

**Key Decisions:**
- Defaults were sourced directly from each pipeline's Python `run()` signature rather than from the CLI command defaults — the Python values are the canonical source of truth (AC02 says "from its `run()` signature").
- `z_image` has no `cfg` field in `ModelDefaults` because its `run()` uses a hardcoded `CFG 1.0` and does not expose a `cfg` parameter.
- `ltx23` has no `steps` field because its `run()` has no `steps` parameter (distilled model, matching `omitSteps: true` in `VIDEO_MODEL_CONFIG`).
- `ace_step` `length` maps to the pipeline's `duration` (seconds), which is the CLI `--length` field.
- `MODEL_DEFAULTS` is a private `const` (not exported); only the `getModelDefaults` helper is exposed.

**Pitfalls Encountered:**
- Several tests in `tests/index.test.ts` were already failing before this story due to error message changes in the runner (`PARALLAX_REPO_ROOT` wording). These are pre-existing failures unrelated to this story.
- The audio pipeline is in a subdirectory (`ace_step/v1_5/checkpoint.py`), not a flat file — do not assume the pipeline structure matches the flat `AUDIO_SCRIPTS` mapping.

**Useful Context for Future Agents:**
- `getModelDefaults` returns `undefined` for models without a `create` script (e.g., `flux_klein`, `qwen` under `create image`) — this is intentional per AC02.
- The `registry.ts` file is the single source of truth; US-001 extends it with defaults while keeping existing exports unchanged.
- The `ltx2` pipeline uses `cfg_pass1` internally, but the CLI exposes it as `--cfg` (dispatched via `cfgFlag: "--cfg-pass1"` in `VIDEO_MODEL_CONFIG`). The `ModelDefaults.cfg` stores the pass-1 value (4.0).

---

## US-002 — CLI applies per-model defaults at runtime

**Summary:** Removed static commander defaults for `--width`, `--height`, `--length`, `--steps`, and `--cfg` from all three `create` subcommands (image, video, audio) in `create.ts`. Each action handler now calls `getModelDefaults(media, model)` after validation and resolves undefined params to model defaults before constructing a typed `*Opts` object for `buildArgs`. Tests added in `tests/commands/create.test.ts`.

**Key Decisions:**
- The action handler constructs an explicit typed opts object (`ImageOpts` / `VideoOpts` / `AudioOpts`) after merging defaults, rather than mutating the raw Commander `opts`. This keeps TypeScript happy (the `*Opts` interfaces expect `string`, not `string | undefined`) and makes the logic explicit.
- Fallback values (e.g. `?? "832"`) are retained as TypeScript safety nets for models without registry entries — in practice, all models reachable through `buildArgs` have `getModelDefaults` entries. These fallbacks mirror the old commander defaults.
- `bpm` and `lyrics` retain their commander defaults (`"120"`, `""`) since the story only targets `--length`, `--steps`, `--cfg` (and `--width`, `--height` for non-audio).
- Tests in `tests/commands/create.test.ts` simulate the action handler's default-merging logic (call `getModelDefaults` then `buildArgs`) to assert the correct args array is produced without needing to spawn Commander.

**Pitfalls Encountered:**
- The `*Opts` interfaces (`VideoOpts`, `AudioOpts`, `ImageOpts`) are exported from their respective modules — importing them via `type VideoOpts` in `create.ts` requires they are already exported, which they are.
- Pre-existing 112 test failures in `tests/index.test.ts` remain unchanged — verified by confirming the pass/fail count is identical before and after the changes.

**Useful Context for Future Agents:**
- The pattern for applying defaults is: `opts.field ?? (defaults?.field != null ? String(defaults.field) : fallback)`. The `!= null` guard prevents `String(undefined)` → `"undefined"` if a model has a partial defaults entry.
- `ace_step` audio model's `length` maps to `--duration` in `buildArgs` (the flag remapping is in `audio.ts`). Defaults for `ace_step` are `length: 120, steps: 8, cfg: 1.0`.
- The `ltx2` cfg test expects `"4"` in the args (not `"4.0"`) because `String(4.0)` → `"4"` in JavaScript.

---

## US-003 — `--help` footer shows per-model defaults table

**Summary:** Added a `buildDefaultsTable(media: string): string` exported function to `create.ts` that generates a padded ASCII table from `getModelDefaults` registry data. Wired it into all three `create` subcommands via a second `.addHelpText("after", ...)` call. Also added `bpm?: number` to `ModelDefaults` and `bpm: 120` to the `ace_step` registry entry (required because AC03 explicitly lists bpm as a displayed column).

**Key Decisions:**
- `buildDefaultsTable` is exported from `create.ts` so tests can import and assert on the output string directly, without spawning Commander.
- Column sets are defined as a `Record<string, Array<[keyof ModelDefaults, string]>>` in `DEFAULTS_COLUMNS`, keyed by media. The audio table maps `length` → label `"duration"` to match the CLI flag name.
- Models without a registry entry (e.g. `flux_klein`, `qwen`) are filtered out with `filter(m => getModelDefaults(media, m) !== undefined)`, so the image table only shows sdxl, anima, and z_image.
- `bpm` was added to `ModelDefaults` (and `ace_step`'s entry) rather than hardcoded in the table, satisfying AC04's "no duplication; generated from registry.ts data" requirement.

**Pitfalls Encountered:**
- None significant. The filter-by-defined-defaults approach cleanly handles models like `flux_klein` that exist in `getModels()` but have no defaults.

**Useful Context for Future Agents:**
- `buildDefaultsTable` column widths are computed dynamically from the actual cell content, so adding new models or changing values in the registry automatically resizes the table.
- The "—" sentinel is used for absent fields (e.g. ltx23 steps, wan22 fps, z_image cfg). Tests look for this sentinel on the relevant model row.
- `cfg` values like `4.0` render as `"4"` via `String(4.0)` in JavaScript — this is expected and intentional.
