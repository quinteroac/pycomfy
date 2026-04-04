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
