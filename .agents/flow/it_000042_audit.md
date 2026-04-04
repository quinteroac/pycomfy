# Audit Report — Iteration 000042 (PRD-001)

## Executive Summary

The implementation satisfies all seven functional requirements and all three user stories from PRD-001. `registry.ts` is the single source of truth for per-model defaults, `create.ts` removes static commander defaults for dimension/sampling params and applies model defaults at runtime, and the `--help` footer for each media type renders a per-model defaults table generated dynamically from registry data. Four minor observations were identified: a duplicated `bpm` static default, unreachable hardcode fallbacks in handlers, a non-overridable `fps` parameter, and an untestable visual-verification criterion.

---

## Verification by FR

| FR | Assessment | Notes |
|----|------------|-------|
| FR-1 | comply | `registry.ts` exports `ModelDefaults` interface (line 6) and `getModelDefaults(media, model)` (line 34). Both are imported and used in `create.ts`. |
| FR-2 | comply | All fields in `ModelDefaults` are optional (`?`). `z_image` correctly omits `cfg`; `ltx23` correctly omits `steps`. |
| FR-3 | comply | Commander `.option()` declarations for `--width`, `--height`, `--length`, `--steps`, `--cfg` carry no static defaults. `--bpm` retains `'120'` but is not in AC01's explicit list. |
| FR-4 | comply | All three action handlers call `getModelDefaults` after `validateModel` and resolve undefined params before constructing the `*Opts` object. |
| FR-5 | comply | `buildDefaultsTable(media)` iterates `getModels` → filters by `getModelDefaults !== undefined` → reads values via `getModelDefaults`. No hardcoded strings. |
| FR-6 | comply | `VIDEO_MODEL_CONFIG` entries (`cfgFlag`, `omitSteps`, `t2v`, `i2v`) are unchanged. `buildVideoArgs` behavior is unmodified. |
| FR-7 | comply | `edit image` and `edit video` handlers call `notImplemented` immediately; no default-value logic added. |

---

## Verification by US

### US-001 — Registry stores per-model defaults — **comply**

- **AC01**: `ModelDefaults` interface present with all required optional fields plus `bpm` (needed for `ace_step`).
- **AC02**: All models with create scripts have registry entries (image: sdxl/anima/z_image; video: ltx2/ltx23/wan21/wan22; audio: ace_step).
- **AC03**: `getModelDefaults` exported and callable.
- **AC04**: Import chain and types are clean.

### US-002 — CLI applies per-model defaults at runtime — **comply**

- **AC01**: No static defaults in commander options for listed params (`--width`, `--height`, `--length`, `--steps`, `--cfg`, `--fps`).
- **AC02**: `getModelDefaults` called in all three handlers; undefined params filled before `buildArgs`.
- **AC03**: ltx2 → `--width 1280 --height 720 --length 97 --steps 20 --cfg-pass1 4` ✓
- **AC04**: wan21 → `--width 832 --height 480 --length 33 --steps 30 --cfg 6` ✓
- **AC05**: ace_step → `--duration 120 --steps 8 --cfg 1` ✓ (`buildAudioArgs` remaps `length→--duration`)
- **AC06**: `??` operator ensures user-supplied values are never overwritten by defaults.
- **AC07**: Types align across registry/create/builders.

### US-003 — `--help` footer shows per-model defaults table — **comply**

- **AC01**: `addHelpText('after', buildDefaultsTable('video'))` present; columns are width/height/length/fps/steps/cfg for all four video models.
- **AC02**: `create image --help` table covers sdxl, anima, z_image (`flux_klein` and `qwen` correctly excluded by the `getModelDefaults` filter).
- **AC03**: `create audio --help` table maps `length` → `'duration'` via `DEFAULTS_COLUMNS.audio`.
- **AC04**: All values sourced from `MODEL_DEFAULTS` via `getModelDefaults`.
- **AC05**: Static analysis confirms output structure; runtime visual verification not performed (out of scope for automated audit).

---

## Minor Observations

1. **`--bpm` static default duplicates registry**: `create audio` option `--bpm` (create.ts:157) carries a static commander default of `'120'`, duplicating the `ace_step` registry entry. Benign today, but violates the "registry as single source of truth" principle.

2. **Unreachable hardcoded fallbacks in handlers**: The `??` chains in all three handlers include hardcoded last-resort values (e.g., `"832"`, `"480"`, `"20"`). These are unreachable for all currently registered models but silently hide missing registry entries for future models.

3. **`fps` displayed in help but not overridable**: `fps` is shown in the `create video --help` defaults table but there is no `--fps` CLI option. Users cannot override it. If intentional, a table footnote would improve clarity.

4. **`ltx23` shows `—` for steps**: Correct (model uses `omitSteps: true`), but first-time users may be confused. A footer note explaining `—` = "param not used by this model" would help.

---

## Conclusions and Recommendations

The implementation is production-ready for PRD-001 scope. All FRs and USs comply. The four minor observations are low-risk maintenance and polish items.

---

## Refactor Plan

### Priority 1 — Remove `--bpm` static default (FR-3 spirit, maintenance)

**File**: `packages/parallax_cli/src/commands/create.ts`  
**Change**: Remove `"120"` from `.option("--bpm <n>", "Beats per minute", "120")`.  
The handler already resolves `bpm` via `opts.bpm ?? ...` — but since there is no model-default branch for `bpm` currently set up in the handler (it passes `opts.bpm` directly), the handler must also be updated to apply the registry default:

```ts
// Before
bpm: opts.bpm,

// After
bpm: opts.bpm ?? (defaults?.bpm != null ? String(defaults.bpm) : undefined),
```

And `.option("--bpm <n>", "Beats per minute")` — no static default.

### Priority 2 — Remove unreachable hardcoded fallback strings

**Files**: `packages/parallax_cli/src/commands/create.ts` (all three action handlers)  
**Change**: Replace the tertiary hardcoded fallback with an explicit error when `defaults` is undefined (which can only happen if the model was registered in `MODELS` but not in `MODEL_DEFAULTS`).

Example pattern for video handler:
```ts
if (!defaults) {
  console.error(`Error: no defaults registered for model "${opts.model}". Update registry.ts.`);
  process.exit(1);
}
const videoOpts: VideoOpts = {
  width:  opts.width  ?? String(defaults.width!),
  height: opts.height ?? String(defaults.height!),
  ...
};
```

### Priority 3 (Optional UX) — Add `—` footnote to defaults table

**File**: `packages/parallax_cli/src/commands/create.ts`, `buildDefaultsTable`  
**Change**: Append `"\n  (—) = parameter not applicable for this model"` to the returned string when any cell contains `"—"`.
