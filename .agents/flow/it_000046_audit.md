# Audit Report — Iteration 000046 (PRD 001)

**parallax CLI — ltx23 ia2v Support**
Generated: 2026-04-07

---

## Executive Summary

The implementation for iteration 000046 is complete and correct. All eight functional requirements are satisfied. The `--audio` flag has been added to `parallax create video`, guarded with the three required validations (wrong model, missing file, missing `--input`), passed through to the runner, included in the async args payload, and the `_ltx23` runner correctly branches to `ia2v.run()` when audio is provided.

The only deviation is that the `_ltx23` signature omits the `s` (steps) positional parameter specified in FR-6 — it is silently absorbed by `**_`, which is functionally harmless because none of the ltx23 pipelines accept a `steps` argument.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ✅ comply | `audio: Annotated[Optional[str], typer.Option("--audio", ...)] = None` added to `create_video`. |
| FR-2 | ✅ comply | `_AUDIO_SUPPORTED_MODELS = {"ltx23"}` guard; prints exact error and exits 1. |
| FR-3 | ✅ comply | `_Path(audio).is_file()` check; prints `Error: audio file not found: <path>` and exits 1. |
| FR-4 | ✅ comply | `if input is None` check; prints `Error: --audio requires --input (image).` and exits 1. |
| FR-5 | ✅ comply | `audio=audio` passed in RUNNERS call; other runners absorb it via `**_`. |
| FR-6 | ⚠️ partially_comply | `_ltx23` accepts `audio=None` and `**_`. Missing `s` (steps) in explicit signature — goes to `**_` instead. Functionally harmless. |
| FR-7 | ✅ comply | ia2v branch calls `ia2v.run(models_dir, prompt, image, audio_path, width, height, length, fps, cfg, seed)`. |
| FR-8 | ✅ comply | `args` dict in async dispatch includes `"audio": audio`. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | All CLI-level ACs implemented (parameter, backwards-compat, 3 guards, syntax clean). End-to-end (AC01) requires GPU. |
| US-002 | ✅ comply | AC01–AC06 verified in code. AC07 (visual verification) requires GPU runtime. |

---

## Minor Observations

1. The `_ltx23` signature uses `(*, mdir, prompt, image, w, h, n, f, c, seed, audio=None, **_)` — the `s` parameter from the PRD spec is absorbed by `**_`. No ltx23 pipeline accepts `steps`, so this has zero functional impact and is consistent with pre-existing ltx23 runner behaviour.
2. US-002-AC07 and US-001-AC01 (end-to-end `.mp4` generation) cannot be assessed without a GPU environment. Validate manually before merging.
3. `_AUDIO_SUPPORTED_MODELS` is a module-level set — adding future audio-capable models requires only one-line edits.
4. Validation order in `create_video` is correct: wrong model → missing file → missing input (cheapest error first).

---

## Conclusions and Recommendations

The implementation is production-ready for merge pending GPU runtime verification of the ia2v end-to-end path. FR-6's minor deviation (missing explicit `s` parameter) has no functional effect.

**Recommended Refactor phase actions:**
1. *(Optional / cosmetic)* Add `s` explicitly to `_ltx23` signature to match PRD spec and improve readability.
2. Add unit tests for the three CLI validation guards (wrong model, missing file, missing input) — these are CPU-only and can be added now.
3. GPU runtime verification of `parallax create video --model ltx23 --prompt "..." --input image.png --audio track.wav --output out.mp4`.

---

## Refactor Plan

| Priority | Action | File | Notes |
|----------|--------|------|-------|
| Low | Add `s` to `_ltx23` signature | `cli/_runners/video.py` | Cosmetic only — no functional change |
| Medium | Add unit tests for CLI audio guards | `tests/cli/` | CPU-only, fast, can be automated in CI |
| High | GPU end-to-end verification | Local | Required before PR merge |
