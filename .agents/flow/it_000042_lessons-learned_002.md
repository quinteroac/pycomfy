# Lessons Learned — Iteration 000042

## US-003 — New `upscale image` command wires up esrgan and latent_upscale

**Summary:** Added `UpscaleImageOpts` interface and `buildUpscaleImageArgs()` to `models/image.ts`, created `commands/upscale.ts` exporting `registerUpscale`, registered it in `index.ts`, and wrote 30 tests covering all acceptance criteria.

**Key Decisions:**
- `buildUpscaleImageArgs` lives in `models/image.ts` alongside `buildArgs` and `buildEditImageArgs` — no new file needed.
- `--checkpoint` is resolved from the CLI flag first, then `PYCOMFY_CHECKPOINT` env var — same pattern for `--esrgan-checkpoint` / `PYCOMFY_ESRGAN_CHECKPOINT` and `--latent-upscale-checkpoint` / `PYCOMFY_LATENT_UPSCALE_CHECKPOINT`.
- Model-specific env-var resolution happens in the action handler (not in `buildUpscaleImageArgs`) so the builder receives already-resolved values.
- `--checkpoint` is declared as `option` (not `requiredOption`) in commander to allow env var fallback; the handler validates it is non-empty before spawning.

**Pitfalls Encountered:**
- None. The pattern from `edit.ts` transferred cleanly; the main addition was the env-var resolution for multiple model-specific checkpoint flags.

**Useful Context for Future Agents:**
- US-004 (`index.ts` registers upscale) is already done within this story — `registerUpscale` is imported and called in `index.ts`.
- `buildUpscaleImageArgs` always emits `--esrgan-checkpoint` / `--latent-upscale-checkpoint` only when the value is present in the opts; the handler ensures it is always set before calling the builder.
- Pre-existing 113 `index.test.ts` failures remain; they are subprocess integration tests unrelated to this story.

## US-004 — `index.ts` registers the upscale command

**Summary:** Production code was already complete (done as part of US-003). Added 3 tests to `tests/index.test.ts` in a new `describe` block verifying the import, the `registerUpscale(program)` call, and that `parallax --help` lists `upscale`. Also updated the US-008 source-inspection test to assert `registerUpscale` is present in `index.ts`.

**Key Decisions:**
- No production changes needed — US-003 had already wired `registerUpscale` into `index.ts`.
- Tests use static source-file inspection (readFileSync) for AC01/AC02 and `runCLI(["--help"])` for AC03.

**Pitfalls Encountered:**
- `index.test.ts` has many earlier `describe` blocks also labelled "US-004" (from prior iterations). Running `--test-name-pattern "US-004"` will pick those up too; filter by `upscale command registration` to isolate the new block.

**Useful Context for Future Agents:**
- The US-008 source-inspection test was updated to expect `registerUpscale` alongside the other three register calls; keep it in sync when adding new top-level commands.
