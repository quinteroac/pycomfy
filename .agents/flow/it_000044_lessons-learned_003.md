# Lessons Learned — Iteration 000044

## US-001 — Generation commands (sync mode)

**Summary:** Implemented a new Python Typer-based CLI under `cli/` at repo root with five commands: `parallax create image/video/audio`, `parallax edit image`, and `parallax upscale image`. All commands are in sync mode (blocking, calling `comfy_diffusion.pipelines` directly). Typer v0.24.1 was added as a core dependency.

**Key Decisions:**
- Created a clean `cli/` package at repo root, separate from the existing TypeScript `packages/parallax_cli/`.
- `cli/main.py` registers three sub-apps (`create`, `edit`, `upscale`); `cli/__main__.py` enables `python -m cli` invocation.
- `cli/_io.py` centralises save helpers (`save_image`, `save_video_frames`, `save_audio`, `resolve_models_dir`) — all lazily imported.
- `cli/commands/upscale.py` implements ESRGAN and latent-upscale inline using `comfy_diffusion` library primitives since no dedicated pipeline modules exist for these.
- Used `pyproject.toml` `[project.scripts]` `parallax = "cli.main:main"` for the `parallax` entry point, and added `cli*` to `[tool.setuptools.packages.find] include`.

**Pitfalls Encountered:**
1. **Typer `CliRunner` has no `mix_stderr` parameter** in v0.24.1. Tests using `mix_stderr=False` will raise `TypeError`. Use plain `CliRunner()`.
2. **Mock patching location**: When testing, always patch at the *import site* in the command module (`cli.commands.create.save_audio`), not at the source module (`cli._io.save_audio`), because `create.py` imports `save_audio` directly via `from cli._io import save_audio`.
3. **Pipeline call signature mismatches**: Several run() functions have non-obvious differences:
   - `ltx2.t2v.run()` uses `cfg_pass1` (not `cfg`); `ltx2.i2v.run()` uses `cfg`.
   - `wan22.t2v.run()` and `wan22.i2v.run()` use mixed positional + keyword-only signature with `*` separator before `models_dir` — they have no `fps` parameter.
   - `ace_step.checkpoint.run()` uses `tags=` (not `prompt=`) and `duration=` (not `length=`).
4. **Python parameter ordering**: Typer options with no default must come before those with defaults in the function signature — or use `= ...` (Typer's required sentinel) to work around this. `upscale.py` had `input` (required) after `prompt = ""` and needed `input = ...` to satisfy Python syntax.

**Useful Context for Future Agents:**
- `CliRunner` in Typer 0.24.1 mixes stdout and stderr into `result.output`; use that for all assertions.
- The existing TypeScript `packages/parallax_cli/` is a separate runtime; **do not modify it** when implementing the Python CLI.
- The `PYCOMFY_MODELS_DIR` env var is the fallback for `--models-dir`. `resolve_models_dir()` in `_io.py` handles both.
- `latent_upscale` upscale image command uses `sample()` from `comfy_diffusion.sampling` with `denoise=0.5` for the hi-res fix pass — this is intentional (50% denoise for refinement, not full generation).
- `typer.Option(...)` (with `...` as default) marks an option as required even when it appears after optional parameters in the function signature.
