# Requirement: parallax CLI — ltx23 ia2v Support

## Context
The `ltx23/ia2v` pipeline already exists in `comfy_diffusion` and generates video conditioned on both an input image and an audio track. The `parallax create video` command does not yet expose this mode. Users currently have no CLI path to invoke ia2v without writing Python directly.

## Goals
- Expose `ltx23/ia2v` through `parallax create video` via a new `--audio` flag.
- Keep the command surface backwards-compatible (existing t2v and i2v invocations are unaffected).
- Guard `--audio` so it is only accepted for `ltx23` (the only model that supports it).

## User Stories

### US-001: Add `--audio` option to `create video`
**As a** developer, **I want** to pass `--audio <path>` to `parallax create video` **so that** I can generate a video conditioned on an audio track without writing Python.

**Acceptance Criteria:**
- [ ] `parallax create video --model ltx23 --prompt "..." --input image.png --audio track.wav --output out.mp4` runs without error and produces a valid `.mp4` file.
- [ ] `--audio` is optional; omitting it leaves existing t2v / i2v behaviour unchanged for all models.
- [ ] If `--audio` is supplied with a model other than `ltx23`, the CLI prints `Error: --audio is only supported for model 'ltx23'.` to stderr and exits with code 1.
- [ ] If the path given to `--audio` does not exist, the CLI prints `Error: audio file not found: <path>` to stderr and exits with code 1.
- [ ] If `--audio` is supplied without `--input`, the CLI prints `Error: --audio requires --input (image).` to stderr and exits with code 1.
- [ ] Typecheck / lint passes.

### US-002: Route to `ia2v` pipeline in the `ltx23` runner
**As a** developer, **I want** the `_ltx23` runner to call `ia2v.run()` when an audio path is provided **so that** the correct pipeline is executed end-to-end.

**Acceptance Criteria:**
- [ ] `_ltx23` accepts an `audio` keyword argument (default `None`).
- [ ] When `audio` is not `None`, `_ltx23` imports and calls `comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run()` with the correct arguments: `models_dir`, `image`, `audio_path`, `prompt`, `width`, `height`, `length`, `fps`, `cfg`, `seed`.
- [ ] When `audio` is `None` and `image` is set, the existing `i2v` path is unchanged.
- [ ] When both are `None`, the existing `t2v` path is unchanged.
- [ ] `RUNNERS["ltx23"]` signature accepts `audio=None` without breaking existing call sites (uses `**_` for unknown kwargs).
- [ ] Typecheck / lint passes.
- [ ] Visually verified in terminal: running the command produces a `.mp4` file.

## Functional Requirements
- **FR-1:** `cli/commands/create.py` — add `audio: Annotated[Optional[str], typer.Option("--audio", help="Input audio file for ia2v (ltx23 only).")]` parameter (default `None`) to `create_video`.
- **FR-2:** `cli/commands/create.py` — validate that `--audio` is only used with `--model ltx23`; error and exit otherwise.
- **FR-3:** `cli/commands/create.py` — validate that the audio file exists on disk when provided; error and exit otherwise.
- **FR-4:** `cli/commands/create.py` — validate that `--input` is also provided when `--audio` is given; error and exit otherwise.
- **FR-5:** `cli/commands/create.py` — pass `audio=audio` to `video.RUNNERS[model](...)` (the runner ignores it via `**_` for models that do not use it).
- **FR-6:** `cli/_runners/video.py` — update `_ltx23` signature to `(*, mdir, prompt, image, w, h, n, f, s, c, seed, audio=None, **_)` and add ia2v branch.
- **FR-7:** `cli/_runners/video.py` — the ia2v branch loads a PIL image from `input` path (already done by the caller), then passes `image` object + `audio_path=audio` to `ia2v.run()`.
- **FR-8:** async mode — include `audio` in the `args` dict passed to `run_async` so the job payload is complete.

## Non-Goals (Out of Scope)
- Exposing advanced `ia2v` parameters (`audio_start_time`, `audio_duration`, `guide_strength_pass1/2`, `distilled_lora_strength`, `te_lora_strength`) — use pipeline defaults.
- Adding `--audio` support to any model other than `ltx23`.
- Changes to `parallax_ms`, `parallax_mcp`, or `server/`.
- New tests (test plan is written during the Refactor phase).

## Open Questions
- None
