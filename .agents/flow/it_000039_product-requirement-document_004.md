# Requirement: Audio Generation via parallax-cli (ace_step)

## Context

`parallax create audio` already exists in `packages/parallax_cli/src/index.ts` with a
stub that calls `notImplemented()` for every model. The ACE Step 1.5 text-to-audio
script already exists at `examples/audio/ace/t2a.py` with a full `argparse` interface.

The goal is to wire the CLI action handler to that script: replace `notImplemented()`
with logic that spawns `uv run python examples/audio/ace/t2a.py`, passes every
applicable flag through, and exits with the child's exit code. No new HTTP layer is
needed. The Python script must not be modified.

## Goals

- `parallax create audio --model ace_step --prompt "electronic ambient" --output out.wav`
  spawns the ACE Step script with the correct flags and produces a WAV file on disk.
- Model component filenames (`--unet`, `--vae`, `--text-encoder-1`, `--text-encoder-2`)
  can be supplied as explicit CLI flags or resolved from env vars
  (`PYCOMFY_ACE_UNET`, `PYCOMFY_ACE_VAE`, `PYCOMFY_ACE_TEXT_ENCODER_1`,
  `PYCOMFY_ACE_TEXT_ENCODER_2`).
- Extended generation parameters (`--cfg`, `--lyrics`, `--bpm`) are exposed as
  optional CLI flags and forwarded verbatim.
- Env var and repo-root resolution follow the same pattern established in PRDs 001–003
  (`PARALLAX_REPO_ROOT`, `PYCOMFY_MODELS_DIR`).

## User Stories

### US-001: ace_step audio generation via CLI

**As a** developer, **I want** to run
`parallax create audio --model ace_step --prompt "electronic ambient" --output out.wav`
**so that** the ACE Step 1.5 pipeline generates a WAV file at the given output path.

**Acceptance Criteria:**
- [ ] Running the command with a valid `PYCOMFY_MODELS_DIR` and `--prompt` spawns
      `uv run python examples/audio/ace/t2a.py` with the correct flags.
- [ ] `--prompt` is forwarded as `--tags <value>` to the subprocess.
- [ ] `--length` is forwarded as `--duration <value>`.
- [ ] `--steps`, `--seed`, and `--output` are forwarded verbatim.
- [ ] `--models-dir <path>` is forwarded to the subprocess.
- [ ] The CLI exits with the subprocess exit code (0 on success, non-zero on error).
- [ ] Typecheck / lint passes (`bun tsc --noEmit`).

### US-002: model component flag resolution

**As a** developer, **I want** to specify ACE Step model filenames as CLI flags or env
vars **so that** I can run the command without manually editing scripts or shell profiles.

**Acceptance Criteria:**
- [ ] `--unet <filename>` on `create audio` is forwarded as `--unet <filename>` to the
      subprocess; if omitted, `PYCOMFY_ACE_UNET` is used as the fallback.
- [ ] `--vae <filename>` is forwarded as `--vae <filename>`; fallback: `PYCOMFY_ACE_VAE`.
- [ ] `--text-encoder-1 <filename>` is forwarded as `--text-encoder-1 <filename>`;
      fallback: `PYCOMFY_ACE_TEXT_ENCODER_1`.
- [ ] `--text-encoder-2 <filename>` is forwarded as `--text-encoder-2 <filename>`;
      fallback: `PYCOMFY_ACE_TEXT_ENCODER_2`.
- [ ] When a model component value is resolved (from flag or env var), it is appended
      to the subprocess command; if neither is provided the subprocess itself handles
      the error (the CLI does not validate component presence).
- [ ] Typecheck / lint passes.

### US-003: extended generation flags

**As a** developer, **I want** `--cfg`, `--lyrics`, and `--bpm` on `create audio`
**so that** I can control generation quality, add lyrics, and set BPM from the CLI.

**Acceptance Criteria:**
- [ ] `--cfg <value>` is exposed as an optional flag (default `2`) and forwarded
      verbatim as `--cfg <value>` to the subprocess.
- [ ] `--lyrics <text>` is exposed as an optional flag (default empty string) and
      forwarded verbatim as `--lyrics <text>`.
- [ ] `--bpm <n>` is exposed as an optional flag (default `120`) and forwarded
      verbatim as `--bpm <n>`.
- [ ] `parallax create audio --help` lists all three new flags.
- [ ] Typecheck / lint passes.

### US-004: models-dir and repo-root resolution

**As a** developer, **I want** the CLI to resolve the models directory and repo root
automatically **so that** I don't have to repeat these paths on every command.

**Acceptance Criteria:**
- [ ] `--models-dir <path>` on `create audio` takes precedence over
      `PYCOMFY_MODELS_DIR`; if neither is set, the CLI prints
      `Error: --models-dir or PYCOMFY_MODELS_DIR is required` and exits with code 1.
- [ ] If `PARALLAX_REPO_ROOT` is not set, the CLI prints
      `Error: PARALLAX_REPO_ROOT is required` and exits with code 1.
- [ ] The resolved models path is passed as `--models-dir <path>` to the subprocess.
- [ ] Typecheck / lint passes.

## Functional Requirements

- FR-1: The `create audio` action handler in `packages/parallax_cli/src/index.ts`
  must dispatch to `examples/audio/ace/t2a.py` when `opts.model` is `ace_step`.
- FR-2: The script path must be resolved using `PARALLAX_REPO_ROOT` as the repo root.
  If unset, print `Error: PARALLAX_REPO_ROOT is required` and exit with code 1.
- FR-3: The subprocess must be spawned with `uv run python <script>` so the correct
  virtual environment is used automatically.
- FR-4: `--models-dir` resolution (FR-4a: explicit flag; FR-4b: env var
  `PYCOMFY_MODELS_DIR`). Resolved value is passed as `--models-dir <path>` to the
  subprocess. If neither is provided, exit with code 1 and a clear error.
- FR-5: Flag forwarding table for `ace_step`:

  | CLI flag            | Script flag          | Notes                        |
  |---------------------|----------------------|------------------------------|
  | `--prompt <text>`   | `--tags <text>`      | renamed                      |
  | `--length <secs>`   | `--duration <secs>`  | renamed                      |
  | `--steps <n>`       | `--steps <n>`        | verbatim                     |
  | `--cfg <value>`     | `--cfg <value>`      | new flag; default `2`        |
  | `--seed <n>`        | `--seed <n>`         | verbatim                     |
  | `--lyrics <text>`   | `--lyrics <text>`    | new flag; default `""`       |
  | `--bpm <n>`         | `--bpm <n>`          | new flag; default `120`      |
  | `--output <path>`   | `--output <path>`    | verbatim                     |
  | `--models-dir`      | `--models-dir`       | resolved (FR-4)              |
  | `--unet <file>`     | `--unet <file>`      | flag or `PYCOMFY_ACE_UNET`   |
  | `--vae <file>`      | `--vae <file>`       | flag or `PYCOMFY_ACE_VAE`    |
  | `--text-encoder-1`  | `--text-encoder-1`   | flag or `PYCOMFY_ACE_TEXT_ENCODER_1` |
  | `--text-encoder-2`  | `--text-encoder-2`   | flag or `PYCOMFY_ACE_TEXT_ENCODER_2` |

- FR-6: `stdin`, `stdout`, and `stderr` of the child process must be inherited so
  progress output is visible to the user in real time.
- FR-7: The CLI process must exit with the child's exit code.
- FR-8: New flags (`--cfg`, `--lyrics`, `--bpm`, `--unet`, `--vae`, `--text-encoder-1`,
  `--text-encoder-2`) must be added to the `create audio` command definition.
- FR-9: The implementation must reuse or follow the same subprocess helper established
  in PRDs 001–003 (`create image` / `create video`) to avoid code duplication.
- FR-10: The Python example script (`examples/audio/ace/t2a.py`) must not be modified.

## Non-Goals (Out of Scope)

- Adding `--sampler` or `--scheduler` CLI flags (the script defaults are used).
- Adding `--trim-end` as a CLI flag (the script default of 5 s is used).
- Implementing audio generation via `parallax_ms` or `server/app.py` HTTP routes.
- Adding automated tests for this iteration (written in the Refactor phase per project
  convention).

## Open Questions

None.
