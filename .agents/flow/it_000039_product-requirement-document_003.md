# Requirement: Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22)

## Context

`parallax create video` currently supports text-to-video (t2v) for four model families
(`ltx2`, `ltx23`, `wan21`, `wan22`) but has no way to provide an input image for
image-to-video (i2v) generation. The i2v pipeline scripts already exist under
`examples/video/`:

| Model  | i2v script path                         |
|--------|-----------------------------------------|
| ltx2   | `examples/video/ltx/ltx2/i2v.py`       |
| ltx23  | `examples/video/ltx/ltx23/i2v.py`      |
| wan21  | `examples/video/wan/wan21/i2v.py`       |
| wan22  | `examples/video/wan/wan22/i2v.py`       |

Each script accepts a full `argparse` interface. The goal is to add an `--input` flag to
`parallax create video` and wire the action handler to dispatch to the i2v script (instead
of the t2v script) whenever `--input` is present. No new HTTP layer is needed.

## Goals

- `parallax create video --model ltx2 --input image.png --prompt "..."` produces an MP4.
- `parallax create video --model ltx23 --input image.png --prompt "..."` produces an MP4.
- `parallax create video --model wan21 --input image.png --prompt "..."` produces an MP4.
- `parallax create video --model wan22 --input image.png --prompt "..."` produces an MP4.
- The `--input` flag selects the i2v code path; omitting it preserves the existing t2v
  behavior (implemented in PRD 002).
- Flag forwarding follows each script's exact interface (e.g. `--cfg` → `--cfg-pass1`
  for ltx2 i2v; `--steps` omitted for ltx23 i2v).

## User Stories

### US-001: ltx2 i2v via CLI

**As an** end user, **I want** to run
`parallax create video --model ltx2 --input image.png --prompt "..."` **so that**
the LTX-Video 2 i2v pipeline animates my image into an MP4.

**Acceptance Criteria:**
- [ ] Running the command with a valid `PYCOMFY_MODELS_DIR`, `--input`, and `--prompt`
      spawns `uv run python examples/video/ltx/ltx2/i2v.py` with the correct flags.
- [ ] `--input <path>` is forwarded as `--image <path>` to the subprocess.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--steps`, `--seed`, `--output`
      are forwarded verbatim.
- [ ] `--cfg` is forwarded as `--cfg-pass1` (ltx2 i2v does not accept a bare `--cfg`).
- [ ] `--models-dir <path>` is forwarded to the subprocess.
- [ ] The CLI exits with the subprocess exit code (0 on success, non-zero on error).
- [ ] Typecheck / lint passes (`bun tsc --noEmit`).

### US-002: ltx23 i2v via CLI

**As an** end user, **I want** to run
`parallax create video --model ltx23 --input image.png --prompt "..."` **so that**
the LTX-Video 2.3 distilled i2v pipeline animates my image into an MP4.

**Acceptance Criteria:**
- [ ] Running the command spawns `uv run python examples/video/ltx/ltx23/i2v.py`
      with the correct flags.
- [ ] `--input <path>` is forwarded as `--image <path>`.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--cfg`, `--seed`, `--output`
      are forwarded verbatim.
- [ ] `--steps` is **not** forwarded (ltx23 i2v is distilled and does not accept it;
      the CLI silently omits it).
- [ ] `--models-dir <path>` is forwarded.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-003: wan21 i2v via CLI

**As an** end user, **I want** to run
`parallax create video --model wan21 --input image.png --prompt "..."` **so that**
the WAN 2.1 i2v pipeline animates my image into an MP4.

**Acceptance Criteria:**
- [ ] Running the command spawns `uv run python examples/video/wan/wan21/i2v.py`
      with the correct flags.
- [ ] `--input <path>` is forwarded as `--image <path>`.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--steps`, `--cfg`, `--seed`,
      `--output` are forwarded verbatim.
- [ ] `--models-dir <path>` is forwarded.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-004: wan22 i2v via CLI

**As an** end user, **I want** to run
`parallax create video --model wan22 --input image.png --prompt "..."` **so that**
the WAN 2.2 i2v pipeline animates my image into an MP4.

**Acceptance Criteria:**
- [ ] Running the command spawns `uv run python examples/video/wan/wan22/i2v.py`
      with the correct flags.
- [ ] `--input <path>` is forwarded as `--image <path>`.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--steps`, `--cfg`, `--seed`,
      `--output` are forwarded verbatim.
- [ ] `--models-dir <path>` is forwarded.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-005: --input flag added to create video

**As an** end user, **I want** an `--input` flag on `parallax create video` **so that**
I can provide a starting image without switching to a different command.

**Acceptance Criteria:**
- [ ] `parallax create video --help` lists `--input <path>` as an optional flag.
- [ ] When `--input` is provided, the action handler dispatches to the i2v script for
      the given model.
- [ ] When `--input` is **not** provided, the action handler dispatches to the t2v script
      (existing behavior from PRD 002) — no regression.
- [ ] If `--input` is provided but the file does not exist on disk, the CLI prints
      `Error: input file not found: <path>` and exits with code 1 before spawning any
      subprocess.
- [ ] Typecheck / lint passes.

## Functional Requirements

- FR-1: Add `.option("--input <path>", "Path to the input image for image-to-video")` to
  the `create video` command definition in `packages/parallax_cli/src/index.ts`.
- FR-2: In the `create video` action handler, check `opts.input`:
  - present → dispatch to the i2v script for the given model.
  - absent  → dispatch to the t2v script (PRD 002 behavior; no change).
- FR-3: i2v script dispatch table (based on `opts.model`):
  - `ltx2`  → `examples/video/ltx/ltx2/i2v.py`
  - `ltx23` → `examples/video/ltx/ltx23/i2v.py`
  - `wan21`  → `examples/video/wan/wan21/i2v.py`
  - `wan22`  → `examples/video/wan/wan22/i2v.py`
- FR-4: Script paths must be resolved using `PARALLAX_REPO_ROOT`. If unset, print
  `Error: PARALLAX_REPO_ROOT is required` and exit with code 1.
- FR-5: Validate that the file at `opts.input` exists before spawning the subprocess.
  On failure: print `Error: input file not found: <path>` and exit with code 1.
- FR-6: `--input <path>` must be forwarded to the subprocess as `--image <path>` for
  all four models.
- FR-7: Flag forwarding rules per model for the i2v path:

  | CLI flag        | ltx2          | ltx23   | wan21   | wan22   |
  |-----------------|---------------|---------|---------|---------|
  | `--input`       | `--image`     | `--image` | `--image` | `--image` |
  | `--prompt`      | `--prompt`    | same    | same    | same    |
  | `--width`       | `--width`     | same    | same    | same    |
  | `--height`      | `--height`    | same    | same    | same    |
  | `--length`      | `--length`    | same    | same    | same    |
  | `--steps`       | `--steps`     | omit    | same    | same    |
  | `--cfg`         | `--cfg-pass1` | `--cfg` | `--cfg` | `--cfg` |
  | `--seed`        | `--seed`      | same    | same    | same    |
  | `--output`      | `--output`    | same    | same    | same    |
  | `--models-dir`  | `--models-dir`| same    | same    | same    |

- FR-8: The subprocess must be spawned with `uv run python <script>` so the correct
  virtual environment is used automatically.
- FR-9: `stdin`, `stdout`, and `stderr` of the child process must be inherited so
  progress output is visible to the user in real time.
- FR-10: The CLI process must exit with the child's exit code.
- FR-11: `--models-dir` resolution follows PRD 002 (FR-4a explicit flag, FR-4b env var
  `PYCOMFY_MODELS_DIR`). If neither is provided, exit with code 1 and a clear error.
- FR-12: The implementation must reuse or follow the same subprocess helper introduced
  in `create image` (PRD 001) and `create video` t2v (PRD 002) to avoid code duplication.

## Non-Goals (Out of Scope)

- Implementing first-last-frame (flf2v) or scene-to-video (s2v) variants.
- Adding `--negative-prompt` to `create video` (not currently part of the command
  definition; out of scope for this iteration).
- Adding `--fps` as a CLI flag (scripts use their own defaults).
- Wiring `edit video` (wan21/wan22) — remains `notImplemented`.
- Adding HTTP routes to `parallax_ms` or `server/app.py`.
- Modifying any Python pipeline or example script.
- Adding automated tests for this iteration (written in the Refactor phase per project
  convention).

## Open Questions

None.
