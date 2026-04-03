# Requirement: Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22)

## Context

`parallax create video` already accepts all the necessary flags (`--model`, `--prompt`,
`--width`, `--height`, `--length`, `--steps`, `--cfg`, `--seed`, `--output`) but the
action handler calls `notImplemented()` for every model. The Python t2v scripts for all
four families already exist under `examples/video/`:

| Model  | Script path                               |
|--------|-------------------------------------------|
| ltx2   | `examples/video/ltx/ltx2/t2v.py`         |
| ltx23  | `examples/video/ltx/ltx23/t2v.py`        |
| wan21  | `examples/video/wan/wan21/t2v.py`         |
| wan22  | `examples/video/wan/wan22/t2v.py`         |

Each script has a full `argparse` interface that closely mirrors the CLI flags. The goal
is to wire the CLI action handler to those scripts: replace `notImplemented()` with
logic that spawns `uv run python <script>`, passes every applicable flag through, and
exits with the child's exit code. No new HTTP layer is needed.

## Goals

- `parallax create video --model ltx2 --prompt "..."` produces an MP4 on disk.
- `parallax create video --model ltx23 --prompt "..."` produces an MP4 on disk.
- `parallax create video --model wan21 --prompt "..."` produces an MP4 on disk.
- `parallax create video --model wan22 --prompt "..."` produces an MP4 on disk.
- The models directory is resolved from `PYCOMFY_MODELS_DIR` / `--models-dir`
  (consistent with the existing `create image` implementation).
- Env var and repo-root resolution follow the same pattern established in PRD 001
  (`PARALLAX_REPO_ROOT`, `PYCOMFY_MODELS_DIR`).

## User Stories

### US-001: ltx2 video generation via CLI

**As a** developer, **I want** to run `parallax create video --model ltx2 --prompt "..."` **so that** the LTX-Video 2 t2v pipeline generates an MP4 at the given output path.

**Acceptance Criteria:**
- [ ] Running the command with a valid `PYCOMFY_MODELS_DIR` and `--prompt` spawns
      `uv run python examples/video/ltx/ltx2/t2v.py` with the correct flags.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--seed`, `--output` are forwarded
      verbatim to the subprocess.
- [ ] `--cfg` is forwarded as `--cfg-pass1` (ltx2 does not accept a bare `--cfg` flag).
- [ ] `--steps` is forwarded as `--steps`.
- [ ] `--models-dir <path>` is forwarded to the subprocess.
- [ ] The CLI exits with the subprocess exit code (0 on success, non-zero on error).
- [ ] Typecheck / lint passes (`bun tsc --noEmit`).

### US-002: ltx23 video generation via CLI

**As a** developer, **I want** to run `parallax create video --model ltx23 --prompt "..."` **so that** the LTX-Video 2.3 distilled t2v pipeline generates an MP4.

**Acceptance Criteria:**
- [ ] Running the command spawns `uv run python examples/video/ltx/ltx23/t2v.py`
      with the correct flags.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--cfg`, `--seed`, `--output`
      are forwarded verbatim.
- [ ] `--steps` is **not** forwarded (ltx23 t2v is distilled and does not accept it;
      the CLI silently omits it).
- [ ] `--models-dir <path>` is forwarded.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-003: wan21 video generation via CLI

**As a** developer, **I want** to run `parallax create video --model wan21 --prompt "..."` **so that** the WAN 2.1 t2v pipeline generates an MP4.

**Acceptance Criteria:**
- [ ] Running the command spawns `uv run python examples/video/wan/wan21/t2v.py`
      with the correct flags.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--steps`, `--cfg`, `--seed`,
      `--output` are all forwarded verbatim.
- [ ] `--models-dir <path>` is forwarded.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-004: wan22 video generation via CLI

**As a** developer, **I want** to run `parallax create video --model wan22 --prompt "..."` **so that** the WAN 2.2 t2v pipeline generates an MP4.

**Acceptance Criteria:**
- [ ] Running the command spawns `uv run python examples/video/wan/wan22/t2v.py`
      with the correct flags.
- [ ] `--prompt`, `--width`, `--height`, `--length`, `--steps`, `--cfg`, `--seed`,
      `--output` are all forwarded verbatim.
- [ ] `--models-dir <path>` is forwarded.
- [ ] The CLI exits with the subprocess exit code.
- [ ] Typecheck / lint passes.

### US-005: models-dir and repo-root resolution

**As a** developer, **I want** the CLI to resolve the models directory and repo root
automatically **so that** I don't have to repeat these paths on every command.

**Acceptance Criteria:**
- [ ] `--models-dir <path>` on `create video` takes precedence over `PYCOMFY_MODELS_DIR`.
- [ ] If neither `--models-dir` nor `PYCOMFY_MODELS_DIR` is set, the CLI prints
      `Error: --models-dir or PYCOMFY_MODELS_DIR is required` and exits with code 1
      before spawning any subprocess.
- [ ] If `PARALLAX_REPO_ROOT` is not set, the CLI prints
      `Error: PARALLAX_REPO_ROOT is required` and exits with code 1.
- [ ] The resolved models path is passed as `--models-dir <path>` to the subprocess.
- [ ] Typecheck / lint passes.

## Functional Requirements

- FR-1: The `create video` action handler in `packages/parallax_cli/src/index.ts`
  must dispatch to one of four scripts based on `opts.model`:
  - `ltx2`  → `examples/video/ltx/ltx2/t2v.py`
  - `ltx23` → `examples/video/ltx/ltx23/t2v.py`
  - `wan21`  → `examples/video/wan/wan21/t2v.py`
  - `wan22`  → `examples/video/wan/wan22/t2v.py`
- FR-2: The script path must be resolved using `PARALLAX_REPO_ROOT` as the repo root.
  If unset, print `Error: PARALLAX_REPO_ROOT is required` and exit with code 1.
- FR-3: The subprocess must be spawned with `uv run python <script>` so the correct
  virtual environment is used automatically.
- FR-4: `--models-dir` resolution (FR-4a: explicit flag; FR-4b: env var
  `PYCOMFY_MODELS_DIR`). Resolved value is passed as `--models-dir <path>` to the
  subprocess. If neither is provided, exit with code 1 and a clear error.
- FR-5: `stdin`, `stdout`, and `stderr` of the child process must be inherited so
  progress output is visible to the user in real time.
- FR-6: The CLI process must exit with the child's exit code.
- FR-7: Flag forwarding rules per model:

  | CLI flag        | ltx2          | ltx23  | wan21  | wan22  |
  |-----------------|---------------|--------|--------|--------|
  | `--prompt`      | `--prompt`    | same   | same   | same   |
  | `--width`       | `--width`     | same   | same   | same   |
  | `--height`      | `--height`    | same   | same   | same   |
  | `--length`      | `--length`    | same   | same   | same   |
  | `--steps`       | `--steps`     | omit   | same   | same   |
  | `--cfg`         | `--cfg-pass1` | same   | same   | same   |
  | `--seed`        | `--seed`      | same   | same   | same   |
  | `--output`      | `--output`    | same   | same   | same   |
  | `--models-dir`  | `--models-dir`| same   | same   | same   |

- FR-8: The implementation should reuse or follow the same subprocess helper introduced
  in `create image` (PRD 001) to avoid code duplication.

## Non-Goals (Out of Scope)

- Implementing `create video` i2v, flf2v, or other variants (only t2v is in scope).
- Adding `--fps` as a CLI flag (the scripts use their own defaults; not exposed this iteration).
- Adding `--negative-prompt` to `create video` (not currently in the CLI command definition).
- Wiring `edit video` (wan21/wan22) — remains `notImplemented`.
- Adding HTTP routes to `parallax_ms` or `server/app.py`.
- Modifying any Python pipeline or example script.
- Adding automated tests for this iteration (written in the Refactor phase per project convention).

## Open Questions

None.
