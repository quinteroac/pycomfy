# Python CLI (`python -m parallax`)

## Context

`parallax_cli` is currently a Bun/TypeScript CLI using Commander. Since all inference logic runs in Python, the CLI is a thin dispatch layer that calls `submit_job()` or `spawnPipeline()`. Migrating to Python with Typer eliminates the TS→Python subprocess boundary, lets the CLI import `server/submit.py` directly, and provides a native `python -m parallax` entry point consistent with the rest of the stack.

## Goals

- Implement a Typer-based CLI under `cli/` with the same command surface as `parallax_cli`.
- Support both sync (blocking) and `--async` (non-blocking, returns job ID) modes on all generation commands.
- Add a `parallax jobs` subcommand group with `list`, `watch`, `status`, `cancel`, and `open` sub-commands.
- Provide a `python -m parallax` entry point registered in `pyproject.toml`.

## User Stories

### US-001: Generation commands (sync mode)
**As a** developer, **I want** to run `parallax create image --model sdxl --prompt "..."` and have it block until the output file is ready **so that** I can use it in scripts that depend on the result.

**Acceptance Criteria:**
- [ ] `parallax create image`, `parallax create video`, `parallax create audio`, `parallax edit image`, `parallax upscale image` are implemented as Typer commands under `cli/commands/`.
- [ ] In sync mode (default, no `--async`), each command calls the pipeline `run()` directly (or `spawnPipeline` equivalent) and blocks until completion.
- [ ] On success, the command prints the output file path and exits with code 0.
- [ ] On failure, the command prints the error to stderr and exits with code 1.
- [ ] `--help` on each command shows all available options with descriptions.

### US-002: `--async` flag on generation commands
**As a** developer, **I want** to append `--async` to any generation command **so that** the CLI returns a job ID immediately and I can continue working without waiting for inference.

**Acceptance Criteria:**
- [ ] All five generation commands accept an `--async` flag (boolean, default False).
- [ ] When `--async` is provided, the command calls `submit_job()` from `server/submit.py` instead of blocking.
- [ ] The command prints exactly: `Job <job_id> queued\n  → parallax jobs watch <job_id>` and exits with code 0.
- [ ] When `--async` is NOT provided, behavior is identical to current sync mode.

### US-003: `parallax jobs` subcommand group
**As a** developer, **I want** a `parallax jobs` subcommand **so that** I can list, monitor, and cancel jobs from the terminal.

**Acceptance Criteria:**
- [ ] `parallax jobs list` prints a table of the 20 most recent jobs with columns: `ID`, `STATUS`, `MODEL`, `CREATED`.
- [ ] `parallax jobs status <job_id>` prints the full job record as formatted JSON.
- [ ] `parallax jobs watch <job_id>` renders a live progress bar (using `rich` or `typer`'s progress utilities) that updates until the job reaches a terminal state, then prints the output path or error.
- [ ] `parallax jobs cancel <job_id>` cancels a queued job and prints `Cancelled <job_id>` or an appropriate error.
- [ ] `parallax jobs open <job_id>` opens the output file with the OS default application (`xdg-open` / `open`).

### US-004: `python -m parallax` entry point
**As a** developer, **I want** to invoke the CLI with `python -m parallax` or `uv run parallax` **so that** I don't need a separate Bun install to use the CLI.

**Acceptance Criteria:**
- [ ] `cli/__main__.py` exists and calls `app()` from `cli/main.py`.
- [ ] `pyproject.toml` registers a `parallax` script entry point pointing to `cli.main:app`.
- [ ] `uv run parallax --help` prints the top-level help text without error.
- [ ] `uv run python -m parallax --help` also works.

---

## Functional Requirements

- FR-1: CLI is implemented with Typer; `rich` may be used for table and progress output.
- FR-2: All CLI source lives under `cli/` at repo root; `cli/commands/` contains one file per command group.
- FR-3: The CLI reads `PARALLAX_RUNTIME_DIR`, `PYCOMFY_MODELS_DIR`, and `PARALLAX_UV_PATH` env vars with the same semantics as the current TS CLI.
- FR-4: Sync mode uses `comfy_diffusion.pipelines` directly — no subprocess spawning for sync execution.
- FR-5: `--async` mode uses `submit_job()` and never imports `torch` or `comfy.*` in the CLI process.
- FR-6: All Typer commands have explicit `--help` texts for every option.

## Non-Goals

- No TUI or interactive prompt beyond `parallax jobs watch` progress bar.
- No shell completion scripts in this PRD.
- No migration of the existing Bun CLI — it remains functional during the transition.

## Open Questions

- Should `parallax jobs watch` connect to the SSE endpoint (requires gateway to be running) or poll the local SQLite DB directly? Direct DB polling avoids the dependency on the gateway being up.
