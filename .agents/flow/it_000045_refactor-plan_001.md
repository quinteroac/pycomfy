# Refactor Completion Report — it_000045 (pass 001)

Generated: 2026-04-06

---

## Summary of changes

### FR-3 — PARALLAX_HOME environment variable support
Created `cli/commands/_common.py` with a single `ENV_DIR` constant that calls `_get_base_dir()` at import time. `_get_base_dir()` reads `os.environ.get("PARALLAX_HOME")` and falls back to `Path.home() / ".parallax"` when the variable is absent. All three command modules now import `ENV_DIR` from this shared module, eliminating the hardcoded `Path.home() / ".parallax" / "env"` constant that was duplicated across `install.py`, `mcp.py`, and `ms.py`.

### FR-4 — rich.progress spinners for long-running subprocess calls
- **`install.py` / `_run_step()`**: when `--verbose` is not set, wraps the blocking `subprocess.run` call inside a `rich.progress.Progress` context manager (SpinnerColumn + TextColumn, `transient=True`). The spinner displays the step name and disappears after the subprocess completes.
- **`ms.py` / `_run_service_cmd()`** (new helper): applies the same spinner pattern around `systemctl --user enable --now` (Linux) and `launchctl load` (macOS). When `--verbose` is set, raw subprocess output is shown instead.

### FR-5 — --verbose flag propagated to mcp.py and ms.py
- **`mcp.py`**: added `verbose: Annotated[bool, typer.Option("--verbose", ...)] = False` to the `install` command. When set, it logs the resolved Claude Desktop config path and confirms the write.
- **`ms.py`**: added the same `--verbose` flag to the `install` command. When set, it logs the written unit/plist file path and passes through raw subprocess output.

### Minor — deduplicate _ENV_DIR constant
The three duplicate `_ENV_DIR = Path.home() / ".parallax" / "env"` lines were replaced with a single shared `ENV_DIR` in `_common.py`. Each module now imports it as `from cli.commands._common import ENV_DIR as _ENV_DIR`, preserving the private naming convention internally.

### Type annotations cleaned up
- `_read_config` / `_write_config` in `mcp.py`: typed as `dict[str, Any]` (required `Any` import added).
- `kwargs` in `install.py` `_run_step`: typed as `dict[str, Any]` (required `Any` import added).

---

## Quality checks

| Check | Command | Outcome |
|-------|---------|---------|
| Syntax parse | `python -c "import ast; ast.parse(...)"` | Pass — all 4 files parse cleanly |
| Lint | `python -m ruff check cli/commands/_common.py install.py mcp.py ms.py` | Pass — 0 errors after auto-fix |
| Type check | `python -m mypy cli/commands/_common.py install.py mcp.py ms.py` | Pass — "Success: no issues found in 4 source files" |

`bun test` is not applicable: no unit tests exist for these modules (noted as a minor observation in the audit). No test files were added in this refactor pass.

---

## Deviations from refactor plan

- **FR-2 structural deviation (mcp.py / ms.py in separate modules)**: The audit recommended keeping the separate-module structure and updating the PRD. This refactor does not move any code between files — the module layout is unchanged, consistent with the audit's "low-risk, keep as-is" recommendation.
- **Minor observation — `_read_config` silent `JSONDecodeError` swallow**: The audit flagged this as a risk. A warning was not added in this pass because the acceptance criterion (AC04) requires returning `{}` on missing/corrupt file, and adding a `typer.echo` warning would alter the output contract. This is left for a future pass if the PRD is updated.
- **Minor observation — US-003 idempotency via file existence**: Not changed. Fixing this would require service-manager queries and is outside the scope of the FRs addressed here.
- **No unit tests written**: The audit noted the absence of tests. Writing tests was outside the explicit scope of this refactor pass (no test-writing FR was opened); this is captured as a follow-up item.
